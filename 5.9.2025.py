#5.6最终 无corr，退化成好结果
import functools


import itertools
import operator
import numpy as np
# numpy=1.21.6，1.26.1不行
# pandas=2.0.0，2,1,4不行
import scipy
import scipy.stats as stats
from scipy.stats import kstest, norm, pearsonr
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score, roc_curve, auc
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, FancyArrowPatch

# 设置全局字体
plt.rcParams['font.family'] = 'Times New Roman'
# 设置全局字号
plt.rcParams['font.size'] = 18
# 删去警告
import warnings

warnings.filterwarnings("ignore")
random_state = np.random.RandomState(0)


# 构造类
class Indices(object):
    def __init__(self, fields_and_sizes):
        self.field_sizes = []
        self.fields = []
        for i, (field, size) in enumerate(fields_and_sizes):
            self.__dict__[field] = i
            self.fields.append(field)
            self.field_sizes.append(size)

    def __len__(self):
        return len(self.fields)

class FactorialHMM(object):
    def __init__(self, params, n_steps, calculate_on_init=True):
        self.params = params
        self.n_steps = n_steps

        self.observed_type = float  # unless otherwise specified downstream

        # TOOD: assert existence of self.hidden_indices
        self.n_hidden_states = len(self.hidden_indices)
        self.n_hidden_state_space = functools.reduce(operator.mul, self.hidden_indices.field_sizes)
        self.all_hidden_states = np.array(
            list(itertools.product(*[range(size) for size in self.hidden_indices.field_sizes])))

        self.transition_matrices_tensor = [[np.ones((size, size)) for size in self.hidden_indices.field_sizes] for n in
                                           range(n_steps - 1)]

        # Preparation for multiply
        self.idx_in = np.arange(self.n_hidden_states, dtype=int).tolist()
        self.idx_out = np.zeros((self.n_hidden_states, self.n_hidden_states), dtype=int)
        for i in range(self.n_hidden_states):
            self.idx_out[i, :] = np.arange(self.n_hidden_states, dtype=int)
            self.idx_out[i, i] = int(self.n_hidden_states)
        self.idx_out = self.idx_out.tolist()

        self.initial_hidden_state_tensor = [np.ones(size) for size in self.hidden_indices.field_sizes]

        if calculate_on_init:
            # 设置transition_matrices_tensor
            for n in range(self.n_steps - 1):
                for field in range(self.n_hidden_states):
                    self.transition_matrices_tensor[n][field] = params['transition_matrices'][field, :, :]

            # 设置initial_hidden_state_tensor
            for field in range(self.n_hidden_states):
                self.initial_hidden_state_tensor[field] = params['initial_hidden_state'][field, :]

    def log_sum_exp(self, x):
        max_x = np.max(x)
        return max_x + np.log(np.sum(np.exp(x - max_x)))

    def normalize(self, submatrix):
        col_sums = np.sum(submatrix, axis=0)
        if 0 in col_sums:
            print("Warning: 0 column sum!!!")
            return np.zeros_like(submatrix)
        normalized = submatrix / col_sums[np.newaxis, :]
        return normalized

    # Matrix_n[a,b] = P(z_{n+1}=a | z_{n}=b)
    def SetTransitionSubmatrix(self, n_step, field, submatrix):
        submatrix = np.array(submatrix, dtype=float)
        submatrix = self.normalize(submatrix)
        # print("submatrix = ",submatrix)

        assert np.allclose(submatrix.sum(axis=0), 1, atol=1e-1)
        self.transition_matrices_tensor[n_step][field] = submatrix

    def SetInitialHiddenSubstate(self, field, substate):
        substate = np.array(substate, dtype=float)
        assert np.isclose(substate.sum(), 1, atol=5e-1)
        self.initial_hidden_state_tensor[field] = substate

    def MultiplyTransitionMatrixVector(self, n_step, vector, transpose=False):
        for i in range(self.n_hidden_states):
            submatrix = self.transition_matrices_tensor[n_step][i]

            vector = np.einsum(
                (submatrix.T if transpose else submatrix),
                [int(self.n_hidden_states), i],
                vector,
                self.idx_in,
                self.idx_out[i],
            )

        return vector

    def InitialHiddenStateConditional(self, initial_observed_state):
        initial_hidden_state = functools.reduce(np.kron, self.initial_hidden_state_tensor).reshape(
            self.hidden_indices.field_sizes)

        # for hidden_state in self.all_hidden_states:
        # * P(x_0|z_0)
        # initial_hidden_state[tuple(hidden_state)] *= self.obs_given_hidden[tuple(initial_observed_state) + tuple(hidden_state)]
        # initial_hidden_state[tuple(hidden_state)] *= self.GetObservedGivenHidden(initial_observed_state)[hidden_state]
        initial_hidden_state *= self.GetObservedGivenHidden(initial_observed_state, self.params['observed_weights'], 0)

        return initial_hidden_state

    # This also calculates the likelihood of the observed
    def Forward(self, observed_states):
        if len(observed_states.shape) == 1:
            observed_states = observed_states.reshape((1, -1))

        n_steps = observed_states.shape[-1]
        assert n_steps <= self.n_steps

        # Initialize
        alphas = np.ones(self.hidden_indices.field_sizes + [n_steps])
        scaling_constants = np.zeros(n_steps)

        # Alphas
        # alphas[..., 0] = self.params['initial_hidden_state']
        alphas[..., 0] = self.InitialHiddenStateConditional(observed_states[..., 0])

        scaling_constants[0] = alphas[..., 0].sum()
        alphas[..., 0] /= scaling_constants[0]

        for n_step in range(1, n_steps):
            alphas[..., n_step] = self.GetObservedGivenHidden(tuple(observed_states[..., n_step]),
                                                              self.params['observed_weights'], n_step)
            alphas[..., n_step] *= self.MultiplyTransitionMatrixVector(n_step - 1, alphas[..., n_step - 1])

            scaling_constants[n_step] = alphas[..., n_step].sum()
            alphas[..., n_step] /= scaling_constants[n_step]

        log_likelihood = np.log(scaling_constants).sum()

        return alphas, scaling_constants, log_likelihood

    def EStep(self, observed_states):
        if len(observed_states.shape) == 1:
            observed_states = observed_states.reshape((1, -1))

        n_steps = observed_states.shape[-1]
        # print('n_steps = ',n_steps)
        assert n_steps <= self.n_steps

        # Forward
        alphas, scaling_constants, log_likelihood = self.Forward(observed_states)

        # Backward
        betas = np.ones(self.hidden_indices.field_sizes + [n_steps])

        for n_step in range(n_steps - 2, -1, -1):
            vec = self.GetObservedGivenHidden(observed_states[..., n_step + 1], self.params['observed_weights'], n_step + 1)
            betas[..., n_step] = self.MultiplyTransitionMatrixVector(n_step, vec * betas[..., n_step + 1],
                                                                     transpose=True)
            betas[..., n_step] /= scaling_constants[n_step + 1]

        # Join
        gammas = alphas * betas
        # print("shape = \n",alphas.shape,"\n",betas.shape,"\n",gammas.shape)

        return alphas, betas, gammas, scaling_constants, log_likelihood

    def CollapseGammas(self, gammas, fields):
        if not isinstance(fields, list):
            fields = [fields]
        else:
            fields = list(fields)

        return gammas.sum(tuple(set(range(self.n_hidden_states + 1)) - set(fields + [self.n_hidden_states])))

    def CollapseXis(self, xis, fields):
        # Slow, use CalculateAndCollapseXis if possible!!!
        if not isinstance(fields, list):
            fields = [fields]
        else:
            fields = list(fields)

        # Shortcuts
        return xis.sum(tuple(set(range(2 * self.n_hidden_states + 1)) - \
                             set(fields + [x + self.n_hidden_states for x in fields] + [2 * self.n_hidden_states])))

    def CalculateAndCollapseXis(self, field, observed_states, alphas, betas, scaling_constants):
        if len(observed_states.shape) == 1:
            observed_states = observed_states.reshape((1, -1))

        n_steps = observed_states.shape[-1]

        field_size = self.hidden_indices.field_sizes[field]

        # Create the matrix of collapsing to field
        project = np.equal.outer(
            self.all_hidden_states[:, field],
            np.arange(field_size)).reshape(self.hidden_indices.field_sizes + [field_size])

        collapsed_xis = np.ones((field_size, field_size, n_steps - 1))
        idx_in1 = self.idx_in + [self.n_hidden_states]
        idx_in2 = self.idx_in + [self.n_hidden_states + 1]
        idx_out = [self.n_hidden_states, self.n_hidden_states + 1]

        for n_step in range(1, n_steps):
            b = self.GetObservedGivenHidden(observed_states[..., n_step],
                                            self.params['observed_weights'], n_step) * betas[..., n_step]

            bp = b[..., np.newaxis] * project

            a = alphas[..., n_step - 1]
            ap = a[..., np.newaxis] * project

            mbp = np.stack([self.MultiplyTransitionMatrixVector(n_step - 1, bp[..., i], transpose=True) for i in
                            range(field_size)], axis=-1)

            collapsed_xis[:, :, n_step - 1] = np.einsum(
                ap,
                idx_in1,
                mbp,
                idx_in2,
                idx_out
            )

            collapsed_xis[:, :, n_step - 1] /= scaling_constants[n_step]

        return collapsed_xis

    def Viterbi(self, observed_states, observed_weights, global_weight):
        import numpy as np, functools
        from scipy.stats import lognorm, norm
        print("\nWeight applied in Viterbi = \n", observed_weights)
        # ---------- 预处理 ----------
        if observed_states.ndim == 1:
            observed_states = observed_states.reshape((1, -1))
        E, T = observed_states.shape

        backp = np.zeros(self.hidden_indices.field_sizes + [self.n_hidden_states, T - 1], dtype=int)
        lls   = np.zeros(self.hidden_indices.field_sizes + [T], dtype=float)

        init        = functools.reduce(np.kron, self.initial_hidden_state_tensor).reshape(self.hidden_indices.field_sizes)
        lls[..., 0] = np.log(init + 1e-12)
        prev_ll     = lls[..., 0]

        # ---------- 递推 ----------
        for t in range(1, T):
            # —— 1) 转移部分（原逻辑完全保留） ——
            vec  = prev_ll.copy()
            argm = np.zeros(self.hidden_indices.field_sizes + [self.n_hidden_states], dtype=int)
            for i in range(self.n_hidden_states):
                logA = np.log(self.transition_matrices_tensor[t - 1][i] + 1e-12)
                idx = [slice(None)] + [None] * self.n_hidden_states
                idx[i + 1] = slice(None)
                B   = logA[tuple(idx)] + vec[np.newaxis, ...]
                vec = np.moveaxis(B.max(axis=i + 1), 0, i)
                argm[..., i] = np.moveaxis(B.argmax(axis=i + 1), 0, i)
            backp[..., :, t - 1] = argm

            # —— 2) 发射概率 ——
            # 2-a 互信息权重
            weighted_emis     = self.GetObservedGivenHidden(observed_states[..., t], observed_weights, t)
            log_weighted_emis = np.log(weighted_emis + 1e-12)

            # 2-b ##### NEW-v #####  温度系数归一化 :  P_weighted^(1/v) → 再归一化
            global_emis       = np.power(weighted_emis, 1.0 / global_weight)
            global_emis      /= global_emis.sum()
            log_global_emis   = np.log(global_emis + 1e-12)

            # —— 3) 合并得分 （用 log_global_emis） ——
            log_weighted_trans = global_weight * vec          # 打印里还要用，名字保留
            combined           = vec + log_global_emis        ##### NEW-v #####
            # print("\nLog_trans = \n", vec)

            # —— 5) 调试输出（仅 t<2） ——

            π0 = self.params.get('zero_inflation_pi', 0.99)
            max_vis = self.params.get('max_visibility', 10)
            EPS = 1e-300

            if t < 10:
                obs_view = observed_states[..., t].copy()
                log_view = obs_view.copy()
                log_view[0] = np.log(np.clip(log_view[0], 1e-12, None))  # PM10
                log_view[1] = np.log(np.clip(log_view[1], 1e-12, None))  # 风速

                print("\n")
                print(f"  t = {t}:\n")
                print("  观测值 {Log_PM10, Log_风速, 能见度, 相对湿度}：")
                print("  [%.4f, %.4f, %.4f, %.4f]" % tuple(log_view))
                print("  隐藏状态: {0:无雾霾无沙尘,1:无雾霾有沙尘,2:有雾霾无沙尘,3:有雾霾有沙尘}\n")

                # ---- 原始参数 / 权重打印 ----
                for idx_h, (z1, z2) in enumerate(self.all_hidden_states):
                    params_list = [(self.mus[z1, z2, e], self.sigmas[z1, z2, e]) for e in range(E)]
                    weight_list = [f"{w:.4f}" for w in observed_weights[z1, z2, :]]
                    print(f"  隐藏状态 {idx_h} (z1={z1},z2={z2}):")
                    print(f"    参数 (mu, sigma) = {params_list}")
                    print(f"    观测值权重 W      = {weight_list}\n")

                # ---- A) 单维概率 & 乘积（含零膨胀，不含权重） ----
                print("  ------ 单维概率与零膨胀修正 ------")
                zero_prod_list = []  # ← 用来收集 π·Πp_e
                for idx_h, (z1, z2) in enumerate(self.all_hidden_states):
                    probs_feat = np.zeros(E)
                    for e in range(E):
                        mu = self.mus[z1, z2, e]
                        sigma = max(self.sigmas[z1, z2, e], 1e-8)
                        x = obs_view[e]
                        if e in [0, 1]:  # PM10 / 风速：对数正态
                            probs_feat[e] = lognorm(s=sigma, scale=np.exp(mu)).pdf(x)
                        elif e == 2:  # 能见度：零膨胀正态
                            if (z1, z2) == (0, 0) and np.isclose(x, max_vis, atol=1e-6):
                                probs_feat[e] = π0
                            else:
                                base = norm(loc=mu, scale=sigma).pdf(x)
                                probs_feat[e] = (1 - π0) * base if (z1, z2) == (0, 0) else base
                        else:  # 相对湿度：普通正态
                            probs_feat[e] = norm(loc=mu, scale=sigma).pdf(x)
                        probs_feat[e] = max(probs_feat[e], EPS)
                    prod_prob = np.prod(probs_feat)  # π·Πp_e
                    zero_prod_list.append(prod_prob)  # ← 收集
                    print(f"    Hid {idx_h}  p=[{probs_feat[0]:.4e}, {probs_feat[1]:.4e}, "
                          f"{probs_feat[2]:.4e}, {probs_feat[3]:.4e}]  → Orig_emis*={prod_prob:.4e}")
                print("  ----------------------------------")

                # —— 保留：原始联合发射概率（含相关性、已归一化，未加权）——
                orig_emis = self.GetObservedGivenHidden(
                    observed_states[..., t],
                    np.ones_like(observed_weights),  # 权重全 1
                    t
                )
                # ---- B) 四种对数发射分数 ----
                orig_emis_star = np.array(zero_prod_list, dtype=float)  # 未归一化、无相关性
                emis_zero = orig_emis_star / orig_emis_star.sum()  # ① 仅零膨胀归一化
                log_zero_emis = np.log(emis_zero + 1e-12)

                log_joint_emis = np.log(orig_emis.flatten() + 1e-12)  # ② +Copula（GetObservedGivenHidden）

                log_weigh_emis = log_weighted_emis.flatten()  # ③ +观测权重 w
                log_glob_emis = log_global_emis.flatten()  # ④ +(1/v) 全局权重

                print(f"  归一化对数发射概率          ~  Log{{Π[P(X|Z)]}}               = {[f'{x:.4f}' for x in log_zero_emis]}")
                print(f"  归一化对数联合发射概率       ~  Log{{Corr·Π[P(X|Z)]}}          = {[f'{x:.4f}' for x in log_joint_emis]}")
                print(f"  加权归一化对数联合发射概率    ~  Log{{Corr·Π[P(X|Z)^w]}}        = {[f'{x:.4f}' for x in log_weigh_emis]}")
                print(f"  全局加权归一化对数联合发射概率 ~  (1/v)·Log{{Corr·Π[P(X|Z)^w]}}  = {[f'{x:.4f}' for x in log_glob_emis]}\n")

                # —— 转移概率输出 ——（保持不变）

                for i in range(self.n_hidden_states):
                    trans_mat = self.transition_matrices_tensor[t - 1][i]
                    log_trans = np.log(trans_mat + 1e-12)
                    print(f"  隐藏链 {i} 原始转移概率      P(z_{{{i},t}}|z_{{{i},t-1}})   = {[f'{x:.4f}' for x in trans_mat.flatten()]}")
                    print(f"  隐藏链 {i} 对数转移概率  log[P(z_{{{i},t}}|z_{{{i},t-1}}    = {[f'{x:.4f}' for x in log_trans.flatten()]}")

                # —— 汇总得分输出 ——（保留旧变量名）
                print("")
                print(f"  对数转移得分        Log_weighted_trans           = {log_trans.flatten()}")
                print(f"  对数全局加权发射得分  Log_Global_emis              = {log_global_emis.flatten()}")
                print(f"  总得分 Max ( log[P(z_{{{i}, t}}|z_{{{i}, t-1}})] + (1/V)·Σ{{W·Log[P(X|Z)]}} ) :")
                print(f"  Max ( Log_trans + Log_Glob_emis )              = {combined.flatten()}\n")
                chosen = np.unravel_index(np.argmax(combined), combined.shape)
                print(f"  Conclusion:  t={t}: 选中 state={chosen}       score = {combined.max()}\n")

            # —— 5) 更新 ——
            lls[..., t] = combined
            prev_ll     = combined

        # ---------- 回溯 ----------
        most = np.zeros((self.n_hidden_states, T), dtype=int)
        last = np.unravel_index(np.argmax(prev_ll), self.hidden_indices.field_sizes)
        for i in range(self.n_hidden_states):
            most[i, -1] = last[i]
        for t in range(T - 1, 0, -1):
            prev = most[:, t]
            for i in range(self.n_hidden_states):
                most[i, t - 1] = backp[tuple(prev.tolist() + [i, t - 1])]

        return most, backp, lls

    def DescaleAlphasBetas(self, alphas, betas, scaling_constants):
        acc_scaling_constants = np.multiply.accumulate(scaling_constants)

        acc_scaling_constants_from_back = np.multiply.accumulate(scaling_constants[::-1])[::-1]
        acc_scaling_constants_from_back[:-1] = acc_scaling_constants_from_back[1:]
        acc_scaling_constants_from_back[-1] = 1.0

        descaled_alphas = alphas * acc_scaling_constants[(np.newaxis,) * (len(betas.shape) - 1) + (slice(None),)]
        descaled_betas = betas * acc_scaling_constants_from_back[
            (np.newaxis,) * (len(betas.shape) - 1) + (slice(None),)]

        return descaled_alphas, descaled_betas

class FullDiscreteFactorialHMM(FactorialHMM):
    def __init__(self, params, n_steps, calculate_on_init=False, ):
        self.params = params  # 保存传入的参数为实例变量
        # 定义观测、隐藏变量
        self.observed_indices = Indices([['x{}'.format(i), 1] for i in range(E)])
        # First initialize the hidden and observed indices
        assert 'hidden_alphabet_size' in params.keys(), "params dictionary must contain 'hidden_alphabet_size':<alphabet size>"
        assert 'n_hidden_chains' in params.keys(), "params dictionary must contain 'n_hidden_chains':<number of hidden chains>"
        # assert 'observed_alphabet_size' in params.keys(), "params dictionary must contain 'observed_alphabet_size':<alphabet size>"
        assert 'n_observed_chains' in params.keys(), "params dictionary must contain 'n_observed_chains':<number of observed chains>"

        self.hidden_indices = self.I = Indices(
            [['z{}'.format(i), params['hidden_alphabet_size']] for i in range(params['n_hidden_chains'])])
        super().__init__(params, n_steps, calculate_on_init)

        # 定义mus、sigmas
        self.mus = np.zeros(
            (params['hidden_alphabet_size'], params['hidden_alphabet_size'], params['n_observed_chains']))
        self.sigmas = np.zeros(
            (params['hidden_alphabet_size'], params['hidden_alphabet_size'], params['n_observed_chains']))
        # print("shape of initial self.sigmas = ",self.sigmas.shape)

        assert len(self.mus) == len(self.sigmas), "mus和sigmas的长度必须相等"
        # 添加观测状态相关定义
        self.observed_indices = Indices([['x{}'.format(i), 1] for i in range(E)])
        self.n_observed_states = len(self.observed_indices)
        self.n_observed_state_space = functools.reduce(operator.mul, self.observed_indices.field_sizes)
        self.all_observed_states = np.array(
            list(itertools.product(*[range(size) for size in self.observed_indices.field_sizes])))
        # self.SetObservedGivenHidden()
        if calculate_on_init:
            self.SetObservedGivenHidden()

    def SetObservedGivenHidden(self):
        # return # Prepare the matrix of P(X=x|Z=z)
        # 初始化mus和sigmas
        self.mus = self.params['mus']
        self.sigmas = self.params['sigmas']

    def GetObservedGivenHidden(self, observed_state, observed_weights, n_step):
        """
        计算联合密度 P(x_t | z_t) ——
          • 保留边缘分布（对数/零膨胀对数正态，含权重与归一化）
          • 通过 Gaussian Copula 引入四个观测量之间的相关性
          • 若相关矩阵非正定或求逆失败，自动回退为“独立”模型
        所有原变量名、打印与注释均保持不变，只插入了数值保护逻辑。
        """
        import numpy as np
        from scipy.stats import lognorm, norm

        EPS       = 1e-300        # 最小概率，防止 log(0)
        MIN_SIGMA = 1e-2          # σ 下限
        MAX_QUAD  = 50            # ‖z‖²·(R⁻¹−I) 截断防溢出
        K, _, E   = self.mus.shape

        x_vec     = np.asarray(observed_state, dtype=float)
        log_unnorm = np.full(self.hidden_indices.field_sizes, -np.inf)

        # —— 内部工具：特征归一化常数 ln Z —— #
        def log_norm_feat(w, mu, sigma):
            w = max(w, 1e-8)
            if np.isclose(w, 1.0):
                return 0.0
            return ((1 - w) * np.log(sigma)
                    + ((1 - w) / 2) * np.log(2 * np.pi)
                    - 0.5 * np.log(w)
                    + mu + sigma ** 2 / (2 * w))

        # —— 遍历所有隐藏状态 (z1,z2) —— #
        for (z1, z2) in self.all_hidden_states:
            hid  = (z1, z2)
            wvec = observed_weights[z1, z2, :]

            logpdf, logZ, uvec = np.zeros(E), np.zeros(E), np.zeros(E)

            # —— 1) 逐特征边缘 —— #
            for e in range(E):
                mu    = self.mus[hid][e]
                sigma = max(self.sigmas[hid][e], MIN_SIGMA)
                obs   = x_vec[e]

                if e in [0, 1, 3]:          # 普通对数正态
                    pdf = lognorm(s=sigma, scale=np.exp(mu)).pdf(obs)
                    cdf = lognorm(s=sigma, scale=np.exp(mu)).cdf(obs)
                    logpdf[e] = np.log(max(pdf, EPS))
                    logZ[e]   = log_norm_feat(wvec[e], mu, sigma)
                    uvec[e]   = np.clip(cdf, 1e-9, 1 - 1e-9)

                else:                       # 能见度：零膨胀对数正态
                    π0      = self.params['zero_inflation_pi']
                    max_vis = self.params.get('max_visibility', 10)
                    if hid == (0, 0):       # 仅晴朗状态零膨胀
                        if obs == max_vis:
                            pdf, cdf = π0, π0
                        else:
                            pdf_c = lognorm(s=sigma, scale=np.exp(mu)).pdf(obs)
                            cdf_c = lognorm(s=sigma, scale=np.exp(mu)).cdf(obs)
                            pdf = (1 - π0) * pdf_c
                            cdf = π0 + (1 - π0) * cdf_c
                        logpdf[e] = np.log(max(pdf, EPS))
                        ln_c  = log_norm_feat(wvec[e], mu, sigma)
                        logZ[e] = np.logaddexp(wvec[e]*np.log(π0),
                                               wvec[e]*np.log(1-π0) + ln_c)
                        uvec[e] = np.clip(cdf, 1e-9, 1 - 1e-9)
                    else:                   # 其它状态仍用对数正态
                        pdf = lognorm(s=sigma, scale=np.exp(mu)).pdf(obs)
                        cdf = lognorm(s=sigma, scale=np.exp(mu)).cdf(obs)
                        logpdf[e] = np.log(max(pdf, EPS))
                        logZ[e]   = log_norm_feat(wvec[e], mu, sigma)
                        uvec[e]   = np.clip(cdf, 1e-9, 1 - 1e-9)

            # —— 2) Copula 密度 —— #
            # === ★★ 新增：相关矩阵退化开关 ★★ ===
            if self.params.get('force_corr_identity', False):
                R = np.eye(E)                           # ← 总是单位矩阵
            else:
                R = self.params['corr_mats'][z1, z2]

            if not np.allclose(R, np.eye(E)):
                try:
                    z      = np.clip(norm.ppf(uvec), -8.0, 8.0)   # 尾部裁剪
                    sign, logdet = np.linalg.slogdet(R)
                    if sign <= 0 or np.isnan(logdet):
                        raise np.linalg.LinAlgError
                    invR   = np.linalg.inv(R)
                    quad   = z @ (invR - np.eye(E)) @ z
                    quad   = np.clip(quad, -MAX_QUAD, MAX_QUAD)
                    log_cop = -0.5 * logdet - 0.5 * quad
                except Exception as err:
                    print(f"[Warn] Copula fallback ({z1},{z2}):", err)
                    log_cop = 0.0
            else:
                log_cop = 0.0

            # —— 3) 合成 —— #
            log_p = (wvec * logpdf).sum() + log_cop - logZ.sum()
            log_unnorm[hid] = log_p

        # —— 4) log-sum-exp 归一化 —— #
        M = np.max(log_unnorm)
        probs = np.exp(log_unnorm - M)
        probs /= probs.sum()



        return probs

    # 根据发射矩阵抽样
    def DrawObservedGivenHidden(self, hidden_states, n_step, random_state):
        observed_state_t = np.zeros((E,))  # 创建一个形状为(E, 1)的数组，用于存储观测状态

        # 从输入的hidden_states中获取a和b
        # print('shape of hidden_states = ',hidden_states.shape)
        a = hidden_states[0,]
        b = hidden_states[1,]

        # 分别抽样E个观测值，每个观测值均值和标准差分别是a * (i+1) 和 b * (i+1)，其中i从0到(E-1)
        for i in range(E):
            mean_i = ((a + 0.5 * b + 1) * (i + 1) * (1 + 1 / (n_step + 1))) / (4 * E)
            std_dev_i = ((b + 0.5 * a + 1) * (i + 1) * (1 + 1 / (n_step + 1))) / (4 * E)

            # 使用正态分布随机抽样
            observed_state_t[i,] = random_state.normal(loc=mean_i, scale=std_dev_i)

        return observed_state_t

    def MStep(self, observed_states, alphas, betas, gammas, scaling_constants):
        """
        ---------- EM 算法 M 步 ( corr 与 Z 无关版本 ) ----------
        * 0  常量与形状处理
        * 1  初始分布 / 转移矩阵                    —— 保留旧实现
        * 2  发射分布参数 μ、σ                    —— 保留旧实现
        * 2-B 绑定 (1,0) / (0,1) 能见度参数       —— 保留旧实现
        * 3  单一全局相关矩阵 R_global            —— 与 Z 无关，每轮重新估计
        * 4  返回参数字典
        """
        # ===== 0. 通用导入与常量 =====
        import numpy as np
        from numpy.linalg import eigh
        from scipy.stats import norm
        from scipy.optimize import root

        MIN_SIG  = 1e-2
        MIN_SIG2 = MIN_SIG ** 2
        EIG_EPS  = 1e-6
        LOG_EPS  = 1e-6
        max_vis  = self.params.get('max_visibility', 10)

        # ---- 形状校正 ----
        if observed_states.ndim == 1:
            observed_states = observed_states.reshape(1, -1)      # (E,T)
        if observed_states.shape[0] < observed_states.shape[1]:
            X = observed_states                                   # (E,T)
        else:
            X = observed_states.T                                 # 转置为 (E,T)
        E, T = X.shape

        K   = self.params['hidden_alphabet_size']
        HN  = self.params['n_hidden_chains']
        π0  = self.params.get('zero_inflation_pi', 0.99)

        # ===== 1. 初始分布与转移矩阵 =====
        initial_hidden_state_est = np.zeros((self.n_hidden_states, K))
        transition_est           = np.zeros((self.n_hidden_states, K, K))
        for chain in range(self.params['n_hidden_chains']):
            xis = self.CalculateAndCollapseXis(
                chain, observed_states, alphas, betas, scaling_constants)
            gms = self.CollapseGammas(gammas, chain)
            initial_hidden_state_est[chain] = gms[:, 0] / gms[:, 0].sum()
            transition_est[chain] = (
                xis.sum(-1) / gms[:, :-1].sum(1)[:, None]
            ).T

        # ===== 2. 发射分布参数 μ、σ =====
        self.mus       = np.zeros((K, K, E))
        self.sigmas    = np.zeros_like(self.mus)
        self.corr_mats = np.zeros((K, K, E, E))      # 先占位

        for e in range(E):
            obs = X[e, :]
            for a in range(K):
                for b in range(K):
                    γ = gammas[a, b, :]
                    if γ.sum() == 0:
                        self.mus[a, b, e]    = self.params['mus'][a, b, e]
                        self.sigmas[a, b, e] = self.params['sigmas'][a, b, e]
                        continue

                    # ----- A) 对数正态：PM10 / 风速 / 相对湿度 -----
                    if e in [0, 1, 3]:
                        mask  = obs > 0
                        γ_pos = γ[mask]
                        y     = np.log(obs[mask])
                        S     = γ_pos.sum()
                        mu    = (γ_pos * y).sum() / S
                        var   = (γ_pos * (y - mu) ** 2).sum() / S

                    # ----- B) 能见度 (e == 2) -----
                    else:
                        if (a, b) == (0, 0):          # 截尾 + 零膨胀
                            mask_cont = (obs > 0) & (obs < max_vis)
                            γ_cont    = γ[mask_cont]
                            y_cont    = np.log(obs[mask_cont])
                            S_cont    = γ_cont.sum()

                            mu0  = (γ_cont * y_cont).sum() / S_cont
                            var0 = (γ_cont * (y_cont - mu0) ** 2).sum() / S_cont

                            def equations(vars):
                                mu, log_s2 = vars
                                s2    = np.exp(log_s2)
                                sigma = np.sqrt(s2)
                                alpha = (np.log(max_vis) - mu) / sigma
                                Phiα  = norm.cdf(alpha)
                                φα    = norm.pdf(alpha)

                                dlnF_dmu = -φα / (Phiα * sigma)
                                dlnF_ds2 = φα / Phiα * (-(np.log(max_vis) - mu) /
                                                        (2 * sigma ** 3))

                                dQ_dmu = ((γ_cont * (y_cont - mu)) / s2).sum() \
                                         - S_cont * dlnF_dmu
                                dQ_ds2 = ((γ_cont * ((y_cont - mu) ** 2 / s2 - 1)) /
                                          (2 * s2)).sum() - S_cont * dlnF_ds2
                                return [dQ_dmu, dQ_ds2]

                            sol = root(equations, [mu0, np.log(var0)])
                            mu  = sol.x[0]
                            var = np.exp(sol.x[1])
                        else:                            # 普通对数正态
                            mask  = obs > 0
                            γ_pos = γ[mask]
                            y     = np.log(obs[mask])
                            S     = γ_pos.sum()
                            mu    = (γ_pos * y).sum() / S
                            var   = (γ_pos * (y - mu) ** 2).sum() / S

                    var = max(var, MIN_SIG2)
                    self.mus[a, b, e]    = mu
                    self.sigmas[a, b, e] = np.sqrt(var)

        # # ===== 2-B 绑定雾霾 (1,0) & 沙尘 (0,1) 能见度参数 =====
        # # # -------- Option A (默认) : 加权平均 ---------- #
        # S_h, S_d = gammas[1, 0].sum(), gammas[0, 1].sum()
        # if S_h + S_d > 0:
        #     μh, μd = self.mus[1, 0, 2], self.mus[0, 1, 2]
        #     σh2, σd2 = self.sigmas[1, 0, 2]**2, self.sigmas[0, 1, 2]**2
        #     μc  = (S_h * μh + S_d * μd) / (S_h + S_d)
        #     σc2 = (S_h * σh2 + S_d * σd2) / (S_h + S_d)
        #     σc  = np.sqrt(max(σc2, MIN_SIG2))
        #     self.mus[1, 0, 2] = self.mus[0, 1, 2] = μc
        #     self.sigmas[1, 0, 2] = self.sigmas[0, 1, 2] = σc

        # # -------- Option B : 简单算术平均（若需要请启用） ---------- #
        # μ̄ = (self.mus[0, 1, 2] + self.mus[1, 0, 2]) / 2
        # σ̄ = (self.sigmas[0, 1, 2] + self.sigmas[1, 0, 2]) / 2
        # for (i, j) in [(0, 1), (1, 0)]:
        #     self.mus[i, j, 2]    = μ̄
        #     self.sigmas[i, j, 2] = σ̄

        # ===== 3. 全局相关矩阵 R_global (与 Z 无关) =====
        # ---- 3-A 计算残差 ----
        X_log_all = np.log(np.clip(X, LOG_EPS, None))              # (E,T)
        #   μ̂_t = Σ_{ab} γ_ab(t) * μ_ab  →  shape (E,T)
        mu_hat = np.einsum('abe,abt->et', self.mus, gammas)
        R_e    = X_log_all - mu_hat                                 # 残差 (E,T)

        #   权重 w_t = Σ_{ab} γ_ab(t)  (形状 (T,))
        w_t    = gammas.sum(axis=(0, 1))
        W_sum  = w_t.sum()
        # ---- 3-B 加权协方差 ----
        cov = (R_e * w_t) @ R_e.T / W_sum                           # (E,E)

        # ---- 3-C → 相关矩阵，正定修补 ----
        std = np.sqrt(np.diag(cov))
        std[std < MIN_SIG] = MIN_SIG
        R_global = cov / std[:, None] / std[None, :]

        R_global = np.clip(0.5 * (R_global + R_global.T), -0.999, 0.999)
        vals, _  = eigh(R_global)
        if vals.min() < EIG_EPS:
            R_global += np.eye(E) * (EIG_EPS - vals.min() + 1e-8)

        if self.params.get('force_corr_identity', False):
            R_global = np.eye(E)

        # ---- 3-D 广播到所有 (a,b) ----
        self.corr_mats = np.tile(R_global, (K, K, 1, 1)).reshape(K, K, E, E)

        # ===== 4. 返回 =====
        return {
            'hidden_alphabet_size': K,
            'n_hidden_chains'     : HN,
            'n_observed_chains'   : E,
            'observed_weights'    : self.params['observed_weights'],
            'initial_hidden_state': initial_hidden_state_est,
            'transition_matrices' : transition_est,
            'mus'                 : self.mus,
            'sigmas'              : self.sigmas,
            'corr_mats'           : self.corr_mats,
            'zero_inflation_pi'   : π0,
            'force_corr_identity' : self.params.get('force_corr_identity', False),
        }

    def EM(self,
           observed_states,
           likelihood_precision,
           n_iterations=100,
           verbose=True,
           print_every=1,
           random_seed=None):
        """
        EM 迭代：
          · 每轮打印：迭代号、对数似然、转移矩阵、μ、σ
          · 若需要同时查看协相关矩阵，在 print_corr = True 处改为 True
        """

        old_log_likelihood = -np.inf
        n_iter             = 0
        print_corr         = False          # 如要打印 corr_mats 改成 True

        # 用当前 params 创建模型
        H = FullDiscreteFactorialHMM(self.params, self.n_steps, True)

        while n_iter < n_iterations:
            # ---------- E 步 ----------
            alphas, betas, gammas, scaling_constants, log_likelihood = \
                H.EStep(observed_states)



            # ─── 新增：统计 MAP 到 (1,1) 的时刻数 ─────────
            K = gammas.shape[0]
            T = gammas.shape[2]
            flat = gammas.reshape(K*K, T)
            idx_11 = np.ravel_multi_index((1, 1), (K, K))
            count_t = int((np.argmax(flat, axis=0) == idx_11).sum())
            print(f"Iter {n_iter:02d} → γ_(1,1) mass = {gammas[1,1,:].sum():.4f},"
                  f" #MAP→(1,1) = {count_t}")

            # ---------- M 步 ----------
            new_params = H.MStep(
                observed_states, alphas, betas, gammas, scaling_constants
            )

            # ---------- 打印 ----------
            if verbose and (n_iter % print_every == 0):
                np.set_printoptions(precision=4, suppress=True)
                print("\n===================== Iteration {:d} =====================".format(n_iter))
                print("Log-Likelihood  : {:.6f}".format(log_likelihood))
                print("\nTransition mats :\n", new_params['transition_matrices'])
                print("\nμ (mus)         :\n", new_params['mus'])
                print("\nσ^2 (sigmas^2)      :\n", new_params['sigmas'])
                print("\nCorrelation metrics:\n", new_params['corr_mats'])
                print("===========================================================\n")

            # ---------- 收敛判定 ----------
            if np.abs(log_likelihood - old_log_likelihood) < likelihood_precision:
                print("[Converged]  ΔLL = {:.6e}".format(
                    log_likelihood - old_log_likelihood))
                return (new_params['transition_matrices'],
                        new_params['mus'],
                        new_params['sigmas'],
                        new_params)

            old_log_likelihood = log_likelihood
            n_iter += 1

            # ---------- 用新参数重新实例化模型 ----------
            H = FullDiscreteFactorialHMM(new_params, self.n_steps, True)

        # 达到迭代上限
        print("[MaxIter]  final LL = {:.6f}".format(old_log_likelihood))
        return (new_params['transition_matrices'],
                new_params['mus'],
                new_params['sigmas'],
                new_params)


# 读取数据
def Get_observed_and_hidden_state(location):
    """
    加载并解析 Excel 中的观测数据和隐藏标签，返回：
      - observed_states: numpy 数组，形状 (E, T)，四个物理量的原始观测值（第2列已+1处理）
      - hidden_states:  一维 int 数组，原始隐藏标签(0/1/2)
      - encoded_hidden_states: 二维 int 数组，解码后的隐藏状态，形状 (2, T)
      - months, years:  一维 int 数组，提取的月份和年份，形状 (T,)
    """
    import pandas as pd
    import numpy as np

    df = pd.read_excel(location, skiprows=0)

    observed_states = []
    hidden_states   = []
    months = []
    years  = []

    for idx, row in df.iterrows():
        # 1) 解析时间，提取月、年
        try:
            date_string = row[0]
            day, month, year = map(int, date_string.split()[0].split('.'))
            months.append(month)
            years.append(year)
        except Exception as e:
            print(f"时间解析错误: 第 {idx+1} 行 - {e}")
            continue

        # 2) 读第3–6列的四维观测数据
        new_row = []
        skip_row = False
        for i, x in enumerate(row[2:6]):
            if pd.isna(x):
                skip_row = True
                print(f"第 {idx+1} 行存在空值，跳过该行")
                break
            value = float(x)
            # if i == 1:   # 第二列（风速）索引为1
            #     value += 1.0
            new_row.append(value)
        if skip_row:
            continue
        observed_states.append(new_row)

        # 3) 读第7列隐藏状态（0=晴，1=雾霾，2=沙尘）
        hidden_states.append(int(row[6]))

    # 转成 numpy
    observed_states = np.array(observed_states, dtype=float)  # 形状 (T, 4)
    hidden_states   = np.array(hidden_states,   dtype=int)    # 形状 (T,)

    # 编码隐藏为两条链 (0→[0,0],1→[1,0],2→[0,1])
    encoded_hidden_states = hidden_encoded(hidden_states)     # 形状 (2, T)

    # 转置观测，返回 (E, T)
    return (
        observed_states.T,
        hidden_states,
        encoded_hidden_states,
        np.array(months, dtype=int),
        np.array(years,  dtype=int)
    )

# 隐藏值解码
def hidden_encoded(hidden_states):
    # 初始化一个形状为 (len(hidden_states), 2) 的数组，填充0
    encoded_array = np.zeros((len(hidden_states), 2), dtype=int)
    # 遍历hidden_states，根据其值填充encoded_array
    for i, state in enumerate(hidden_states):
        if state == 0:
            encoded_array[i] = [0, 0]
        elif state == 1:
            encoded_array[i] = [1, 0]
        elif state == 2:
            encoded_array[i] = [0, 1]
    return encoded_array.T

# 观测值权重444
def Get_observed_weights(observed_states, hidden_states, params):
    """
    计算晴朗、雾霾、沙尘三种状态下各观测特征的互信息权重，并构造 (2,2,E) 权重矩阵。
    """
    from sklearn.feature_selection import mutual_info_classif
    import numpy as np

    # 转置观测矩阵：每行一个样本，每列一个特征
    X = observed_states.T  # shape (T, E)
    E = params['n_observed_chains']

    # 定义各状态标签映射：state_name -> hidden label
    state_labels = {
        'clear': 0,   # 晴朗
        'haze':  1,   # 雾霾
        'dust':  2    # 沙尘
    }

    # 计算每个状态的互信息，并归一化、放大至和为 E
    mi_weights = {}
    for name, label in state_labels.items():
        y = (hidden_states == label).astype(int)
        mi = mutual_info_classif(X, y)
        mi = mi / mi.sum()       # 归一化到和为1
        mi = mi * E              # 放大至和为E
        mi_weights[name] = mi

    # 生成 sigle_weights：4行分别是 clear、全1、haze、dust
    sigle_weights = np.zeros((4, E))
    sigle_weights[0] = mi_weights['clear']       # 晴朗
    sigle_weights[1] = np.ones(E)                # 无权重 (和为E)
    sigle_weights[2] = mi_weights['haze']        # 雾霾
    sigle_weights[3] = mi_weights['dust']        # 沙尘

    # print("\nsigle weights:\n Clear, No weight, Haze, Dust (sum each row=E)\n", sigle_weights)

    # 构造 (2×2×E) 权重矩阵：
    # [0,0]=clear, [1,0]=haze, [0,1]=dust, [1,1]=no weight
    weights = np.zeros((2, 2, E))
    weights[0, 0] = sigle_weights[0]
    weights[1, 0] = sigle_weights[2]
    weights[0, 1] = sigle_weights[3]
    weights[1, 1] = sigle_weights[1]

    # print("\nFinal weights = \n", weights)
    return sigle_weights, weights

# 初始转移矩阵
def initial_transition_matrix(params, histr_hidden_states):
    # 创建一个全零的(M, K, K)数组
    transition_matrix_real = np.zeros(
        (params['n_hidden_chains'], params['hidden_alphabet_size'], params['hidden_alphabet_size']))

    # 遍历 histr_hidden_states 中的每一对连续状态
    for i in range(len(histr_hidden_states) - 1):
        current_value = histr_hidden_states[i]  # 当前状态
        next_value = histr_hidden_states[i + 1]  # 下一个状态

        # 更新转移矩阵
        if current_value == 0:
            if next_value != 1:
                transition_matrix_real[0, 0, 0] += 1
            else:
                transition_matrix_real[0, 0, 1] += 1
            if next_value != 2:
                transition_matrix_real[1, 0, 0] += 1
            else:
                transition_matrix_real[1, 0, 1] += 1
        elif current_value == 1:
            if next_value != 1:
                transition_matrix_real[0, 1, 0] += 1
            else:
                transition_matrix_real[0, 1, 1] += 1
            if next_value != 2:
                transition_matrix_real[1, 0, 0] += 1
            else:
                transition_matrix_real[1, 0, 1] += 1
        elif current_value == 2:
            if next_value != 1:
                transition_matrix_real[0, 0, 0] += 1
            else:
                transition_matrix_real[0, 0, 1] += 1
            if next_value != 2:
                transition_matrix_real[1, 1, 0] += 1
            else:
                transition_matrix_real[1, 1, 1] += 1

    # 归一化处理
    for i in range(params['n_hidden_chains']):
        for j in range(params['hidden_alphabet_size']):
            sum_values = np.sum(transition_matrix_real[i, j])
            if sum_values != 0:
                transition_matrix_real[i, j] /= sum_values
    for i in range(params['n_hidden_chains']):
        transition_matrix_real[i, ...] = transition_matrix_real[i, ...].T

    return transition_matrix_real

# 初始发射矩阵
def initial_ObservedGivenHidden_matrix(params, observed_states):
    """
    原版：对数正态+零膨胀初始化观测给定隐藏状态下的 μ 和 σ。
    修改：只对三个簇进行聚类，第四个簇均值赋值0，方差0.0001，排序使用新的 hidden_state_order，并与原版保持一致的 sigma 下限处理（var 下限）。
    返回：mus, sigmas
    """
    import numpy as np
    from sklearn.cluster import KMeans
    import itertools

    K = params['hidden_alphabet_size']
    E = params['n_observed_chains']
    num_clusters = 3  # 仅聚类三个簇
    # 控制打印格式
    np.set_printoptions(precision=4, suppress=True)

    # 初始化容器
    mus = np.zeros((K, K, E))
    sigmas = np.zeros_like(mus)

    # 1. 对数正态特征：链索引 0,1,3
    for e in [0, 1, 3]:
        data = observed_states[:, e]
        data = np.clip(data, a_min=1e-6, a_max=None)
        log_data = np.log(data).reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(log_data)
        labels = kmeans.labels_
        for cid in range(num_clusters):
            vals = log_data[labels == cid, 0]
            mu = vals.mean()
            var = vals.var(ddof=0)
            # 保持原版 sigma 下限：var 下限 1e-4
            var = max(var, 1e-4)
            var = round(var, 5)
            sigma = np.sqrt(var)
            i, j = divmod(cid, K)
            mus[i, j, e] = mu
            sigmas[i, j, e] = sigma

    # 2. 能见度（链索引2）：零膨胀对数正态
    vals2 = observed_states[:, 2]
    pi0 = params.get('zero_inflation_pi', 0.9)
    max_vis = params.get('max_visibility', 10)
    mask_peak = (vals2 == max_vis)
    cont_vals = vals2[~mask_peak]
    log_cont = np.log(np.clip(cont_vals, 1e-6, None)).reshape(-1, 1)
    kmeans2 = KMeans(n_clusters=num_clusters, random_state=0).fit(log_cont)
    labels2 = kmeans2.labels_
    centers = kmeans2.cluster_centers_.flatten()

    hidden_state_order = {
        (0, 0): [3, 2, 1, 2],
        (1, 0): [2, 3, 3, 1],
        (0, 1): [1, 1, 2, 3],
    }
    temp_mu2 = np.zeros((K, K))
    temp_sigma2 = np.zeros((K, K))
    for cid in range(num_clusters):
        mask_c = (labels2 == cid)
        if mask_c.sum() > 0:
            x = log_cont[mask_c, 0]
            mu_c = x.mean()
            var_c = x.var(ddof=0)
        else:
            mu_c = np.log(max_vis)
            var_c = 1e-4
        # 保持原版 sigma 下限：var_c 下限 1e-4
        var_c = max(var_c, 1e-4)
        var_c = round(var_c, 5)
        sigma_c = np.sqrt(var_c)
        # 根据中心排序匹配到 (i,j)
        for (i, j), order in hidden_state_order.items():
            if np.argsort(-centers)[order[2] - 1] == cid:
                temp_mu2[i, j] = mu_c
                temp_sigma2[i, j] = sigma_c
    mus[:, :, 2] = temp_mu2
    sigmas[:, :, 2] = temp_sigma2

    # 3. 跨链统一簇映射（修正后）
    # 仅对 (0,0), (1,0), (0,1) 三个状态做统一簇映射
    state_list = list(hidden_state_order.keys())  # [(0,0),(1,0),(0,1)]
    new_mus = np.zeros_like(mus)
    new_sigmas = np.zeros_like(sigmas)
    for e in range(E):
        # 提取三个簇的 mus
        flat = np.array([mus[i, j, e] for (i, j) in state_list])
        # 从大到小排序，得到索引
        order_inds = np.argsort(-flat)
        for (i, j) in state_list:
            rank = hidden_state_order[(i, j)][e] - 1
            # 对应簇在初始 state_list 中的位置
            cid = order_inds[rank]
            oi, oj = state_list[cid]
            new_mus[i, j, e] = mus[oi, oj, e]
            new_sigmas[i, j, e] = sigmas[oi, oj, e]

    # 4. 强制第四簇赋值
    new_mus[1, 1, :] = 0
    new_sigmas[1, 1, :] = 0.0001

    # 打印结果
    print("\ninitial mus:\n", new_mus)
    print("\ninitial sigmas:\n", new_sigmas)

    return new_mus, new_sigmas

# 4.29 四簇聚类，单能见度控制，单（1，1），相关性，绑定
def jumpEM_1(params):
    # 使用 numpy 数组定义最终的转移矩阵（1）
    final_transition_matrices = np.array([
        [[0.968,  0.1006],
          [0.032,  0.8994]],

        [[0.7699, 0.1754],
        [0.2301, 0.8246]]
    ])
    # 使用 numpy 数组定义最终的均值矩阵
    final_mus = np.array([
        [[3.7473, 1.0986, 2.0794, 3.8391],
         [3.7812, 1.399,  2.1673, 3.6139]],

        [[4.4845, 0.848,  2.1673, 4.4122],
         [6.9632, 2.1972, 0.0953, 2.6391]]
    ])
    # 使用 numpy 数组定义最终的标准差矩阵
    final_sigmas = np.array([
        [[0.7345, 0.01,   0.01,  0.4884],
         [0.9485, 0.5528, 0.3073, 0.5786]],

        [[0.4746, 0.3258, 0.3073, 0.1294],
         [0.01,   0.01,   0.01,   0.01]]
    ])

    final_corr = np.array([
        [[[0.999, - 0.2586, - 0.4028,  0.1674],
          [-0.2586,  0.999,   0.1196, - 0.5766],
          [-0.4028, 0.1196,  0.999, - 0.2399],
          [0.1674, - 0.5766, - 0.2399,  0.999]],

         [[0.999, - 0.2586, - 0.4028,  0.1674],
          [-0.2586, 0.999, 0.1196, - 0.5766],
          [-0.4028,  0.1196,  0.999, - 0.2399],
          [0.1674, - 0.5766, - 0.2399, 0.999]]],


        [[[0.999, - 0.2586, - 0.4028,  0.1674],
          [-0.2586,  0.999,   0.1196, - 0.5766],
          [-0.4028,  0.1196,  0.999, - 0.2399],
          [0.1674, - 0.5766, - 0.2399,  0.999]],

         [[0.999, - 0.2586, - 0.4028,  0.1674],
          [-0.2586,  0.999,   0.1196, - 0.5766],
          [-0.4028,  0.1196,   0.999, - 0.2399],
          [0.1674, - 0.5766, - 0.2399,  0.999]]]
    ])

    final_pi = 0.99
    # 创建新参数的副本并更新相应的值
    new_params = params.copy()
    new_params['mus'] = final_mus
    new_params['sigmas'] = final_sigmas
    new_params['transition_matrices'] = final_transition_matrices
    new_params['final_pi'] = final_pi
    new_params['corr_mats'] = final_corr

    # 返回最终的结果
    return final_transition_matrices, final_mus, final_sigmas, final_pi, new_params

# 标准答案
def jumpEM_2(params):
    # 使用 numpy 数组定义最终的转移矩阵（1）
    final_transition_matrices = np.array([
        [[0.9867, 0.08],
          [0.0133, 0.92]],

        [[0.996,  0.0803],
        [0.004,  0.9197]]
    ])
    # 使用 numpy 数组定义最终的均值矩阵
    final_mus = np.array([
        [[3.74,   0.878,  2.0794, 3.7772],
          [5.5595, 1.095,  1.7352, 3.367]],

        [[4.5688, 0.453,  1.7352, 4.4192],
        [0.,     0.,     0.,     0.]]
    ])
    # 使用 numpy 数组定义最终的标准差矩阵
    final_sigmas = np.array([
        [[0.7637, 0.6711, 0.01,   0.5646],
          [0.6068, 0.738,  0.4273, 0.7167]],

        [[0.6425, 0.4725, 0.4273, 0.1577],
        [0.01,   0.01,   0.01,   0.01]]
    ])

    final_corr = np.array([
    [[[1., - 0.0768, - 0.4164,  0.4652],
       [-0.0768,  1.,      0.0703, - 0.1427],
      [-0.4164,  0.0703,  1.,      0.026],
     [0.4652, - 0.1427,  0.026,   1.]],

    [[1., - 0.179, - 0.4427,  0.1579],
     [-0.179,   1.,      0.109, - 0.2384],
    [-0.4427,    0.109,    1., - 0.2866],
    [0.1579, - 0.2384, - 0.2866,  1.]]],


    [[[1.,      0.,      0.,      0.],
      [0.,      1.,      0.,      0.],
     [0.,      0.,      1.,      0.],
    [0.,    0.,    0.,    1.]],

    [[1.,      0.,      0.,      0.],
     [0.,      1.,      0.,      0.],
    [0.,    0.,    1.,    0.],
    [0.,      0.,      0.,      1.]]]
    ])

    final_pi = 0.99
    # 创建新参数的副本并更新相应的值
    new_params = params.copy()
    new_params['mus'] = final_mus
    new_params['sigmas'] = final_sigmas
    new_params['transition_matrices'] = final_transition_matrices
    new_params['final_pi'] = final_pi
    new_params['corr_mats'] = final_corr

    # 返回最终的结果
    return final_transition_matrices, final_mus, final_sigmas, final_pi, new_params

# 隐藏值分类画图（条带）
def hidden_state_differentiation_chart(hidden_states, most_likely_hidden_state):
    accuracy_chain1 = np.zeros(len(hidden_states), dtype=int)
    accuracy_chain2 = np.zeros(len(hidden_states), dtype=int)

    # 统计双真值（1,1）的情况
    count_double_ones = 0
    #  3: 真阳性 2：真阴性  1：第一类错误  0：第二类错误
    for i in range(len(hidden_states)):
        prediction = (most_likely_hidden_state[0, i], most_likely_hidden_state[1, i])
        if prediction == (1, 1):
            count_double_ones += 1
        # 设置隐藏状态0, 1, 2的准确性评分
        if hidden_states[i] == 0:
            accuracy_chain1[i] = 2 if prediction[0] == 0 else 0
            accuracy_chain2[i] = 2 if prediction[1] == 0 else 0
        elif hidden_states[i] == 1:
            accuracy_chain1[i] = 3 if prediction[0] == 1 else 1
            accuracy_chain2[i] = 2 if prediction[1] == 0 else 0
        elif hidden_states[i] == 2:
            accuracy_chain1[i] = 2 if prediction[0] == 0 else 0
            accuracy_chain2[i] = 3 if prediction[1] == 1 else 1
    # 统计（1,1）
    print('Overall double ones = ', count_double_ones, "\n")

    correct_rate_chain1 = np.mean(np.isin(accuracy_chain1, [2, 3])) * 100
    correct_rate_chain2 = np.mean(np.isin(accuracy_chain2, [2, 3])) * 100

    # 定义颜色映射
    little_baby_color = ListedColormap(['#F4B608', '#3B96FF', '#0D8B43', '#006400'])

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), dpi=100)
    titles = ["Haze Forecast Results", "Sand and Dust Forecast Results"]
    correct_rates = [correct_rate_chain1, correct_rate_chain2]

    for i, accuracy in enumerate([accuracy_chain1, accuracy_chain2]):
        axs[i].imshow([accuracy], aspect='auto', cmap=little_baby_color, interpolation='none')
        axs[i].set_frame_on(False)
        axs[i].set_yticks([])
        axs[i].set_xticks([])  # 删除时间轴的数字
        axs[i].set_title(f'{titles[i]} - Accuracy: {correct_rates[i]:.2f}%')
        for spine in axs[i].spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.85, hspace=0.4)  # hspace的默认值大约为0.2，增加此值以增加间距

    legend_elements = [
        Patch(facecolor='#006400', label='True Positive'),
        Patch(facecolor='#0D8B43', label='True Negative'),
        Patch(facecolor='#F4B608', label='Type I Error'),
        Patch(facecolor='#3B96FF', label='Type II Error')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.85, 0.95),
               fancybox=True, shadow=True, ncol=2, fontsize=14)

    # 添加指向右侧的长箭头
    fig.add_artist(FancyArrowPatch((0.01, 0.12), (0.99, 0.12), transform=fig.transFigure,
                                   arrowstyle='->', mutation_scale=30, linewidth=1.25))
    plt.text(0.50, 0.1, 'T i m e     D i r e c t i o n', transform=fig.transFigure, ha='center', va='top', fontsize=14)

    plt.show()
# 隐藏值分类画图（月份）999
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def hidden_state_monthly_accuracy_chart(hidden_states, most_likely_hidden_state, months, years):
    # 输入校验
    if len(hidden_states) != len(months) or most_likely_hidden_state.shape[1] != len(hidden_states):
        raise ValueError("hidden_states, months, and most_likely_hidden_state dimensions must match!")

    # 初始化评分数据结构
    accuracy_chain1 = np.zeros(len(hidden_states), dtype=int)  # 雾霾
    accuracy_chain2 = np.zeros(len(hidden_states), dtype=int)  # 沙尘

    # 遍历每个时间点，计算评分
    for i in range(len(hidden_states)):
        prediction = (most_likely_hidden_state[0, i], most_likely_hidden_state[1, i])

        if hidden_states[i] == 0:
            accuracy_chain1[i] = 2 if prediction[0] == 0 else 0
            accuracy_chain2[i] = 2 if prediction[1] == 0 else 0
        elif hidden_states[i] == 1:
            accuracy_chain1[i] = 3 if prediction[0] == 1 else 1
            accuracy_chain2[i] = 2 if prediction[1] == 0 else 0
        elif hidden_states[i] == 2:
            accuracy_chain1[i] = 2 if prediction[0] == 0 else 0
            accuracy_chain2[i] = 3 if prediction[1] == 1 else 1

    # 统计每月数据
    unique_year_months = sorted(set((y, m) for y, m in zip(years, months)))
    num_time_points = len(unique_year_months)
    monthly_counts = np.zeros((num_time_points, 4), dtype=int)  # [TP, TN, FN, FP]
    monthly_total_counts = np.zeros(num_time_points, dtype=int)
    monthly_correct_counts_chain1 = np.zeros(num_time_points, dtype=int)
    monthly_correct_counts_chain2 = np.zeros(num_time_points, dtype=int)

    # 构建年份-月份索引
    year_month_to_index = {ym: idx for idx, ym in enumerate(unique_year_months)}

    for i in range(len(hidden_states)):
        year_month = (years[i], months[i])
        month_index = year_month_to_index[year_month]

        if accuracy_chain1[i] == 3:  # True Positive
            monthly_counts[month_index, 0] += 1
        elif accuracy_chain1[i] == 2:  # True Negative
            monthly_counts[month_index, 1] += 1
        elif accuracy_chain1[i] == 1:  # False Negative
            monthly_counts[month_index, 2] += 1
        elif accuracy_chain1[i] == 0:  # False Positive
            monthly_counts[month_index, 3] += 1

        if accuracy_chain1[i] in [2, 3]:
            monthly_correct_counts_chain1[month_index] += 1
        if accuracy_chain2[i] in [2, 3]:
            monthly_correct_counts_chain2[month_index] += 1

        monthly_total_counts[month_index] += 1

    # 计算每月的比例和正确率
    monthly_ratios = (monthly_counts.T / monthly_total_counts).T * 100
    monthly_accuracy_chain1 = (monthly_correct_counts_chain1 / monthly_total_counts) * 100
    monthly_accuracy_chain2 = (monthly_correct_counts_chain2 / monthly_total_counts) * 100

    # 计算全局平均正确率
    overall_accuracy_chain1 = np.mean(np.isin(accuracy_chain1, [2, 3])) * 100
    overall_accuracy_chain2 = np.mean(np.isin(accuracy_chain2, [2, 3])) * 100

    # 动态生成横坐标标签，月份换行
    x_labels = [f"{y}\n{['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m - 1]}"
                for y, m in unique_year_months]

    # 绘制叠加柱状图和正确率折线图
    fig, ax1 = plt.subplots(figsize=(14, 8), dpi=100)

    labels = ["True Positive", "True Negative", "False Negative", "False Positive"]
    colors = ['#4CAF50', '#2196F3', '#FFC107', '#FF5722']

    bottom = np.zeros(num_time_points)
    for i in range(4):
        ax1.bar(x_labels, monthly_ratios[:, i], bottom=bottom, label=labels[i], color=colors[i], alpha=0.8)
        bottom += monthly_ratios[:, i]

    # 绘制雾霾和沙尘的正确率折线
    ax2 = ax1.twinx()
    haze_line, = ax2.plot(
        x_labels,
        monthly_accuracy_chain1,
        color='green',
        marker='o',
        label='Haze Monthly Accuracy',
        linestyle='-',
        linewidth=2
    )
    sand_line, = ax2.plot(
        x_labels,
        monthly_accuracy_chain2,
        color='blue',
        marker='s',
        label='Sand and Dust Monthly Accuracy',
        linestyle='--',
        linewidth=2
    )

    # 添加全局平均正确率线并标注值
    ax2.axhline(overall_accuracy_chain1, color='green', linestyle=':', linewidth=2, label='Haze Overall Mean')
    ax2.text(len(x_labels) - 1, overall_accuracy_chain1 + 1, f"{overall_accuracy_chain1:.2f}%", color='green')
    ax2.axhline(overall_accuracy_chain2, color='blue', linestyle='-.', linewidth=2, label='Sand and Dust Overall Mean')
    ax2.text(len(x_labels) - 1, overall_accuracy_chain2 + 1, f"{overall_accuracy_chain2:.2f}%", color='blue')

    # 设置标题和标签
    ax1.set_title("Monthly Prediction Accuracy and Error Composition", fontsize=20)
    ax1.set_xlabel("Months", fontsize=18)
    ax1.set_ylabel("Percentage (%)", fontsize=18)
    ax2.set_ylabel("Accuracy (%)", fontsize=18)

    # 合并两个轴的图例并设置到图底部
    handles1, labels1 = ax1.get_legend_handles_labels()  # 柱状图图例
    handles2, labels2 = ax2.get_legend_handles_labels()  # 折线图图例
    fig.legend(
        handles1 + handles2, labels1 + labels2,
        fontsize=14, loc="lower center", bbox_to_anchor=(0.74, 0.22), ncol=2
    )

    # 美化图表
    ax1.grid(visible=True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# F1 and AUC plot
def plot_f1_and_auc(hidden_states, most_likely_hidden_state):
    from sklearn.metrics import confusion_matrix
    """
    Draw F1 scores and AUC plots to evaluate the model's ability to distinguish rare events (e.g., haze and dust).

    Parameters:
    hidden_states (np.ndarray): Ground truth hidden states, 1D array with values 0, 1, or 2.
    most_likely_hidden_state (np.ndarray): Predicted hidden states, 2D array with shape (2, n_steps).
    """
    # Ensure input shapes are consistent
    assert len(hidden_states) == most_likely_hidden_state.shape[1], "Mismatch between hidden states and predicted states length"

    # 预测隐藏状态转换为一维标签：
    # 与你在 plot_f1_auc 中保持一致：
    # 如果第一个链为 0 且第二个链为 0，则预测为 0（正常）
    # 如果第一个链为 0 且第二个链不为 0，则预测为 2（沙尘）
    # 否则预测为 1（雾霾）
    T = hidden_states.shape[0]
    predicted_states = np.zeros(T, dtype=int)

    for t in range(T):
       z1, z2 = most_likely_hidden_state[0, t], most_likely_hidden_state[1, t]

       if z1 == 0 and z2 == 0:
           predicted_states[t] = 0  # 晴朗
       elif z1 == 0 and z2 == 1:
           predicted_states[t] = 2  # 沙尘
       elif z1 == 1 and z2 == 0:
           predicted_states[t] = 1  # 雾霾
       else:  # (1,1) 双事件
           predicted_states[t] = 1  # 或者根据业务需要，丢弃它/合并到某一类

    # Output the distribution of actual and predicted states
    print("Actual hidden state distribution:", np.bincount(hidden_states, minlength=3))
    print("Predicted hidden state distribution:", np.bincount(predicted_states, minlength=3))

    predicted_states = np.zeros(T, dtype=int)
    for t in range(T):
        if most_likely_hidden_state[0, t] == 0:
            if most_likely_hidden_state[1, t] == 0:
                predicted_states[t] = 0  # 晴朗
            else:
                predicted_states[t] = 2  # 沙尘
        else:
            predicted_states[t] = 1  # 雾霾
    # —— 在这里插入混淆矩阵的输出 —— #
    cm = confusion_matrix(hidden_states, predicted_states, labels=[0, 1, 2])
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(cm, "\n")

    # Define rare events: Dust (2) and Haze (1)
    rare_event_labels = [1, 2]

    # Custom calculation of TP, FP, TN, FN
    TP_FP_TN_FN = np.zeros((2, 4))  # [Label][TP, FP, TN, FN]
    for i in range(len(hidden_states)):
        for idx, label in enumerate(rare_event_labels):
            if hidden_states[i] == label and predicted_states[i] == label:
                TP_FP_TN_FN[idx, 0] += 1  # TP
            elif hidden_states[i] != label and predicted_states[i] == label:
                TP_FP_TN_FN[idx, 1] += 1  # FP
            elif hidden_states[i] != label and predicted_states[i] != label:
                TP_FP_TN_FN[idx, 2] += 1  # TN
            elif hidden_states[i] == label and predicted_states[i] != label:
                TP_FP_TN_FN[idx, 3] += 1  # FN

    print("Custom TP/FP/TN/FN for each label:")
    print(TP_FP_TN_FN)

    # Custom TPR, FAR, SI
    TPR = TP_FP_TN_FN[:, 0] / (TP_FP_TN_FN[:, 0] + TP_FP_TN_FN[:, 3])
    FAR = TP_FP_TN_FN[:, 1] / (TP_FP_TN_FN[:, 1] + TP_FP_TN_FN[:, 2])
    SI = TPR - FAR
    print("Custom Metrics:")
    print("True Positive Rate (TPR):", TPR)
    print("False Alarm Rate (FAR):", FAR)
    print("Success Index (SI):", SI)

    # Calculate F1 scores for each label
    f1_scores = {}
    # # 计算单类F1
    # for label in rare_event_labels:
    #     f1_scores[label] = f1_score(
    #         (hidden_states == label).astype(int),  # 真实值转换为二分类（0/1）
    #         (predicted_states == label).astype(int),  # 预测值转换为二分类（0/1）
    #         average="binary"  # 明确使用二分类 F1 计算
    #     )
    # 加权F1
    for label in rare_event_labels:
        f1_scores[label] = f1_score(
            (hidden_states == label).astype(int),  # 真实值转换为二分类（0/1）
            (predicted_states == label).astype(int),  # 预测值转换为二分类（0/1）
            average="weighted"  # 明确使用二分类 F1 计算
        )

    # Print F1 scores
    print("F1 Scores:")
    for label, f1 in f1_scores.items():
        print(f"Label {label}: F1 Score = {f1:.4f}")

    # Calculate AUC for each label
    auc_scores = {}
    for label in rare_event_labels:
        fpr, tpr, _ = roc_curve(hidden_states == label, predicted_states == label)
        auc_scores[label] = auc(fpr, tpr)

    # Print AUC
    print("AUC Scores:")
    for label, auc_score in auc_scores.items():
        print(f"Label {label}: AUC = {auc_score:.4f}")

    # Check ROC curve data for each label
    for label in rare_event_labels:
        fpr, tpr, thresholds = roc_curve(hidden_states == label, predicted_states == label)
        print(f"Label {label} ROC Curve:")
        print("  FPR:", fpr)
        print("  TPR:", tpr)
        print("  Thresholds:", thresholds)

    # Plot F1 and AUC figures
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # F1 figure
    axs[0].bar(f1_scores.keys(), f1_scores.values(), color=['skyblue', 'orange'])
    axs[0].set_title("F1 Scores for Haze and Dust")
    axs[0].set_xlabel("Event Label")
    axs[0].set_ylabel("F1 Score")
    axs[0].set_xticks(rare_event_labels)
    axs[0].set_xticklabels(["Haze", "Dust"])
    axs[0].set_ylim(0, 1)
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # AUC figure
    for label in rare_event_labels:
        fpr, tpr, _ = roc_curve(hidden_states == label, predicted_states == label)
        axs[1].plot(fpr, tpr, label=f"{['Haze', 'Dust'][label-1]} (AUC = {auc_scores[label]:.2f})")
    axs[1].plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Guess")
    axs[1].set_title("AUC Curves for Haze and Dust")
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate")
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# 雾霾、沙尘事件区分结果检验
def result_verification(hidden_states, most_likely_hidden_state):
    # 计算雾霾、沙尘的TP（真阳率）, FP（假阳率）, TN（真阴率）, FN（假阴率）
    TP_FP_TN_FN = np.zeros((2, 4))
    # 总体正确率
    T = np.zeros((2))
    for i in range(len(hidden_states)):
        prediction = (most_likely_hidden_state[0, i], most_likely_hidden_state[1, i])
        if prediction[0] == 0:
            if hidden_states[i] == 0 or hidden_states[i] == 2:
                TP_FP_TN_FN[0, 2] += 1
                T[0] += 1
            else:
                TP_FP_TN_FN[0, 3] += 1
        elif prediction[0] == 1:
            if hidden_states[i] == 1:
                TP_FP_TN_FN[0, 0] += 1
                T[0] += 1
            else:
                TP_FP_TN_FN[0, 1] += 1

        if prediction[1] == 0:
            if hidden_states[i] == 0 or hidden_states[i] == 1:
                TP_FP_TN_FN[1, 2] += 1
                T[1] += 1
            else:
                TP_FP_TN_FN[1, 3] += 1
        elif prediction[1] == 1:
            if hidden_states[i] == 2:
                TP_FP_TN_FN[1, 0] += 1
                T[1] += 1
            else:
                TP_FP_TN_FN[1, 1] += 1

    # 计算TPR, FAR, SI
    TPR = np.zeros((2))
    FAR = np.zeros((2))
    SI = np.zeros((2))
    for i in range(2):
        TPR[i] = TP_FP_TN_FN[i, 0] / (TP_FP_TN_FN[i, 0] + TP_FP_TN_FN[i, 3]) \
            if (TP_FP_TN_FN[i, 0] + TP_FP_TN_FN[i, 3]) != 0 else 0
        FAR[i] = TP_FP_TN_FN[i, 1] / (TP_FP_TN_FN[i, 1] + TP_FP_TN_FN[i, 2]) \
            if (TP_FP_TN_FN[i, 1] + TP_FP_TN_FN[i, 2]) != 0 else 0
        SI[i] = TPR[i] - FAR[i]

    # 输出结果
    print("True Positive Rate (TPR) of event differentiation = \n", TPR)
    print("False Alarm Rate (FAR) of event differentiation = \n", FAR)
    print("Success Index (SI) of event differentiation = \n", SI, "\n")

    return T

# 文件中隐藏值解码
def check_result(row):
    if row['1=雾/霾，2=沙/尘'] == 0 and row['雾霾估计'] == 0 and row['沙尘估计'] == 0:
        return 1
    elif row['1=雾/霾，2=沙/尘'] == 1 and row['雾霾估计'] == 1 and row['沙尘估计'] == 0:
        return 1
    elif row['1=雾/霾，2=沙/尘'] == 2 and row['雾霾估计'] == 0 and row['沙尘估计'] == 1:
        return 1
    else:
        return 0

# 三维图像，联合调整全局权重 v 与观测值权重 w。F1 Score
def plot_3d_performance_metric1(hidden_states, observed_states, weights, numb):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import f1_score

    """
    增强版三维性能曲面图绘制函数

    参数：
      hidden_states: 真实的隐藏状态标签，1D 数组（取值 0, 1, 2）
      observed_states: 观测数据（形状与代码中一致）
      weights: 观测权重变量，形状为 (4, n_observed_chains)，其中
               weights[2, :] 为雾霾权重，weights[3, :] 为沙尘权重
      numb: 控制 alpha 和 global_weight 的取点数
    """

    # 参数范围定义
    alpha_range = np.linspace(0.5, 2, numb)
    global_weight_range = np.linspace(0.01, 4, numb)

    # 性能矩阵初始化（雾霾F1，沙尘F1，加权F1）
    full_performance_scores = np.zeros((3, len(alpha_range), len(global_weight_range)))

    # 遍历参数范围，计算各个 F1 score
    for i, alpha in enumerate(alpha_range):
        weights[0, 0] = alpha * weights[0, 0]
        final_observed_weights = weights

        for j, global_weight in enumerate(global_weight_range):
            # 运行 HMM 维特比推理（假设 H.Viterbi 已定义）
            most_likely_hidden_state, _, _ = H.Viterbi(
                observed_states, final_observed_weights, global_weight
            )

            # 状态解码
            T = hidden_states.shape[0]
            predicted_states = np.zeros(T, dtype=int)
            for t in range(T):
                if most_likely_hidden_state[0, t] == 0:
                    if most_likely_hidden_state[1, t] == 0:
                        predicted_states[t] = 0  # 晴朗
                    else:
                        predicted_states[t] = 2  # 沙尘
                else:
                    predicted_states[t] = 1  # 雾霾

            # 计算雾霾（1）的 F1-score
            f1_haze = f1_score(
                (hidden_states == 1).astype(int),
                (predicted_states == 1).astype(int),
                average="binary"
            )
            full_performance_scores[0, i, j] = f1_haze  # 雾霾 F1

            # 计算沙尘（2）的 F1-score
            f1_dust = f1_score(
                (hidden_states == 2).astype(int),
                (predicted_states == 2).astype(int),
                average="binary"
            )
            full_performance_scores[1, i, j] = f1_dust  # 沙尘 F1

            # 计算整体分类的加权 F1-score
            weighted_f1 = f1_score(hidden_states, predicted_states, average="weighted")
            full_performance_scores[2, i, j] = weighted_f1  # 加权 F1

            print(
                f"计算进度：alpha {alpha:.3f}, global_weight {global_weight:.3f}, "
                f"Haze F1: {f1_haze:.4f}, Dust F1: {f1_dust:.4f}, Weighted F1: {weighted_f1:.4f}"
            )

    # 美化设置
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'font.size': 14,
        'axes.labelsize': 12,
        'axes.titlesize': 14
    })

    # 构建网格
    X, Y = np.meshgrid(global_weight_range, alpha_range)

    # -------------------------- 图1：两个子图（雾霾 F1 与 沙尘 F1） --------------------------
    fig1 = plt.figure(figsize=(12, 8), dpi=120)
    ax1 = fig1.add_subplot(121, projection='3d')
    ax2 = fig1.add_subplot(122, projection='3d')

    def plot_surface_with_contours1(ax, performance_scores, score_label):
        # 绘制曲面
        surf = ax.plot_surface(
            X, Y, performance_scores,
            cmap='viridis',
            rstride=1,
            cstride=1,
            edgecolor='none',
            antialiased=True,
            alpha=0.85
        )
        # 等高线增强：底部轮廓填充
        offset_z = performance_scores.min() - 0.1 * (performance_scores.ptp())
        ax.contourf(
            X, Y, performance_scores,
            zdir='z', offset=offset_z,
            cmap='viridis', alpha=0.3, levels=15
        )
        offset_y = alpha_range.max() + 0.05 * (alpha_range.ptp())
        ax.contour(
            X, Y, performance_scores,
            zdir='y', offset=offset_y,
            colors='k', alpha=0.4, levels=8
        )
        # 坐标轴设置
        ax.set_xlabel('Global Weight\n(Transition vs Emission Balance)', labelpad=10)
        ax.set_ylabel('Alpha\n(Haze vs Dust Weight Ratio)', labelpad=10)
        ax.set_zlabel(f'{score_label} F1 Score', labelpad=10)
        ax.tick_params(axis='both', which='major', pad=3)
        ax.tick_params(labelsize=14)
        # 颜色条
        cbar = fig1.colorbar(surf, ax=ax, pad=0.08, shrink=0.5, aspect=15)
        cbar.set_label('Classification Performance', rotation=270, labelpad=12, fontsize=12)
        # 背景与网格优化
        ax.xaxis.pane.set_alpha(0.95)
        ax.yaxis.pane.set_alpha(0.95)
        ax.zaxis.pane.set_alpha(0.95)
        ax.xaxis.pane.set_facecolor('white')
        ax.yaxis.pane.set_facecolor('white')
        ax.zaxis.pane.set_facecolor('white')
        ax.grid(True, linestyle=':', alpha=0.6)
        # 视角优化
        ax.view_init(elev=28, azim=-124)
        # 性能峰值标注
        max_loc = np.unravel_index(performance_scores.argmax(), performance_scores.shape)
        ax.plot(
            [global_weight_range[max_loc[1]]],
            [alpha_range[max_loc[0]]],
            [performance_scores.max()],
            'r*', markersize=9, markeredgecolor='k',
            label=f'Optimal Point (α={alpha_range[max_loc[0]]:.2f}, v={global_weight_range[max_loc[1]]:.2f})'
        )
        ax.legend(loc='upper right', framealpha=0.9, bbox_to_anchor=(1, 1.1))
        ax.set_title(f"Parameter Optimization Landscape\nfor FHMM-based Classification\n({score_label} F1)",
                     pad=25, fontsize=15, fontweight='bold', y=1.07)

    # 子图1：使用 full_performance_scores[0] —— 雾霾 F1
    plot_surface_with_contours1(ax1, full_performance_scores[0], "Haze")
    # 子图2：使用 full_performance_scores[1] —— 沙尘 F1
    plot_surface_with_contours1(ax2, full_performance_scores[1], "Dust")
    fig1.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.88, wspace=0.3)
    plt.show()

    # -------------------------- 图2：单个图（加权 F1） --------------------------
    fig2 = plt.figure(figsize=(12, 8), dpi=120)
    ax3 = fig2.add_subplot(111, projection='3d')

    def plot_surface_with_contours2(ax3, performance_scores):

        # 绘制曲面
        surf = ax3.plot_surface(
            X, Y, performance_scores,
            cmap='viridis',
            rstride=1,
            cstride=1,
            edgecolor='none',
            antialiased=True,
            alpha=0.85
        )

        # 等高线增强：底部轮廓填充
        offset_z = performance_scores.min() - 0.1 * (performance_scores.ptp())
        ax3.contourf(
            X, Y, performance_scores,
            zdir='z', offset=offset_z,
            cmap='viridis', alpha=0.3, levels=15
        )
        offset_y = alpha_range.max() + 0.05 * (alpha_range.ptp())
        ax3.contour(
            X, Y, performance_scores,
            zdir='y', offset=offset_y,
            colors='k', alpha=0.4, levels=8
        )

        # 坐标轴设置
        ax3.set_xlabel('Global Weight\n(Transition vs Emission Balance)', labelpad=10)
        ax3.set_ylabel('Alpha\n(Haze vs Dust Weight Ratio)', labelpad=10)
        ax3.set_zlabel('MSE', labelpad=10)
        ax3.tick_params(axis='both', which='major', pad=3)
        ax3.tick_params(labelsize=14)

        # 颜色条
        cbar = fig2.colorbar(surf, pad=0.08, shrink=0.7, aspect=15)
        cbar.set_label('Classification Performance', rotation=270, labelpad=15)

        # 背景与网格优化
        ax3.xaxis.pane.set_alpha(0.95)
        ax3.yaxis.pane.set_alpha(0.95)
        ax3.zaxis.pane.set_alpha(0.95)
        ax3.xaxis.pane.set_facecolor('white')
        ax3.yaxis.pane.set_facecolor('white')
        ax3.zaxis.pane.set_facecolor('white')
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.view_init(elev=28, azim=-124)

        # 性能峰值标注
        max_loc = np.unravel_index(performance_scores.argmax(), performance_scores.shape)
        ax3.plot(
            [global_weight_range[max_loc[1]]],
            [alpha_range[max_loc[0]]],
            [performance_scores.max()],
            'r*', markersize=14, markeredgecolor='k',
            label=f'Optimal Point (α={alpha_range[max_loc[0]]:.2f}, v={global_weight_range[max_loc[1]]:.2f})'
        )
        ax3.legend(loc='upper right', framealpha=0.9)
        plt.title(
            "Parameter Optimization Landscape\nfor FHMM-based Classification (MSE)",
            pad=25, fontsize=20, fontweight='bold', y=1.07
        )

    # 图2：使用 full_performance_scores[2] —— 雾霾/沙尘/晴朗 加权F1
    plot_surface_with_contours2(ax3, full_performance_scores[2])
    fig2.subplots_adjust(left=0.15, right=0.9, bottom=0.1, top=0.88)
    plt.show()

def initial_corr_mats(params,
                      observed_states,        # shape (E, T)
                      encoded_hidden_states,  # ← 仅保留接口，不再使用
                      min_samples: int = 10,  # ← 仍可留作备用
                      eps_pd: float = 1e-6):
    """
    计算全局 Gaussian Copula 相关矩阵 R_global，并复制到
    (K,K,E,E) 形状；从而使相关性只与 X 有关，而与 Z 无关。
    """
    import numpy as np
    from numpy.linalg import eigh

    K = params['hidden_alphabet_size']   # =2
    E = params['n_observed_chains']      # =4

    # —— 1) 用全部观测值（对数域）计算经验相关 —— #
    X_log = np.log(np.clip(observed_states.T, 1e-6, None))   # (T,E)
    R = np.corrcoef(X_log, rowvar=False)
    np.fill_diagonal(R, 1.0)
    R = np.clip(R, -0.999, 0.999)

    # —— 2) 正定性保护 —— #
    vals, _ = eigh(R)
    if vals.min() < eps_pd:
        R += np.eye(E) * (eps_pd - vals.min() + 1e-8)

    # —— 3) 复制到所有隐藏状态 —— #
    corr_mats = np.tile(R, (K, K, 1, 1)).reshape(K, K, E, E)
    return corr_mats


# 计时
start_time = time.time()

# 原始数据地址与保存文件地址
location = 'D:/Desktop/桌面旧/合并数据删1.xls'
file_path = 'D:/Desktop/桌面旧/合并数据删1.xlsx'
histr_location = 'D:/Desktop/桌面旧/历史数据1.xls'

# 读取数据与历史数据
(observed_states, hidden_states,
                    encoded_hidden_states, months, years) = Get_observed_and_hidden_state(location)

(histr_observed_states, histr_hidden_states, histr_encoded_hidden_states,
                    histr_months, histr_years) = Get_observed_and_hidden_state(histr_location)
print("\nmonths:", months)
print('\nobserved_states =\n', observed_states)
print('\nshape of observed_states = ', observed_states.shape)
print("\nhidden_states = \n", hidden_states)
print("\nshape of hidden_states", hidden_states.shape)

# 设定参数
n_steps = len(observed_states[0])  # 时间步
E = observed_states.shape[0]  # 观测链个数
M = 2  # 隐藏链个数
K = 2  # 隐藏链范围
precision = 2  # 迭代终止条件

# 字典便于传递和更新参数
params = {
    'hidden_alphabet_size': K,  # 隐藏值范围
    'n_hidden_chains': M,  # 隐藏链个数
    'n_observed_chains': E,  # 观测链个数
    'initial_hidden_state': np.zeros((M, K)),  # 初始隐藏状态
    'transition_matrices': np.zeros((M, K, K)),  # 转移矩阵
    'mus': np.zeros((K,) * M + (E,)),  # 均值224
    'sigmas': np.zeros((K,) * M + (E,)),  # 标准差224
    'observed_weights': np.ones((K,) * M + (E,)),  # 观测链权重224
    'zero_inflation_pi': 0.99, # 零膨胀概率
    'corr_mats': np.tile(np.eye(E), (K, K, 1, 1)).reshape(K, K, E, E), # 相关矩阵（单位矩阵）
    'force_corr_identity': False # 是否禁用相关性
}

# 参数初始化
# 设置初始隐藏值
params['initial_hidden_state'][0, :] = [548 / 8476] * K
params['initial_hidden_state'][1, :] = [76 / 8476] * K
# 设置初始转移矩阵
transition_matrix_real = initial_transition_matrix(params, histr_hidden_states)
params['transition_matrices'] = transition_matrix_real
print('\nshape of transition_matrices:', params['transition_matrices'].shape)
print('\ninitial transition_matrices:\n', params['transition_matrices'])
print('\nzero_inflation_pi:\n', params['zero_inflation_pi'])

# 设置初始mus，sigmas，corr
params['mus'], params['sigmas'] = initial_ObservedGivenHidden_matrix(params, observed_states.T)
# params['corr_mats'] = initial_corr_mats(params, observed_states, encoded_hidden_states)
print("\ninitial corr:\n", params['corr_mats'])

# params['mus'] = np.array([[[2.8276,   0.3483,   1.9637,   3.643],
#   [4.9088,   1.4853,   1.463,    2.8348]],
#
# [[3.9525, - 13.8155,   0.9672,   4.3288],
# [0.,0.,0.,0.]]])
# params['sigmas'] =  np.array([[[0.3794, 0.3466, 0.1163, 0.2063],
#   [0.4482, 0.3875, 0.14,   0.3234]],
#
#  [[0.2885, 0.01,   0.2055, 0.184 ],
#   [0.0001, 0.0001, 0.0001, 0.0001]]])

# 创建实体
F = FullDiscreteFactorialHMM(params=params, n_steps=n_steps, calculate_on_init=True)

# 运行EM算法
# 对数似然度 Log Likelihood = log(P(观测数据 | 模型参数))
final_transition_matrices, final_mus, final_sigmas, new_params = F.EM(
    observed_states, likelihood_precision = precision, verbose=True, print_every=1)

# 跳过EM算法
# final_transition_matrices, final_mus, final_sigmas, final_pi, new_params = jumpEM_1(params)

print('\nfinal_transition_matrices = \n', final_transition_matrices)
print('\nfinal mus = \n', new_params['mus'])
print('\nfinal sigmas = \n', new_params['sigmas'])
print('\nfinal corr\n', new_params['corr_mats'])

# 互信息法计算观测链权重 (将params里（4）的权重（1，1，1，1）变成了new_params里的（4，4）维）
sigle_weights, new_params['observed_weights'] = Get_observed_weights(histr_observed_states, histr_hidden_states, params)
print("weights: Clear, Haze, Smog, Smog and Haze =\n", new_params['observed_weights'])

# 设置全局权重
# transition_emission_ratio
global_weight = 1
print("\nglobal_weight = ", global_weight)

# 创建实体
H = FullDiscreteFactorialHMM(params=new_params, n_steps=n_steps, calculate_on_init=True)
# 运行Viterbi算法（使用均值权重）
most_likely_hidden_state, back_pointers, lls = H.Viterbi(observed_states, new_params['observed_weights'], global_weight)
# # 隐藏状态区分结果检验(F1 AUC画图替代)
_ = result_verification(hidden_states, most_likely_hidden_state)
# 隐藏状态区分结果画图(条带)
hidden_state_differentiation_chart(hidden_states, most_likely_hidden_state)

# 事件区分图（月份条形图）
hidden_state_monthly_accuracy_chart(hidden_states, most_likely_hidden_state, months, years)

# F1 AUC画图
f1_scores = []
f1_scores = plot_f1_and_auc(hidden_states, most_likely_hidden_state)

# # 三维图像，联合调整全局权重 v 与观测值权重 w，F1 Score
# # 设置取样密度
# numb1 = 10
# plot_3d_performance_metric1(hidden_states, observed_states, new_params['observed_weights'], numb1)
#

# 显示运行时间
end_time = time.time()
print('\n', f"代码执行时间: {end_time - start_time:.6f} 秒")
