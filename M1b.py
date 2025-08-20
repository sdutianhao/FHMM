# copula 19 vtb 去掉（1，1），sigma 0.1
import functools
import itertools
import operator
import numpy as np
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
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors  # 用于k-NN搜索的参考，但以下实现中手动计算
from collections import Counter
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

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
        # 【新增】对每个 t 归一化，使得 sum_{i,j} gamma_{i,j,t} = 1
        sum_over_states = gammas.sum(axis=(0, 1), keepdims=True)  # 形状 (1,1,T)
        gammas = gammas / sum_over_states  # 自动广播

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

    def Viterbi(self, observed_states, observed_weights, global_weight, hidden_states):
        import numpy as np, functools
        from scipy.stats import lognorm, norm

        print("\nWeight applied in Viterbi = \n", observed_weights)
        # ---------- 预处理 ----------
        if observed_states.ndim == 1:
            observed_states = observed_states.reshape((1, -1))
        E, T = observed_states.shape

        backp = np.zeros(self.hidden_indices.field_sizes + [self.n_hidden_states, T - 1], dtype=int)
        lls = np.zeros(self.hidden_indices.field_sizes + [T], dtype=float)

        init = functools.reduce(np.kron, self.initial_hidden_state_tensor).reshape(self.hidden_indices.field_sizes)
        lls[..., 0] = np.log(init + 1e-12)
        prev_ll = lls[..., 0]

        # ---------- 递推 ----------
        for t in range(1, T):
            # —— 1) 转移部分（原逻辑完全保留） ——
            vec = prev_ll.copy()
            argm = np.zeros(self.hidden_indices.field_sizes + [self.n_hidden_states], dtype=int)
            for i in range(self.n_hidden_states):
                logA = np.log(self.transition_matrices_tensor[t - 1][i] + 1e-12)
                idx = [slice(None)] + [None] * self.n_hidden_states
                idx[i + 1] = slice(None)
                B = logA[tuple(idx)] + vec[np.newaxis, ...]
                vec = np.moveaxis(B.max(axis=i + 1), 0, i)
                argm[..., i] = np.moveaxis(B.argmax(axis=i + 1), 0, i)
            backp[..., :, t - 1] = argm


            # —— 2) 发射概率计算与正规化 ——
            # 2-a 互信息权重
            weighted_emis     = self.GetObservedGivenHidden(observed_states[..., t], observed_weights, t)
            # log_weighted_emis = np.log(weighted_emis + 1e-12)
            log_weighted_emis = np.where(weighted_emis > 0, np.log(weighted_emis), -np.inf)


            # 2-b 应用全局权重 v (global_weight)
            global_emis = np.power(weighted_emis, 1.0 / global_weight)

            # 2-c 【实装】对所有隐藏状态进行归一化，使其成为合法的概率分布
            sum_emis = global_emis.sum()
            if sum_emis > 1e-100:  # 增加一个极小值以避免浮点数问题
                normalized_global_emis = global_emis / sum_emis
            else:
                # 处理总和为0的极端情况，避免除以零
                normalized_global_emis = global_emis

            # 2-d 取对数，作为最终的发射得分
            log_normalized_global_emis = np.where(normalized_global_emis > 0, np.log(normalized_global_emis), -np.inf)

            # —— 3) 合并得分（使用归一化后的发射得分） ——
            Log_trans = vec
            combined = Log_trans + log_normalized_global_emis

            # —— 5) 调试输出（仅 t<2） ——

            π0 = self.params.get('zero_inflation_pi', 0.99)
            max_vis = self.params.get('max_visibility', 10)
            EPS = 1e-300

            if t < 4 or t == T-1:
                if t == T - 1:
                    print("\n\n\n         ------------------------------------------------------")
                    print("           ... Skipping output for intermediate time steps ...")
                    print("         ------------------------------------------------------\n\n\n")

                obs_view = observed_states[..., t].copy()
                log_view = obs_view.copy()
                log_view[0] = np.log(np.clip(log_view[0], 1e-50, None))  # PM10
                log_view[1] = np.log(np.clip(log_view[1], 1e-50, None))  # 风速
                log_view[2] = np.log(np.clip(log_view[2], 1e-50, None))  # 能见度
                log_view[3] = np.log(np.clip(log_view[3], 1e-50, None))  # 相对湿度

                print("\n")
                print(f"  t = {t}:\n")
                print("  观测值 {Log_PM10, Log_风速, Log_能见度, Log_相对湿度}：")
                print("  [%.4f, %.4f, %.4f, %.4f]" % tuple(log_view))
                print("  隐藏状态: { 0:晴朗, 1:沙尘, 2:有霾, 3:空状态 }\n")

                # ---- 原始参数 / 权重打印 ----
                for idx_h, (z1, z2) in enumerate(self.all_hidden_states):
                    params_list = [(self.mus[z1, z2, e], self.sigmas[z1, z2, e]) for e in range(E)]
                    weight_list = [f"{w:.4f}" for w in observed_weights[z1, z2, :]]
                    print(f"  隐藏状态 {idx_h} (z1={z1},z2={z2}):")
                    print(f"    参数 (mu, sigma) = {params_list}")
                    print(f"    观测值权重 W      = {weight_list}\n")

                # ---- A) 单维概率 & 乘积（含零膨胀，不含权重） ----
                print("  ------ 各个隐藏链发射概率（零膨胀修正） ------")

                zero_prod_list = []  # Π p_e                  （无权重）
                weight_raw_list = []  # Π p_e^w               （未除 Z_e）
                weight_norm_list = []  # Π p_e^w / Π Z_e(w)    （已除 Z_e）
                log_cop_list = []  # Copula 对数因子
                logZ_tot_list = []  # Σ log Z_e(w) 仅供核对

                for idx_h, (z1, z2) in enumerate(self.all_hidden_states):
                    probs_feat = np.zeros(E)  # 各维 pdf(x)
                    logZ = np.zeros(E)  # ln Z_e(w)
                    wvec = observed_weights[z1, z2, :]
                    cdfs_feat = np.zeros(E) # 存各维度 cdf(x)

                    # —— 1) 单维 pdf / ln Z_e —— #
                    for e in range(E):
                        mu = self.mus[z1, z2, e]
                        sigma = max(self.sigmas[z1, z2, e], 1e-8)
                        x = obs_view[e]

                        if e in [0, 1, 3]:  # 对数正态
                            probs_feat[e] = lognorm(s=sigma, scale=np.exp(mu)).pdf(x)
                            if not np.isclose(wvec[e], 1.0):
                                logZ[e] = ((1 - wvec[e]) * np.log(sigma)
                                           + ((1 - wvec[e]) / 2) * np.log(2 * np.pi)
                                           - 0.5 * np.log(wvec[e])
                                           + mu + sigma ** 2 / (2 * wvec[e]))
                        else:  # 能见度（零膨胀）
                            if (z1, z2) == (0, 0) and np.isclose(x, max_vis, atol=1e-6):
                                probs_feat[e] = π0
                            else:
                                base_pdf = lognorm(s=sigma, scale=np.exp(mu)).pdf(x)
                                probs_feat[e] = (1 - π0) * base_pdf if (z1, z2) == (0, 0) else base_pdf
                            if not np.isclose(wvec[e], 1.0):
                                logZ[e] = ((1 - wvec[e]) * np.log(sigma)
                                           + ((1 - wvec[e]) / 2) * np.log(2 * np.pi)
                                           - 0.5 * np.log(wvec[e])
                                           + mu + sigma ** 2 / (2 * wvec[e]))

                        probs_feat[e] = max(probs_feat[e], EPS)  # 防 0 下溢

                    # —— 2) 乘积 —— #
                    prod_no_w = np.prod(probs_feat)  # Π p_e
                    prod_w_raw = np.prod(probs_feat ** wvec)  # Π p_e^w
                    Z_tot = np.exp(logZ.sum())  # Π Z_e(w)
                    prod_w_norm = prod_w_raw / Z_tot  # Π p_e^w / Π Z_e(w)

                    zero_prod_list.append(prod_no_w)
                    weight_raw_list.append(prod_w_raw)
                    weight_norm_list.append(prod_w_norm)
                    logZ_tot_list.append(logZ.sum())

                    print(f"    Hid {idx_h}  p=[{probs_feat[0]:.4e}, {probs_feat[1]:.4e}, "
                          f"{probs_feat[2]:.4e}, {probs_feat[3]:.4e}]  → Orig_emis*={prod_no_w:.4e}")

                    # —— 3) Copula 对数项 —— #
                    mode = self.params.get('corr_mode', 0)
                    if mode == 0:
                        log_cop_list.append(0.0)
                    else:
                        if mode == 1:
                            R = self.params['corr_mats'][0, 0]
                        elif mode == 2:
                            R = self.params['corr_mats'][z1, z2]
                        elif mode == 3:
                            R = np.eye(E) if (z1 + z2) == 0 else self.params['corr_mats'][0, 0]
                        elif mode == 4:
                            R = np.eye(E) if (z1 + z2) == 0 else self.params['corr_mats'][z1, z2]
                        else:
                            R = np.eye(E)

                        if np.allclose(R, np.eye(E)):  # R=I ⇒ log Copula = 0
                            log_cop_list.append(0.0)
                        else:
                            z_norm = np.clip(norm.ppf(cdfs_feat), -8.0, 8.0)
                            sign, ld = np.linalg.slogdet(R)
                            quad = z_norm @ (np.linalg.inv(R) - np.eye(E)) @ z_norm
                            log_cop = -0.5 * ld - 0.5 * quad
                            log_cop_list.append(log_cop)
                print("  ----------------------------------\n")

                # ---- B) 七种对数发射分数 ----

                # ---- B) 七种对数发射分数（“凡是有权重的，都对 x 做正规化”） ----

                # 〇 原始发射概率（无权重，未归一化）
                zero_prod_arr = np.array(zero_prod_list, dtype=float)

                # ① 无权重对数
                log_unweight_emis = np.log(zero_prod_arr + EPS)

                # ② 加权对数（未除 Z_e，未归一化）
                weight_raw_arr = np.array(weight_raw_list, dtype=float)  # Π p_e(x)^{w_e}
                log_weight_raw = np.log(weight_raw_arr + EPS)  # Log{Π[P(X|Z)^w]}（未除Z_e）

                #    —— 对 x 做正规化 ——
                #    A) 阶段已将每个隐藏状态对应的 Σ_e log Z_e(w_e) 存入 logZ_tot_list
                logZ_tot_arr = np.array(logZ_tot_list, dtype=float)  # Σ_e log Z_e(w_e)
                Z_tot_arr = np.exp(logZ_tot_arr)  # Π_e Z_e(w_e)
                weight_norm_arr = weight_raw_arr / (Z_tot_arr + EPS)  # Π p_e^w / Π Z_e

                # ③ 加权归一化对数（已除 Z_e，未对 z 归一化）
                log_weight_norm_emis = np.where(weight_norm_arr > 0,
                                                np.log(weight_norm_arr), -np.inf)
                #    此时 log_weight_norm_emis ≡ Log{Π[P(X|Z)^w]/ΠZ_e}

                # ④ 全局加权对数（已除 Z_e，未对 z 归一化；先温度 1/v）
                glob_raw_arr = weight_norm_arr ** (1.0 / global_weight)
                log_glob_raw = np.where(glob_raw_arr > 0,
                                        np.log(glob_raw_arr), -np.inf)
                #    这是 (1/v)·Log{Π[P(X|Z)^w]/ΠZ_e}（未对 z 归一化）

                # ⑤ 全局加权对数发射得分（已除 Z_e，已对 z 归一化）——Viterbi 真正使用
                glob_norm_arr = glob_raw_arr / glob_raw_arr.sum()
                log_glob_norm = np.where(glob_norm_arr > 0,
                                         np.log(glob_norm_arr), -np.inf)
                #    这是 (1/v)·Log{Π[P(X|Z)^w]/ΠZ_e}_norm

                # ⑥ Copula 对数项（已在 A) 阶段填入 log_cop_list）
                log_cop_arr = np.array(log_cop_list, dtype=float)

                # ⑦ 全局加权归一化联合对数（含 Copula）
                #     先用 weight_norm_arr（已除 Z_e） 加 log_cop，再温度 1/v，再对 z 归一化
                log_weight_cop = np.where(weight_norm_arr > 0,
                                          np.log(weight_norm_arr), -np.inf) + log_cop_arr
                cop_raw_arr = np.exp(log_weight_cop / global_weight)
                cop_norm_arr = cop_raw_arr / cop_raw_arr.sum()
                log_glob_cop_norm = np.where(cop_norm_arr > 0,
                                             np.log(cop_norm_arr), -np.inf)
                #    这是 (1/v)·Log{Corr·Π[P(X|Z)^w]/ΠZ_e}_norm

                # ----------- 打印 -----------
                # print(f"  发射概率                               ~  Π[P(X|Z)]                               = {[f'{x:.2e}' for x in zero_prod_arr]}")
                # print(f"  对数发射概率                            ~  Log{{Π[P(X|Z)]}}                          = {[f'{x:.4f}' for x in log_unweight_emis]}")
                # print(f"  加权对数发射概率（已除 Z_e）               ~  Log{{Π[P(X|Z)^w]/ΠZ_e}}                    = {[f'{x:.4f}' for x in log_weight_norm_emis]}")
                # print(f"  全局加权对数发射概率（已除 Z_e，未归一化）    ~  (1/v)·Log{{Π[P(X|Z)^w]/ΠZ_e}}                = {[f'{x:.4f}' for x in log_glob_raw]}")
                # print(f"  全局加权对数发射得分（已除 Z_e，已归一化）    ~  (1/v)·Log{{Π[P(X|Z)^w]/ΠZ_e}}_norm           = {[f'{x:.4f}' for x in log_glob_norm]}")
                # print(f"  全局加权归一化联合对数发射概率（含 Copula）   ~  (1/v)·Log{{Corr·Π[P(X|Z)^w]/ΠZ_e}}_norm      = {[f'{x:.4f}' for x in log_glob_cop_norm]}")
                #

                # —— 转移概率输出 ——（保持不变）
                print("")
                for i in range(self.n_hidden_states):
                    trans_mat = self.transition_matrices_tensor[t - 1][i]
                    log_trans = np.log(trans_mat + 1e-12)
                    print(f"  隐藏链 {i} 原始转移概率      P(z_{{{i},t}}|z_{{{i},t-1}})    = {[f'{x:.4f}' for x in trans_mat.flatten()]}")
                    print(f"  隐藏链 {i} 对数转移概率  log[P(z_{{{i},t}}|z_{{{i},t-1}})]   = {[f'{x:.4f}' for x in log_trans.flatten()]}")

                # —— 汇总得分输出 ——（保留旧变量名）
                print("")
                print(f"  对数转移得分（历史最优累积对数概率）  Log_trans         = {Log_trans.flatten()}")
                print(f"  对数全局加权发射得分 (已归一化)      Log_Global_emis   = {log_normalized_global_emis.flatten()}")
                print(f"\n  总得分 Max ( log[P(z_{{{i}, t}}|z_{{{i}, t-1}})] + (1/V)·Σ{{W·Log[P(X|Z)]}} ) :")
                print(f"  Max ( Log_trans + (1/V)·Log_weighted_emis )     = {combined.flatten()}\n")
                chosen = np.unravel_index(np.argmax(combined), combined.shape)
                print(f"  Conclusion:  t={t}: 选中 state={chosen}       score  = {combined.max()}\n")

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

        # ===== NEW : (1,1) 状态后处理（对数化 + 详细调试打印） ===============
        EPS = 1e-6
        for t in range(T):
            if most[0, t] == 1 and most[1, t] == 1:
                vis_val = observed_states[2, t]
                # ----------------- 修改 2：打印真实状态 -----------------
                print(f"[后处理] t={t}, 原状态=(1,1), 真实状态={hidden_states[t]}, 能见度={vis_val:.4f}")
                # ------------------------------------------------------
                if np.isclose(vis_val, 10.0, atol=1e-6):
                    print("         能见度为 10 → 直接替换为 (0,0)")
                    most[0, t], most[1, t] = 0, 0
                    continue

                # —— 计算详细概率（先对数化） ——
                from scipy.stats import norm

                def feat_stats(z1, z2, e):
                    mu = self.mus[z1, z2, e]  # μ, σ 针对 ln X
                    sigma = max(self.sigmas[z1, z2, e], 1e-8)
                    x = observed_states[e, t]
                    lnX = np.log(max(x, EPS))  # 观测值取 ln
                    pdf = norm.pdf(lnX, loc=mu, scale=sigma)
                    lpdf = np.log(max(pdf, EPS))
                    return x, lnX, mu, sigma, pdf, lpdf

                def twofeat_logp(z1, z2):
                    return sum(feat_stats(z1, z2, e)[5] for e in (1, 3))

                logp_10 = twofeat_logp(1, 0)
                logp_01 = twofeat_logp(0, 1)

                # —— 打印详细信息 ——
                print(f"         比较 (1,0) ↔ (0,1)：")
                for state, (z1, z2), lp in [("(1,0)", (1, 0), logp_10),
                                            ("(0,1)", (0, 1), logp_01)]:
                    print(f"           状态 {state} 总 logP = {lp:.4f}")
                    for e in (1, 3):
                        x, lnX, mu, sigma, pdf, lpdf = feat_stats(z1, z2, e)
                        feat_name = '风速' if e == 1 else '相对湿度'
                        print(f"             {feat_name}: "
                              f"x={x:.4f}, lnX={lnX:.4f}, "
                              f"μ={mu:.4f}, σ={sigma:.4f}, "
                              f"pdf={pdf:.4e}, log={lpdf:.4f}")

                chosen = "(1,0)" if logp_10 >= logp_01 else "(0,1)"
                print(f"         结果 → 选 {chosen}")

                # —— 替换 —— vtb
                if logp_10 >= logp_01:
                    most[0, t], most[1, t] = 1, 0
                else:
                    most[0, t], most[1, t] = 0, 1
        # ===== END NEW ======================================================

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
    # 555
    def GetObservedGivenHidden(self, observed_state, observed_weights, n_step):
        """
        计算联合密度 P(x_t | z_t)

        corr_mode:
          0 → 无相关 (R = I)
          1 → 全局 Gaussian Copula
          2 → 状态依赖 Gaussian Copula
          3 → 全局联合正态 (z1+z2==0 时退化独立)
          4 → 状态依赖联合正态 (z1+z2==0 时退化独立)
        其余注释、格式、输出保持不变，可直接覆盖。
        """
        import numpy as np
        from scipy.stats import lognorm, norm

        EPS, MIN_SIGMA, MAX_QUAD = 1e-300, 1e-2, 50
        K, _, E = self.mus.shape

        x_vec      = np.asarray(observed_state, dtype=float)
        log_unnorm = np.full(self.hidden_indices.field_sizes, -np.inf)

        def log_norm_feat(w, mu, sigma):
            """
            计算对数正态 pdf^w 的精确归一化常数 ln Z_LN,PDF(w) (闭式公式)
            对应理论推导中的 Eq. (3.2)。
            """
            w = max(w, 1e-8)
            if np.isclose(w, 1.0):
                return 0.0

            s_sq = sigma ** 2
            w_m1 = w - 1

            # 对应 Eq. (3.2) 中的各项
            term1 = (1 - w) * np.log(sigma)
            term2 = 0.5 * (1 - w) * np.log(2 * np.pi)
            term3 = -0.5 * np.log(w)
            # 修正：根据最终理论推导，mu项的符号为负
            term4 = -w_m1 * mu
            term5 = (w_m1 ** 2 / (2 * w)) * s_sq

            return term1 + term2 + term3 + term4 + term5

        # # —— 辅助：每维归一化常数 ln Z —— #
        # def log_norm_feat(w, mu, sigma):
        #     """对数正态 pdf^w 的精确归一化常数 ln Z_LN(w)（闭式公式）"""
        #     w = max(w, 1e-8)
        #     if np.isclose(w, 1.0):
        #         return 0.0
        #     return ((1 - w) * np.log(sigma)
        #             + (1 - w) * 0.5 * np.log(2 * np.pi)
        #             - 0.5 * np.log(w)
        #             - mu * (w - 1)
        #             + sigma ** 2 * (w - 1) ** 2 / (2 * w))

        # # —— 辅助：每维归一化常数 ln Z —— #
        # def log_norm_feat(w, mu, sigma):
        #     w = max(w, 1e-8)
        #     if np.isclose(w, 1.0):
        #         return 0.0
        #     return ((1 - w) * np.log(sigma)
        #             + ((1 - w) / 2) * np.log(2 * np.pi)
        #             - 0.5 * np.log(w)
        #             + mu + sigma ** 2 / (2 * w))

        # —— 遍历所有隐藏状态 —— #
        for (z1, z2) in self.all_hidden_states:
            hid  = (z1, z2)
            wvec = observed_weights[z1, z2, :]

            logpdf = np.zeros(E)
            logZ   = np.zeros(E)
            uvec   = np.zeros(E)

            mode = self.params.get('corr_mode', 1)

            # —— 1) 逐维边缘密度与归一化常数 —— #
            for e in range(E):
                mu = self.mus[hid][e]
                sigma = max(self.sigmas[hid][e], MIN_SIGMA)
                obs = x_vec[e]
                w = wvec[e]

                # 修正：恢复了区分不同维度的 if/else 结构
                if e in [0, 1, 3]:  # 标准对数正态维度
                    pdf = lognorm(s=sigma, scale=np.exp(mu)).pdf(obs)
                    cdf = lognorm(s=sigma, scale=np.exp(mu)).cdf(obs)
                    logpdf[e] = np.log(max(pdf, EPS))
                    logZ[e] = log_norm_feat(w, mu, sigma)
                    uvec[e] = np.clip(cdf, 1e-9, 1 - 1e-9)

                else:  # 能见度维度 (e=2)
                    pi0 = self.params['zero_inflation_pi']
                    max_vis = self.params.get('max_visibility', 10)

                    # ZI-Lognormal 只在特定状态下激活
                    if hid == (0, 0):
                        # --- 计算边缘 PDF 和 CDF ---
                        if obs == max_vis:
                            pdf = cdf = pi0
                        else:
                            pdf_c = lognorm(s=sigma, scale=np.exp(mu)).pdf(obs)
                            cdf_c = lognorm(s=sigma, scale=np.exp(mu)).cdf(obs)
                            pdf = (1 - pi0) * pdf_c
                            cdf = pi0 + (1 - pi0) * cdf_c

                        logpdf[e] = np.log(max(pdf, EPS))
                        uvec[e] = np.clip(cdf, 1e-9, 1 - 1e-9)

                        # --- 高精度混合近似计算 Z(w) ---
                        d = lognorm(s=sigma, scale=np.exp(mu)).pdf(max_vis)
                        T1 = w * (pi0 ** (w - 1)) * (1 - pi0) * d
                        T2 = 0.5 * w * (w - 1) * (pi0 ** (w - 2)) * ((1 - pi0) ** 2) * (d ** 2)
                        Z_interact = (pi0 ** w) + T1 + T2

                        ln_Z_LN_PDF = log_norm_feat(w, mu, sigma)
                        ln_Z_body = w * np.log(1 - pi0) + ln_Z_LN_PDF
                        Z_body = np.exp(ln_Z_body)

                        Z_hybrid = Z_interact + Z_body
                        logZ[e] = np.log(max(Z_hybrid, EPS))

                    else:  # 非(0,0)状态下的能见度是标准对数正态
                        pdf = lognorm(s=sigma, scale=np.exp(mu)).pdf(obs)
                        cdf = lognorm(s=sigma, scale=np.exp(mu)).cdf(obs)
                        logpdf[e] = np.log(max(pdf, EPS))
                        logZ[e] = log_norm_feat(w, mu, sigma)
                        uvec[e] = np.clip(cdf, 1e-9, 1 - 1e-9)

            # —— 2) 相关矩阵 R 选择 —— #
            if mode == 0:
                R = np.eye(E)

            elif mode == 1:          # 全局 Gaussian Copula
                R = self.params['corr_mats'][0, 0]

            elif mode == 2:          # 状态依赖 Gaussian Copula
                R = self.params['corr_mats'][z1, z2]

            elif mode == 3:          # 全局联合正态 (退化条件)
                R = np.eye(E) if (z1 + z2) == 0 else self.params['corr_mats'][0, 0]

            elif mode == 4:          # 状态依赖联合正态 (退化条件)
                R = np.eye(E) if (z1 + z2) == 0 else self.params['corr_mats'][z1, z2]

            else:
                raise ValueError(f"Invalid corr_mode = {mode}")

            # —— 3) Copula 密度项 —— #
            if not np.allclose(R, np.eye(E)):
                try:
                    z        = np.clip(norm.ppf(uvec), -12.0, 12.0)
                    sign, ld = np.linalg.slogdet(R)
                    if sign <= 0 or np.isnan(ld):
                        raise np.linalg.LinAlgError
                    invR = np.linalg.inv(R)
                    quad = z @ (invR - np.eye(E)) @ z
                    quad = np.clip(quad, -MAX_QUAD, MAX_QUAD)
                    log_cop = -0.5 * ld - 0.5 * quad
                except Exception as err:
                    print(f"[Warn] Copula fallback ({z1},{z2}):", err)
                    log_cop = 0.0
            else:
                log_cop = 0.0

            # —— 4) 合成总对数密度 —— #222
            log_p = (wvec * logpdf).sum() + log_cop - logZ.sum()
            log_p = (wvec * logpdf).sum() - logZ.sum()
            # print("(wvec * logpdf).sum() = ",(wvec * logpdf).sum())
            # print("log_cop = ",log_cop)

            log_unnorm[hid] = log_p

        # —— 5) log-sum-exp 归一化 —— #
        M = np.max(log_unnorm)
        probs = np.exp(log_unnorm - M)

        # probs /= probs.sum()
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
        ---------- EM 算法 M 步 ----------
        * corr_mode 0 / 1 / 3 → 估计单一全局相关矩阵 R_global 并广播
        * corr_mode 2 / 4     → 为每个 (a,b) 独立估计相关矩阵 R_(a,b)
                                (若 gamma_ab 总权重为 0，则回退成 R_global 并打印；
                                 corr_mode=4 且 (a,b)==(0,0) 则强制退化为 I)
        其它逻辑保持原状
        """

        # ===== 0. 通用导入与常量 =====
        import numpy as np
        from numpy.linalg import eigh
        from scipy.stats import norm
        from scipy.optimize import root

        MIN_SIG, MIN_SIG2 = 1e-1, 1e-2
        EIG_EPS, LOG_EPS  = 1e-6, 1e-6
        max_vis = self.params.get('max_visibility', 10)

        # ---- 形状校正 ----
        if observed_states.ndim == 1:
            observed_states = observed_states.reshape(1, -1)
        if observed_states.shape[0] < observed_states.shape[1]:
            X = observed_states
        else:
            X = observed_states.T
        E, T = X.shape

        K = self.params['hidden_alphabet_size']
        HN = self.params['n_hidden_chains']
        π0 = self.params.get('zero_inflation_pi', 0.99)
        mode = self.params.get('corr_mode', 1)

        # ===== 1. 初始分布 / 转移矩阵 =====
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

        # ===== 2. μ、σ =====
        self.mus    = np.zeros((K, K, E))
        self.sigmas = np.zeros_like(self.mus)

        for e in range(E):
            obs = X[e, :]
            for a in range(K):
                for b in range(K):
                    γ = gammas[a, b, :]
                    if γ.sum() == 0:
                        self.mus[a, b, e]    = self.params['mus'][a, b, e]
                        self.sigmas[a, b, e] = self.params['sigmas'][a, b, e]
                        continue

                    if e in [0, 1, 3]:
                        mask  = obs > 0
                        γ_pos = γ[mask]
                        y     = np.log(obs[mask])
                        S     = γ_pos.sum()
                        mu    = (γ_pos * y).sum() / S
                        var   = (γ_pos * (y - mu) ** 2).sum() / S
                    else:  # 能见度
                        if (a, b) == (0, 0):
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
                                α     = (np.log(max_vis) - mu) / sigma
                                Φα    = norm.cdf(α)
                                φα    = norm.pdf(α)

                                dlnF_dmu = -φα / (Φα * sigma)
                                dlnF_ds2 = φα / Φα * (-(np.log(max_vis) - mu) / (2 * sigma ** 3))

                                dQ_dmu = ((γ_cont * (y_cont - mu)) / s2).sum() - S_cont * dlnF_dmu
                                dQ_ds2 = ((γ_cont * ((y_cont - mu) ** 2 / s2 - 1)) / (2 * s2)).sum() - S_cont * dlnF_ds2
                                return [dQ_dmu, dQ_ds2]

                            sol = root(equations, [mu0, np.log(var0)])
                            mu  = sol.x[0]
                            var = np.exp(sol.x[1])
                        else:
                            mask  = obs > 0
                            γ_pos = γ[mask]
                            y     = np.log(obs[mask])
                            S     = γ_pos.sum()
                            mu    = (γ_pos * y).sum() / S
                            var   = (γ_pos * (y - mu) ** 2).sum() / S

                    var = max(var, MIN_SIG2)
                    self.mus[a, b, e]    = mu
                    self.sigmas[a, b, e] = np.sqrt(var)

        # ===== 2-B 绑定雾霾 (1,0) & 沙尘 (0,1) 能见度参数 =====
        # -------- Option A (默认) : 加权平均 ---------- #
        # S_h, S_d = gammas[1, 0].sum(), gammas[0, 1].sum()
        # if S_h + S_d > 0:
        #     μh, μd = self.mus[1, 0, 2], self.mus[0, 1, 2]
        #     σh2, σd2 = self.sigmas[1, 0, 2]**2, self.sigmas[0, 1, 2]**2
        #     μc  = (S_h * μh + S_d * μd) / (S_h + S_d)
        #     σc2 = (S_h * σh2 + S_d * σd2) / (S_h + S_d)
        #     σc  = np.sqrt(max(σc2, MIN_SIG2))
        #     self.mus[1, 0, 2] = self.mus[0, 1, 2] = μc
        #     self.sigmas[1, 0, 2] = self.sigmas[0, 1, 2] = σc

        # -------- Option B : 简单算术平均（若需要请启用） ---------- #
        # μ̄ = (self.mus[0, 1, 2] + self.mus[1, 0, 2]) / 2
        # σ̄ = (self.sigmas[0, 1, 2] + self.sigmas[1, 0, 2]) / 2
        # for (i, j) in [(0, 1), (1, 0)]:
        #     self.mus[i, j, 2]    = μ̄
        #     self.sigmas[i, j, 2] = σ̄

        # ===== 3. 计算一次全局相关矩阵 R_global =====
        X_log_all = np.log(np.clip(X, LOG_EPS, None))                    # (E,T)
        mu_hat    = np.einsum('abe,abt->et', self.mus, gammas)           # (E,T)
        R_e       = X_log_all - mu_hat
        w_t       = gammas.sum(axis=(0, 1))
        cov       = (R_e * w_t) @ R_e.T / w_t.sum()                      # (E,E)

        std = np.sqrt(np.diag(cov))
        std[std < MIN_SIG] = MIN_SIG
        R_global = cov / std[:, None] / std[None, :]
        R_global = np.clip(0.5 * (R_global + R_global.T), -0.999, 0.999)
        vals, _  = eigh(R_global)
        if vals.min() < EIG_EPS:
            R_global += np.eye(E) * (EIG_EPS - vals.min() + 1e-8)

        # ===== 4. 根据 corr_mode 填充 self.corr_mats =====
        if mode in (2, 4):  # 状态依赖分支，含 fallback & degenerate
            self.corr_mats = np.zeros((K, K, E, E))
            for a in range(K):
                for b in range(K):
                    # corr_mode=4 且 (0,0) 强退化
                    if mode == 4 and (a + b) == 0:
                        self.corr_mats[a, b] = np.eye(E)
                        continue
                    γ_ab = gammas[a, b, :]
                    if γ_ab.sum() == 0:
                        print(f"[Info] gamma sum zero at state ({a},{b}), fallback to R_global")
                        self.corr_mats[a, b] = R_global
                        continue

                    R_ab   = X_log_all - self.mus[a, b, :, None]
                    cov_ab = (R_ab * γ_ab) @ R_ab.T / γ_ab.sum()

                    std_ab = np.sqrt(np.diag(cov_ab))
                    std_ab[std_ab < MIN_SIG] = MIN_SIG
                    R_tmp  = cov_ab / std_ab[:, None] / std_ab[None, :]
                    R_tmp  = np.clip(0.5 * (R_tmp + R_tmp.T), -0.999, 0.999)
                    vals_ab, _ = eigh(R_tmp)
                    if vals_ab.min() < EIG_EPS:
                        R_tmp += np.eye(E) * (EIG_EPS - vals_ab.min() + 1e-8)

                    self.corr_mats[a, b] = R_tmp

        else:  # 全局分支，含 mode==0 与 mode==1/3
            R_broadcast = np.eye(E) if mode == 0 else R_global
            self.corr_mats = np.tile(R_broadcast, (K, K, 1, 1)).reshape(K, K, E, E)

            if mode == 3:
                # 只对 (0,0) 这个格子做退化
                self.corr_mats[0, 0] = np.eye(E)

        # ===== 5. 返回 =====
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
            'corr_mode'           : mode,
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

        # 消灭（1，1）
        self.params["mus"][1, 1, :] = 0
        self.params["sigmas"][1, 1, :] = 0.0001

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
            # if log_likelihood - old_log_likelihood < likelihood_precision:

                print("[Converged]  ΔLL = {:.6e}".format(
                    log_likelihood - old_log_likelihood))

                # —— 在此处做最后一次迭代的打印调试 ——
                #    仅打印一次 final_gammas 在 t_debug 时刻各状态的后验及其和
                final_gammas = gammas
                T = final_gammas.shape[2]
                t_debug = T - 1  # 可根据需要修改为 T-1、T//2 等
                np.set_printoptions(precision=6, suppress=True)
                print(f"\n[调试] 收敛时（第 {n_iter} 轮），t = {t_debug} 时刻各隐藏组合后验：")
                for i in range(K):
                    for j in range(K):
                        print(f"  gamma[{i},{j},{t_debug}] = {final_gammas[i, j, t_debug]}")
                print("  sum =", final_gammas[:, :, t_debug].sum(), "\n")

                return (new_params['transition_matrices'],
                        new_params['mus'],
                        new_params['sigmas'],
                        new_params, gammas)

            old_log_likelihood = log_likelihood
            n_iter += 1

            # ---------- 用新参数重新实例化模型 ----------
            H = FullDiscreteFactorialHMM(new_params, self.n_steps, True)

        # 达到迭代上限
        print("[MaxIter]  final LL = {:.6f}".format(old_log_likelihood))
        return (new_params['transition_matrices'], new_params['mus'],
                new_params['sigmas'], new_params, gammas)


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


def _reimplemented_estimate_mi_dd(feature_vec, target_vec):
    """
    估计离散特征和离散目标之间的互信息。
    """
    # 输入校验
    if feature_vec.ndim != 1 or target_vec.ndim != 1:
        raise ValueError("特征向量和目标向量必须是一维的。")
    if len(feature_vec) != len(target_vec):
        raise ValueError("特征向量和目标向量必须具有相同的长度。")

    N = len(target_vec)
    if N == 0:
        return 0.0

    # 构建列联表 (频数)
    classes_y = np.unique(target_vec)
    classes_x = np.unique(feature_vec)

    contingency_table = np.zeros((len(classes_x), len(classes_y)))
    for i, val_x in enumerate(classes_x):
        for j, val_y in enumerate(classes_y):
            contingency_table[i, j] = np.sum((feature_vec == val_x) & (target_vec == val_y))

    # 计算概率 P(x,y), P(x), P(y)
    joint_prob = contingency_table / N
    # P(x) - 边缘概率
    p_x = np.sum(joint_prob, axis=1, keepdims=True)
    # P(y) - 边缘概率
    p_y = np.sum(joint_prob, axis=0, keepdims=True)

    mi = 0.0
    # 避免 log(0) 和除以0
    # 使用 np.where 来保护计算
    # P(x)P(y)
    # p_x 是 (n_classes_x, 1), p_y 是 (1, n_classes_y)
    # p_x @ p_y 得到 (n_classes_x, n_classes_y) 的外积矩阵
    # 或者直接使用广播： p_x * p_y
    denominator = p_x * p_y

    # 只对 joint_prob > 0 的项计算
    terms = np.zeros_like(joint_prob, dtype=float)  # 确保是浮点数类型
    valid_mask = (joint_prob > 1e-12) & (denominator > 1e-12)  # 使用小的epsilon避免浮点问题

    terms[valid_mask] = joint_prob[valid_mask] * np.log(
        joint_prob[valid_mask] / denominator[valid_mask]
    )
    mi = np.sum(terms)

    return max(0.0, mi)  # 确保非负

def _reimplemented_estimate_mi_cd(feature_vec, target_vec, n_neighbors, rng):
    """
    估计连续特征和离散目标之间的互信息。
    使用基于Ross (2014) 和 Kraskov et al. (2004) 的k-NN估计器。
    """
    # 输入校验
    if feature_vec.ndim != 1 or target_vec.ndim != 1:
        raise ValueError("特征向量和目标向量必须是一维的。")
    if len(feature_vec) != len(target_vec):
        raise ValueError("特征向量和目标向量必须具有相同的长度。")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors 必须为正数。")

    N = len(target_vec)
    if N == 0:
        return 0.0

    # 1. 对连续特征进行抖动 (Jittering)
    feature_jittered = feature_vec.copy().astype(np.float64)
    if N > 1:
        sorted_unique_vals = np.sort(np.unique(feature_jittered))
        if len(sorted_unique_vals) < N:  # 存在重复值
            min_diff = np.min(np.diff(sorted_unique_vals)) if len(sorted_unique_vals) > 1 else 1e-5
            # 噪声尺度应非常小，以避免显著改变数据分布
            noise_scale = min_diff * 1e-6 if min_diff > 0 else 1e-10
            if noise_scale == 0:  # 如果所有值都相同，min_diff可能是0
                noise_scale = 1e-10  # 使用一个绝对的小噪声
            feature_jittered += rng.uniform(-noise_scale, noise_scale, size=feature_jittered.shape)

    # Ross 2014 MI estimator: I(X,Y) = ψ(N) - <ψ(Nx)> + ψ(k) - <ψ(mi)>
    # X is discrete (target_vec), Y is continuous (feature_jittered)
    # k is n_neighbors

    psi_N = digamma(N)

    unique_classes, class_counts = np.unique(target_vec, return_counts=True)

    # <ψ(Nx)> term: 对每个样本i，其类别为cx_i，使用Nx_i (该类别的样本数)
    psi_Nx_terms = np.zeros(N, dtype=float)
    class_map_counts = {cls_val: count for cls_val, count in zip(unique_classes, class_counts)}
    for i in range(N):
        Nx_i = class_map_counts[target_vec[i]]
        if Nx_i <= 0:  # 不应发生
            return 0.0
        psi_Nx_terms[i] = digamma(Nx_i)
    avg_psi_Nx = np.mean(psi_Nx_terms) if N > 0 else 0.0

    psi_k = digamma(n_neighbors)

    # <ψ(mi)> term
    psi_mi_terms = np.zeros(N, dtype=float)
    feature_reshaped = feature_jittered.reshape(-1, 1)  # k-NN通常需要2D输入

    for i in range(N):
        current_val_scalar = feature_jittered[i]  # 标量值
        current_class = target_vec[i]

        # 1. 找到当前点在其类别内的第k个近邻距离 (epsilon_half)
        class_mask = (target_vec == current_class)
        feature_in_class = feature_jittered[class_mask]  # 1D array
        num_in_class = len(feature_in_class)

        if num_in_class <= n_neighbors:
            # 如果类内样本数不足以找到k个不同的邻居（不包括自身）
            # 或者等于k个，第k个是自身或最远的点。
            # Ross的论文假设 Nx >= k。sklearn的行为可能是在这种情况下返回0 MI。
            # 为了稳健性，如果无法可靠计算，则该特征的MI可能应为0。
            # 此处我们让该点的psi_mi贡献为psi(1)，这可能导致MI不为0。
            # 一个更接近sklearn的行为可能是如果任何Nx < n_neighbors，则整个特征MI为0。
            # 或者，如果num_in_class <= n_neighbors，则该点的epsilon_half可能无法良好定义
            # 或导致m_i的计算出现问题。
            # 暂定：如果一个类别的点太少，无法找到k个邻居，则该点的mi贡献设为psi(1)
            # 这需要与sklearn的行为仔细对比。
            # 如果num_in_class == 0 (不应发生，因为current_val来自该类)
            # 如果num_in_class <= n_neighbors，则第k个邻居的距离可能定义不佳或非常大。
            psi_mi_terms[i] = digamma(1)  # psi(1) = -EulerMascheroni
            continue

        # 计算到同类中其他所有点的距离
        distances_to_classmates = np.abs(feature_in_class - current_val_scalar)
        distances_to_classmates.sort()  # 排序后，第一个元素是0 (到自身的距离)

        # 第k个邻居的距离。distances_to_classmates是自身。
        # distances_to_classmates[1]是第1近邻。
        # distances_to_classmates[n_neighbors]是第k近邻。
        if n_neighbors >= len(distances_to_classmates):  # 如果k大于类内点数（减1，因为有自身）
            epsilon_half = distances_to_classmates[-1]  # 取最远的距离
        else:
            epsilon_half = distances_to_classmates[n_neighbors]

        # 2. 统计全局数据集中有多少点落在 epsilon_half 距离内 (不含自身)
        # m_i: number of points x_j in X (full dataset) such that ||x_j - x_i|| < epsilon_half
        # (excluding x_i itself)
        # 注意：KSG论文和一些实现对这里的 "<" 或 "<=" 有不同处理。
        # Ross (2014) 描述为 "number of points m_i whose distance from y_i is strictly less than ε_i/2"
        # 这里 epsilon_half 对应 Ross 的 ε_i/2
        count_mi = 0
        for j in range(N):
            if i == j:
                continue
            # 严格小于
            if np.abs(feature_jittered[j] - current_val_scalar) < epsilon_half:
                count_mi += 1

        mi_val = count_mi  # 这就是 m_i

        if mi_val <= 0:
            # 如果 m_i = 0, digamma(0) 是负无穷。
            # 使用 digamma(1) 作为启发式处理，避免错误。
            psi_mi_terms[i] = digamma(1)
        else:
            psi_mi_terms[i] = digamma(mi_val)

    avg_psi_mi = np.mean(psi_mi_terms) if N > 0 else 0.0

    # 互信息计算
    mi = psi_N - avg_psi_Nx + psi_k - avg_psi_mi

    # 根据sklearn的实现，如果任何Nx < n_neighbors，MI可能直接为0。
    # 检查是否有类别的样本数小于n_neighbors
    if np.any(class_counts < n_neighbors) and N > 0:  # 仅当有数据时检查
        # 这是sklearn中一种可能的处理方式，如果某个类别点太少，则该特征MI为0
        # 但这需要通过实验验证sklearn的具体行为。
        # 如果不加此判断，则依赖于上面num_in_class <= n_neighbors时的处理。
        # 为了更接近sklearn的稳健性，如果某个类别的样本数不足k，则该特征的MI可能是0。
        # 这个判断放在这里可能更全局。
        # pass # 暂时不强制设为0，依赖之前的逻辑
        pass

    return max(0.0, mi)  # 确保非负


def reimplemented_mutual_info_classif(X, y, discrete_features='auto',
                                      n_neighbors=3, copy=True,
                                      random_state=None):
    """
    复现 sklearn.feature_selection.mutual_info_classif 的核心逻辑。

    参数:
    X : array-like or sparse matrix, shape (n_samples, n_features)
        特征矩阵。
    y : array-like, shape (n_samples,)
        离散的目标向量。
    discrete_features : 'auto', bool or array-like, default='auto'
        指明哪些特征是离散的。
        - 'auto': 稀疏X视为离散，密集X视为连续。
        - bool: 若为True，所有特征视为离散；若为False，所有特征视为连续。
        - array-like: 布尔掩码或整数索引数组，指明离散特征。
    n_neighbors : int, default=3
        用于连续特征MI估计的k-NN数量。
    copy : bool, default=True
        是否复制输入数据。
    random_state : int, RandomState instance or None, default=None
        用于控制连续特征抖动的随机数生成。

    返回:
    mi_scores : array, shape (n_features,)
        每个特征的估计互信息量。
    """

    # 1. 输入校验 (简化版，实际sklearn有更全面的校验)
    if not isinstance(X, np.ndarray):  # 简单处理，sklearn支持更多类型
        X = np.asarray(X)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim != 1:
        raise ValueError("目标向量y必须是一维的。")

    # 修改这里的校验逻辑：
    if X.shape[0] != y.shape[0]: # 只比较第一个维度 (样本数)
        raise ValueError("X和y的样本数必须一致。")

    n_samples, n_features = X.shape

    # 2. 处理 discrete_features 参数
    if isinstance(discrete_features, str) and discrete_features == 'auto':
        # 简单假设：如果X是密集型numpy数组，则视为连续 (False)
        # sklearn的'auto'对稀疏矩阵有不同行为 (issparse(X))
        # 此处简化为全连续，除非用户明确指定
        is_discrete_mask = np.zeros(n_features, dtype=bool)
    elif isinstance(discrete_features, bool):
        is_discrete_mask = np.full(n_features, discrete_features, dtype=bool)
    else:  # 假设是布尔掩码或索引数组
        discrete_features = np.asarray(discrete_features)
        if discrete_features.dtype == bool:
            if len(discrete_features) != n_features:
                raise ValueError("discrete_features布尔掩码长度与特征数不匹配。")
            is_discrete_mask = discrete_features
        else:  # 假设是索引
            is_discrete_mask = np.zeros(n_features, dtype=bool)
            is_discrete_mask[discrete_features] = True

    # 3. 初始化随机数生成器
    if isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    elif isinstance(random_state, np.random.Generator):  # 或者 RandomState for older numpy
        rng = random_state
    else:  # None or other
        rng = np.random.default_rng()  # 默认行为

    # 4. 如果 copy=True，复制 X
    if copy:
        X_processed = X.copy()
    else:
        X_processed = X

    # 5. 对每个特征计算互信息
    mi_scores = np.zeros(n_features, dtype=float)
    for feature_idx in range(n_features):
        feature_vec = X_processed[:, feature_idx]

        # 检查特征是否全为相同值 (零方差)
        if len(np.unique(feature_vec)) <= 1:
            mi_scores[feature_idx] = 0.0
            continue

        if is_discrete_mask[feature_idx]:
            mi_scores[feature_idx] = _reimplemented_estimate_mi_dd(feature_vec, y)
        else:
            mi_scores[feature_idx] = _reimplemented_estimate_mi_cd(feature_vec, y,
                                                                   n_neighbors, rng)
    return mi_scores

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

        # mi = mutual_info_classif(X, y)

        # 换成自己的函数
        mi = np.zeros((4))
        for i in range(4):
            # print(y.shape)
            # print(X[:, i].shape,"!!!!")
            mi[i] = reimplemented_mutual_info_classif(X[:, i], y)
        #     print("mi[", i, "] = ", mi[i] )
        # print("mi = ", mi)


        # mi = mi / mi.sum()       # 归一化到和为1
        # mi = mi * E              # 放大至和为E111
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
        data = np.clip(data, a_min=0.01, a_max=None)
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

    # 4. 第四簇赋值
    # new_mus[1, 1, :] = 0
    # new_sigmas[1, 1, :] = 0.0001

    mean_of_mus_00 = new_mus[0, 0, :]
    mean_of_mus_10 = new_mus[1, 0, :]
    mean_of_mus_01 = new_mus[0, 1, :]
    new_mus[1, 1, :] = (mean_of_mus_00 + mean_of_mus_10 + mean_of_mus_01) / 3.0

    var_00 = np.square(new_sigmas[0, 0, :])
    var_10 = np.square(new_sigmas[1, 0, :])
    var_01 = np.square(new_sigmas[0, 1, :])
    mean_var_for_11 = (var_00 + var_10 + var_01) / 3.0
    new_sigmas[1, 1, :] = np.sqrt(mean_var_for_11)

    # 打印结果
    print("\ninitial mus:\n", new_mus)
    print("\ninitial sigmas:\n", new_sigmas)

    return new_mus, new_sigmas

def jumpEM_710C(params):
    # 使用 numpy 数组定义最终的转移矩阵（1）
    final_transition_matrices = np.array(
        [[[0.988666, 0.082985],
          [0.011334, 0.917015]],

        [[0.993729, 0.076203],
        [0.006271,    0.923797]]]
    )
    # 使用 numpy 数组定义最终的均值矩阵
    final_mus = np.array(
        [[[3.731474, 0.867081, 2.039281, 3.789505],
          [5.331559, 0.958805, 1.899216, 3.534603]],

        [[4.492667, 0.485278, 1.649536, 4.449508],
        [0.,    0.,    0.,    0.]]]
    )
    # 使用 numpy 数组定义最终的标准差矩阵
    final_sigmas = np.array(
        [[[0.757942, 0.672587, 0.1,      0.560197],
          [0.60209,  0.716927, 0.362592, 0.720981]],

        [[0.668006, 0.488981, 0.438209, 0.133822],
        [0.0001,    0.0001,    0.0001,    0.0001]]]
    )

    final_corr = np.array(
        [[[[0.999, - 0.157998, - 0.072795,  0.179896],
           [-0.157998,  0.999, - 0.09289, - 0.269063],
          [-0.072795, - 0.09289,   0.999, - 0.014101],
         [0.179896, - 0.269063, - 0.014101,  0.999]],

        [[0.999, - 0.157998, - 0.072795,  0.179896],
         [-0.157998,  0.999, - 0.09289, - 0.269063],
         [-0.072795, - 0.09289,    0.999, - 0.014101],
         [0.179896, - 0.269063, - 0.014101,  0.999]]],


       [[[0.999, - 0.157998, - 0.072795,  0.179896],
         [-0.157998,  0.999, - 0.09289, - 0.269063],
         [-0.072795, - 0.09289,   0.999, - 0.014101],
         [0.179896, - 0.269063, - 0.014101,    0.999]],

        [[0.999, - 0.157998, - 0.072795,  0.179896],
         [-0.157998,  0.999, - 0.09289, - 0.269063],
         [-0.072795, - 0.09289,    0.999, - 0.014101],
         [0.179896, - 0.269063, - 0.014101,  0.999]]]]
    )

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

def jumpEM_710L(params):
    # 使用 numpy 数组定义最终的转移矩阵（1）
    final_transition_matrices = np.array(
        [[[0.987268, 0.078436],
          [0.012732, 0.921564]],

        [[0.996362, 0.072218],
        [0.003638,    0.927782]]]
    )
    # 使用 numpy 数组定义最终的均值矩阵
    final_mus = np.array(
        [[[3.740922, 0.873675, 2.079442, 3.783451],
          [5.552873, 1.099156, 1.870403, 3.327609]],

        [[4.577315, 0.46709,  1.678273, 4.412475],
        [0.,   0.,    0.,    0.]]]
    )
    # 使用 numpy 数组定义最终的标准差矩阵
    final_sigmas = np.array(
        [[[0.761527, 0.67155,  0.01,     0.563157],
          [0.609935, 0.742934, 0.386227, 0.725533]],

        [[0.65119,  0.476297, 0.431108, 0.166309],
        [0.0001,    0.0001,    0.0001,    0.0001]]]
    )

    final_corr = np.array(
        [[[[1.,        0.,        0.,        0.],
           [0.,        1.,        0.,        0.],
          [0.,        0.,        1.,        0.],
         [0.,        0.,        0.,        1.]],

    [[0.999, - 0.151388, - 0.058939,  0.166789],
     [-0.151388,  0.999, - 0.098016, - 0.258643],
    [-0.058939, - 0.098016,    0.999, - 0.008955],
    [0.166789, - 0.258643, - 0.008955,  0.999]]],


    [[[0.999, - 0.151388, - 0.058939,  0.166789],
      [-0.151388,  0.999, - 0.098016, - 0.258643],
     [-0.058939, - 0.098016,  0.999, - 0.008955],
    [0.166789, - 0.258643, - 0.008955,    0.999]],

    [[0.999, - 0.151388, - 0.058939,  0.166789],
     [-0.151388,  0.999, - 0.098016, - 0.258643],
    [-0.058939, - 0.098016,    0.999, - 0.008955],
    [0.166789, - 0.258643, - 0.008955,  0.999]]]]
    )

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

# 7.6 copula 近似 归一化有corr sigma下限提到0.1 二次分类 风速0=》0.01
def jumpEM_0(params):
    # 使用 numpy 数组定义最终的转移矩阵（1）
    final_transition_matrices = np.array(
        [[[0.988879, 0.083243],
          [0.011121, 0.916757]],

        [[0.993571, 0.076199],
         [0.006429,  0.923801]]]
    )
    # 使用 numpy 数组定义最终的均值矩阵
    final_mus = np.array(
        [[[3.732187, 0.867291, 2.035432, 3.789277],
          [5.32199,  0.943938, 1.896708, 3.556082]],

         [[4.48515,  0.485383, 1.64477,  4.453697],
          [0.     ,  0.      , 0.     ,  0.]]]
    )
    # 使用 numpy 数组定义最终的标准差矩阵
    final_sigmas = np.array(
        [[[0.75826,  0.672665, 0.1,      0.560638],
          [0.599267, 0.714553, 0.36052,  0.718432]],

         [[0.671158, 0.490012, 0.439362, 0.130474],
          [0.0001,    0.0001,    0.0001,    0.0001]]]
    )

    final_corr = np.array(
        [[[[0.999, - 0.158461, - 0.072513,  0.176278],
           [-0.158461,  0.999, - 0.092167, - 0.270479],
           [-0.072513, - 0.092167,  0.999, - 0.013922],
           [0.176278, - 0.270479, - 0.013922,  0.999]],

          [[0.999, - 0.158461, - 0.072513,  0.176278],
           [-0.158461,  0.999, - 0.092167, - 0.270479],
           [-0.072513, - 0.092167,    0.999, - 0.013922],
           [0.176278, - 0.270479, - 0.013922,  0.999]]],


         [[[0.999, - 0.158461, - 0.072513,  0.176278],
           [-0.158461,  0.999, - 0.092167, - 0.270479],
           [-0.072513, - 0.092167,  0.999, - 0.013922],
           [0.176278, - 0.270479, - 0.013922,    0.999]],

          [[0.999, - 0.158461, - 0.072513,  0.176278],
           [-0.158461,  0.999, - 0.092167, - 0.270479],
           [-0.072513, - 0.092167,    0.999, - 0.013922],
           [0.176278, - 0.270479, - 0.013922,  0.999]]]]
    )

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

# 7.6 联合对数正态 近似 归一化有corr sigma下限提到0.1 风速0=》0.01
def jumpEM_00(params):
    # 使用 numpy 数组定义最终的转移矩阵（1）
    final_transition_matrices = np.array(
        [[[0.987268, 0.078438],
          [0.012732, 0.921562]],

         [[0.996362, 0.072224],
          [0.003638, 0.927776]]]
    )
    # 使用 numpy 数组定义最终的均值矩阵
    final_mus = np.array(
        [[[3.740927, 0.873673, 2.079442, 3.783452],
          [5.552866, 1.099161, 1.870389, 3.327606]],

         [[4.57731,  0.467091, 1.678265, 4.41248],
          [0.,    0.,    0.,    0.]]]

    )
    # 使用 numpy 数组定义最终的标准差矩阵
    final_sigmas = np.array(
        [[[0.761531, 0.67155,  0.01,     0.563157],
          [0.609938, 0.742931, 0.386226, 0.725529]],

         [[0.651194, 0.476298, 0.431106, 0.166304],
          [0.0001,    0.0001,    0.0001,    0.0001]]]
    )

    final_corr = np.array(
        [[[[1.,        0.,        0.,        0.],
           [0.,        1.,        0.,        0.],
           [0.,        0.,        1.,        0.],
           [0.,        0.,        0.,        1.]],

         [[0.999, - 0.15139, - 0.058941,  0.166792],
          [-0.15139,   0.999, - 0.098015, - 0.258643],
          [-0.058941, - 0.098015,    0.999, - 0.008958],
          [0.166792, - 0.258643, - 0.008958,  0.999]]],


        [[[0.999, - 0.15139, - 0.058941,  0.166792],
          [-0.15139,   0.999, - 0.098015, - 0.258643],
          [-0.058941, - 0.098015,  0.999, - 0.008958],
          [0.166792, - 0.258643, - 0.008958,    0.999]],

         [[0.999, - 0.15139, - 0.058941,  0.166792],
          [-0.15139,   0.999, - 0.098015, - 0.258643],
          [-0.058941, - 0.098015,    0.999, - 0.008958],
          [0.166792, - 0.258643, - 0.008958,  0.999]]]]
    )

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

# 6.18 copula 近似 归一化有corr sigma下限提到0.1
def jumpEM_1(params):
    # 使用 numpy 数组定义最终的转移矩阵（1）
    final_transition_matrices = np.array(
        [[[0.988666, 0.082986],
          [0.011334, 0.917014]],

         [[0.993729, 0.076203],
          [0.006271,  0.923797]]]
    )
    # 使用 numpy 数组定义最终的均值矩阵
    final_mus = np.array(
        [[[3.731476, 0.86708,  2.039285, 3.789507],
          [5.331559, 0.958805, 1.899215, 3.534603]],

         [[4.492665, 0.485278, 1.649527, 4.449509],
          [0.      , 0.      , 0.      , 0.      ]]]
    )
    # 使用 numpy 数组定义最终的标准差矩阵
    final_sigmas = np.array(
        [[[0.757942, 0.672587, 0.1,      0.560197],
          [0.602089, 0.716928, 0.362592, 0.720984]],

         [[0.668009, 0.488982, 0.438206, 0.133822],
          [0.0001  ,   0.0001,   0.0001, 0.0001]]]
    )

    final_corr = np.array(
        [[[[0.999, - 0.157998, - 0.072797,  0.179899],
           [-0.157998,  0.999, - 0.09289, - 0.269063],
           [-0.072797, - 0.09289,   0.999, - 0.014103],
           [0.179899, - 0.269063, - 0.014103,  0.999]],

          [[0.999, - 0.157998, - 0.072797,  0.179899],
           [-0.157998,  0.999, - 0.09289, - 0.269063],
           [-0.072797, - 0.09289,    0.999, - 0.014103],
           [0.179899, - 0.269063, - 0.014103,  0.999]]],


         [[[0.999, - 0.157998, - 0.072797,  0.179899],
           [-0.157998,  0.999, - 0.09289, - 0.269063],
           [-0.072797, - 0.09289,   0.999, - 0.014103],
           [0.179899, - 0.269063, - 0.014103,    0.999]],

          [[0.999, - 0.157998, - 0.072797,  0.179899],
           [-0.157998,  0.999, - 0.09289, - 0.269063],
           [-0.072797, - 0.09289,    0.999, - 0.014103],
           [0.179899, - 0.269063, - 0.014103,  0.999]]]]
    )

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

# 6.7 copula 近似 归一化有corr
def jumpEM_3(params):
    # 使用 numpy 数组定义最终的转移矩阵（1）
    final_transition_matrices = np.array(
        [[[0.988003, 0.07936],
          [0.011997, 0.92064]],

        [[0.996228, 0.061228],
        [0.003772, 0.938772]]]
    )
    # 使用 numpy 数组定义最终的均值矩阵
    final_mus = np.array(
        [[[3.738947, 0.863086, 2.079442, 3.798223],
          [5.43407,  1.113318, 1.929871, 3.284826]],

         [[4.556062, 0.486711, 1.64626,  4.41506],
          [0.       , 0.      ,  0.    ,   0.]]]
    )
    # 使用 numpy 数组定义最终的标准差矩阵
    final_sigmas = np.array(
        [[[0.759687, 0.670413, 0.01,     0.553256],
          [0.643555, 0.745375, 0.382298, 0.757118]],

         [[0.663594, 0.481469, 0.421816, 0.165245],
          [0.0001,   0.0001,   0.0001,   0.0001]]]
    )

    final_corr = np.array([
        [[[0.999, - 0.161454, - 0.081993,  0.210357],
          [-0.161454,  0.999, - 0.086596, - 0.268643],
          [-0.081993, - 0.086596,  0.999, - 0.023654],
          [0.210357, - 0.268643, - 0.023654,  0.999]],

         [[0.999, - 0.161454, - 0.081993,  0.210357],
          [-0.161454,  0.999, - 0.086596, - 0.268643],
          [-0.081993, - 0.086596,    0.999, - 0.023654],
          [0.210357, - 0.268643, - 0.023654,  0.999]]],


        [[[0.999, - 0.161454, - 0.081993,  0.210357],
          [-0.161454,  0.999, - 0.086596, - 0.268643],
          [-0.081993, - 0.086596,  0.999, - 0.023654],
          [0.210357, - 0.268643, - 0.023654,    0.999]],

         [[0.999, - 0.161454, - 0.081993,  0.210357],
          [-0.161454,  0.999, - 0.086596, - 0.268643],
          [-0.081993, - 0.086596,   0.999, - 0.023654],
          [0.210357, - 0.268643, - 0.023654,  0.999]]]
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

# 6.18 联合正态 近似 归一化有corr sigma下限提到0.1
def jumpEM_2(params):
    # 使用 numpy 数组定义最终的转移矩阵（1）
    final_transition_matrices = np.array(
        [[[0.983339, 0.088579],
          [0.016661, 0.911421]],

        [[0.992635, 0.07648],
        [0.007365,  0.92352]]]
    )
    # 使用 numpy 数组定义最终的均值矩阵
    final_mus = np.array(
        [[[3.684421, 0.930661, 2.041168, 3.720292],
          [5.101427, 0.814372, 1.689862, 3.89814]],

        [[4.541776, 0.333115, 1.95536,  4.437392],
        [0.,    0.,    0.,    0.]]]
    )
    # 使用 numpy 数组定义最终的标准差矩阵
    final_sigmas = np.array(
        [[[0.761913, 0.667234, 0.1,      0.565132],
          [0.928876, 0.667279, 0.416092, 0.650375]],

        [[0.384321, 0.409499, 0.430346, 0.117733],
        [0.0001,    0.0001,    0.0001,    0.0001]]]
    )

    final_corr = np.array(
        [[[[1.,        0.,        0.,        0.],
           [0.,        1.,        0.,        0.],
           [0.,        0.,        1.,        0.],
           [0.,        0.,        0.,        1.]],

          [[0.999, - 0.09587, - 0.022691, - 0.019911],
           [-0.09587,   0.999, - 0.087673, - 0.225149],
          [-0.022691, - 0.087673,  0.999, - 0.036515],
         [-0.019911, - 0.225149, - 0.036515,  0.999]]],


          [[[0.999, - 0.09587, - 0.022691, - 0.019911],
           [-0.09587,   0.999, - 0.087673, - 0.225149],
          [-0.022691, - 0.087673,  0.999, - 0.036515],
         [-0.019911, - 0.225149, - 0.036515,  0.999]],

           [[0.999, - 0.09587, - 0.022691, - 0.019911],
            [-0.09587, 0.999, - 0.087673, - 0.225149],
            [-0.022691, - 0.087673, 0.999, - 0.036515],
            [-0.019911, - 0.225149, - 0.036515, 0.999]]]]
    )



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

# 6.7 联合正态 近似 归一化有corr
def jumpEM_4(params):
    # 使用 numpy 数组定义最终的转移矩阵（1）
    final_transition_matrices = np.array(
        [[[0.987268, 0.078438],
          [0.012732, 0.921562]],

        [[0.996362, 0.072224],
        [0.003638,    0.927776]]]
    )
    # 使用 numpy 数组定义最终的均值矩阵
    final_mus = np.array(
        [[[3.740927, 0.873673, 2.079442, 3.783452],
          [5.552866, 1.099161, 1.870389, 3.327606]],

         [[4.57731,  0.467091, 1.678265, 4.41248],
          [0.,       0.,       0.,       0.]]]
    )
    # 使用 numpy 数组定义最终的标准差矩阵
    final_sigmas = np.array(
        [[[0.761531, 0.67155,  0.01,     0.563157],
          [0.609938, 0.742931, 0.386226, 0.725529]],

         [[0.651194, 0.476298, 0.431106, 0.166304],
          [0.0001,    0.0001,    0.0001,    0.0001]]]
    )

    final_corr = np.array(
        [[[[1.,        0.,        0.,        0.],
           [0.,        1.,        0.,        0.],
           [0.,        0.,        1.,        0.],
           [0.,        0.,        0.,        1.]],

          [[0.999, - 0.15139, - 0.058941,  0.166792],
           [-0.15139,   0.999, - 0.098015, - 0.258643],
           [-0.058941, - 0.098015,    0.999, - 0.008958],
           [0.166792, - 0.258643, - 0.008958,  0.999]]],


         [[[0.999, - 0.15139, - 0.058941,  0.166792],
           [-0.15139,   0.999, - 0.098015, - 0.258643],
           [-0.058941, - 0.098015,  0.999, - 0.008958],
           [0.166792, - 0.258643, - 0.008958,    0.999]],

          [[0.999, - 0.15139, - 0.058941,  0.166792],
           [-0.15139,   0.999, - 0.098015, - 0.258643],
           [-0.058941, - 0.098015,    0.999, - 0.008958],
           [0.166792, - 0.258643, - 0.008958,  0.999]]]]
    )



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
    print("\nTrue Positive Rate (TPR) of event differentiation = \n", TPR)
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

def test_lognormality(observed_states, alpha: float = 0.05, verbose: bool = True):
    import pandas as pd
    from scipy import stats

    """
    对四个观测通道逐一执行对数正态性检验。

    参数
    ----
    observed_states : ndarray, shape (E, T)
        四行观测序列，每一行为同一物理量的时间序列，要求全为正数。
    alpha : float, default 0.05
        显著性水平，用于判定“拒绝 / 不拒绝”对数正态性。
    verbose : bool, default True
        若为 True，则在控制台打印检验结果汇总；总是返回 DataFrame。

    返回
    ----
    results_df : pandas.DataFrame, shape (4, 8)
        行索引为观测通道 (0–3)，列包括四种检验的统计量与 p-value。
    """
    # ------- 0. 输入校验 -------
    if observed_states.ndim != 2 or observed_states.shape[0] != 4:
        raise ValueError("observed_states 需为形状 (4, T) 的二维数组")

    E, T = observed_states.shape
    channel_names = ["PM10", "风速", "能见度", "相对湿度"]  # 如需自定义可修改

    # ------- 1. 初始化结果容器 -------
    cols = ["Shapiro_W", "Shapiro_p",
            "Anderson_A2", "Anderson_p",
            "DAgostino_K2", "DAgostino_p",
            "KS_D", "KS_p"]
    results = pd.DataFrame(index=channel_names, columns=cols, dtype=float)

    # ------- 2. 循环四个观测量 -------
    for idx in range(E):
        x = observed_states[idx, :].astype(float)
        x = x[np.isfinite(x) & (x > 0)]  # 剔除非正或 NaN
        y = np.log(x)  # 对数变换

        # —— 2-a Shapiro–Wilk —— #
        W, p_sw = stats.shapiro(y)
        results.iloc[idx, results.columns.get_loc("Shapiro_W")] = W
        results.iloc[idx, results.columns.get_loc("Shapiro_p")] = p_sw

        # —— 2-b Anderson–Darling —— #
        A2, crit_vals, sig_levels = stats.anderson(y, dist="norm")
        # SciPy 无直接 p 值，采用 Stephens (1974) 近似
        # p ≈ exp(1.2937 − 5.709·A2 + 0.0186·A2²) (A2 ∈ [0.1, 13])
        if A2 < 0.6:
            p_ad = np.exp(-13.436 + 101.14 * A2 - 223.73 * A2 ** 2)
        elif A2 < 13:
            p_ad = np.exp(1.2937 - 5.709 * A2 + 0.0186 * A2 ** 2)
        else:
            p_ad = 0.0
        p_ad = float(np.clip(p_ad, 0.0, 1.0))
        results.iloc[idx, results.columns.get_loc("Anderson_A2")] = A2
        results.iloc[idx, results.columns.get_loc("Anderson_p")] = p_ad

        # —— 2-c D’Agostino K² —— #
        K2, p_k2 = stats.normaltest(y)
        results.iloc[idx, results.columns.get_loc("DAgostino_K2")] = K2
        results.iloc[idx, results.columns.get_loc("DAgostino_p")] = p_k2

        # —— 2-d Kolmogorov–Smirnov (Lilliefors) —— #
        mu, sigma = np.mean(y), np.std(y, ddof=0)
        D, p_ks = stats.kstest(y, 'norm', args=(mu, sigma))
        results.iloc[idx, results.columns.get_loc("KS_D")] = D
        results.iloc[idx, results.columns.get_loc("KS_p")] = p_ks

    # ------- 3. 打印汇总表（可选） -------
    if verbose:
        print("\n—— 对数正态性检验结果 (α = {:.2f}) ——".format(alpha))
        for idx, row in results.iterrows():
            decisions = ["拒绝" if row[pcol] < alpha else "不拒绝"
                         for pcol in ["Shapiro_p", "Anderson_p", "DAgostino_p", "KS_p"]]
            print(f"{idx:<6} | SW_p={row.Shapiro_p:6.4f} ({decisions[0]}) | "
                  f"AD_p={row.Anderson_p:6.4f} ({decisions[1]}) | "
                  f"K2_p={row.DAgostino_p:6.4f} ({decisions[2]}) | "
                  f"KS_p={row.KS_p:6.4f} ({decisions[3]})")
        print("说明：'拒绝' 表示在显著性水平 α 拒绝对数正态性假设。\n")

    return results

def F1_score__Confusion_Matrix__ARI(hidden_states, most_likely_hidden_state):
    from sklearn.metrics import confusion_matrix, f1_score
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
           predicted_states[t] = 0  # 或者根据业务需要，丢弃它/合并到某一类


    # —— 在这里插入混淆矩阵的输出 —— #
    labels_for_cm = sorted(np.unique(np.concatenate((hidden_states, predicted_states))))
    cm = confusion_matrix(hidden_states, predicted_states, labels=labels_for_cm)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"标签: [晴朗, 雾霾, 沙尘]")


    # ARI
    from sklearn.metrics import adjusted_rand_score

    # —— 计算并返回 ARI —— #
    ari_value = adjusted_rand_score(hidden_states, predicted_states)
    print(f"\nARI = {ari_value:.4f}")


    # --- 新增: 计算各类别F1, 宏F1, 微F1 ---
    num_classes = len(labels_for_cm)
    tp = np.zeros(num_classes)            # 真阳性
    fp = np.zeros(num_classes)            # 假阳性
    fn = np.zeros(num_classes)            # 假阴性
    tn = np.zeros(num_classes)            # 真阴性 (新添加)
    f1_per_class = np.zeros(num_classes)  # 每个类别的F1分数

    # 存储每个类别的 TP, FP, FN, TN 值
    performance_metrics_per_class = {}

    print("\n--- 各类别性能指标 (TP, FP, FN, TN) ---")
    for i in range(num_classes):
        current_label = labels_for_cm[i]  # 获取当前处理的类别标签值

        tp[i] = cm[i, i]
        fp[i] = np.sum(cm[:, i]) - cm[i, i]  # 第i列的和减去TP[i]
        fn[i] = np.sum(cm[i, :]) - cm[i, i]  # 第i行的和减去TP[i]

        # 计算 TN (真阴性)
        # TN 是所有不属于当前类别 i，也没有被预测为类别 i 的样本数
        # TN = 总样本数 - (TP + FP + FN)
        # 总样本数可以从混淆矩阵的总和得到
        total_samples = np.sum(cm)
        tn[i] = total_samples - (tp[i] + fp[i] + fn[i])

        performance_metrics_per_class[current_label] = {
            'TP': int(tp[i]),
            'FP': int(fp[i]),
            'FN': int(fn[i]),
            'TN': int(tn[i])
        }
        print(f"类别 {current_label}: TP={int(tp[i])}, FP={int(fp[i])}, FN={int(fn[i])}, TN={int(tn[i])}")

        precision_i = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
        recall_i = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0

        if (precision_i + recall_i) > 0:
            f1_per_class[i] = 2 * (precision_i * recall_i) / (precision_i + recall_i)
        else:
            f1_per_class[i] = 0

    print("\n--- F1 分数 ---")
    for i, label_val in enumerate(labels_for_cm):
        print(f"类别 {label_val} 的 F1 分数: {f1_per_class[i]:.4f}")

    # 计算宏F1 (Macro-F1)
    macro_f1 = np.mean(f1_per_class)
    print(f"\n宏F1 (Macro-F1) 分数: {macro_f1:.4f}")

    # 计算微F1 (Micro-F1)
    total_tp = np.sum(tp)
    total_fp = np.sum(fp)
    # total_fn = np.sum(fn) # 对于混淆矩阵，total_fp 通常等于 total_fn

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + np.sum(fn)) if (total_tp + np.sum(fn)) > 0 else 0  # 与 micro_precision 相同

    if (micro_precision + micro_recall) > 0:
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
    else:
        micro_f1 = 0
    print(f"微F1 (Micro-F1) 分数: {micro_f1:.4f}")

    # # 使用sklearn验证微F1
    # micro_f1_sklearn = f1_score(hidden_states, predicted_states, average='micro', labels=labels_for_cm, zero_division=0)
    # print(f"微F1 (Micro-F1) 分数 (sklearn): {micro_f1_sklearn:.4f}")

def compute_ROC(gammas, hidden_states, target_label):
    """
    计算指定“原始隐藏状态”（0, 1 或 2）的 ROC 曲线和 AUC 分数。

    参数：
    ------
    gammas : np.ndarray, shape = (2, 2, T)
        EM 算法得到的归一化后验概率张量。索引方式为 gamma[z1, z2, t]，
        其中 z1, z2 ∈ {0,1}，T 为时间长度。合法组合：
          - 原始状态 0 ↔ (Z1=0, Z2=0)
          - 原始状态 1 ↔ (Z1=1, Z2=0)
          - 原始状态 2 ↔ (Z1=0, Z2=1)
        （这里假设本模型不会出现 (1,1) 组合，或其概率为 0。）

    hidden_states : np.ndarray, shape = (T,)
        真实的一维隐藏标签数组，取值 ∈ {0,1,2}。
        必须与 hidden_encoded() 的映射一致：
            0 → [0, 0]，1 → [1, 0]，2 → [0, 1]

    target_label : int, ∈ {0, 1, 2}
        希望绘制 ROC 的“原始隐藏状态”编号。例如 target_label=1 时，
        实际表示 (Z1=1, Z2=0) 这一种组合的后验。

    返回：
    ------
    fpr : np.ndarray
        假正例率 (False Positive Rate) 数组。
    tpr : np.ndarray
        真正例率 (True Positive Rate) 数组。
    thresholds : np.ndarray
        对应的概率阈值数组。
    auc_score : float
        该状态下的 ROC 曲线下面积 (Area Under Curve)。
    """

    # —— 1. 校验 gammas 维度 ——
    if gammas.ndim != 3 or gammas.shape[0] != 2 or gammas.shape[1] != 2:
        raise ValueError(f"gammas 的形状应为 (2,2,T)，实际为 {gammas.shape}。")

    # —— 2. 校验 target_label ——
    if target_label not in (0, 1, 2):
        raise ValueError(f"target_label 必须是 0、1 或 2，实际为 {target_label}。")

    # —— 3. 对应“原始标签”→(Z1,Z2) 组合 ——
    #    0 → (0,0)，1 → (1,0)，2 → (0,1)
    if target_label == 0:
        i_star, j_star = 0, 0
    elif target_label == 1:
        i_star, j_star = 1, 0
    else:  # target_label == 2
        i_star, j_star = 0, 1

    # —— 4. 构造“预测分数” scores 与“真实二分类标签” labels ——
    #    对每个 t：score[t] = gammas[i*,j*,t]；label[t] = 1 if hidden_states[t]==target_label else 0
    T = gammas.shape[2]
    scores = gammas[i_star, j_star, :]                    # shape = (T,)
    labels = (hidden_states == target_label).astype(int)  # shape = (T,)

    # —— 5. 调用 sklearn.metrics 计算 ROC、AUC ——
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, thresholds, auc_score

def compute_and_plot_three_ROCs(gammas, hidden_states):
    """
    分别计算 target_label=0,1,2 三条 ROC，并在一个 1×3 子图中绘制，所有文字为英文。
    同时在控制台输出每个状态的 AUC。
    """

    # —— 1. 逐状态调用 compute_ROC，收集结果并打印 AUC ——
    roc_data = {}
    for k in (0, 1, 2):
        fpr, tpr, thr, auc_sc = compute_ROC(gammas, hidden_states, target_label=k)
        roc_data[k] = {'fpr': fpr, 'tpr': tpr, 'auc': auc_sc}
        print("", f"State {k} AUC = {auc_sc:.6f}")

    # —— 2. 创建一个 1 行 3 列的子图布局 ——
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)
    fig.suptitle("ROC Curves for Each State", fontsize=16)

    state_names = {0: "(Z1=0, Z2=0)", 1: "(Z1=1, Z2=0)", 2: "(Z1=0, Z2=1)"}
    for idx, k in enumerate((0, 1, 2)):
        ax = axes[idx]
        ax.plot(roc_data[k]['fpr'], roc_data[k]['tpr'],
                lw=2, label=f"State {k} {state_names[k]}, AUC={roc_data[k]['auc']:.3f}")
        ax.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC for State {k}")
        ax.legend(loc="lower right")
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.90])  # leave space for suptitle
    plt.show()

def compute_and_plot_multiclass_ROC(gammas, hidden_states):
    """
    计算多类别（0,1,2）的 ROC-AUC（One‐vs‐Rest 宏平均与微平均），
    并画出每个类别及宏/微平均 ROC 曲线，所有文字为英文。
    同时在控制台输出各类别、宏平均及微平均的 AUC。
    """

    # —— 1. 将 gammas 转成 (T, 3) 格式的预测概率矩阵 pred_probs ——
    #    每行 [ P(state=0|t), P(state=1|t), P(state=2|t) ]
    T = gammas.shape[2]
    pred_probs = np.zeros((T, 3), dtype=float)
    pred_probs[:, 0] = gammas[0, 0, :]  # state 0 ↔ (0,0)
    pred_probs[:, 1] = gammas[1, 0, :]  # state 1 ↔ (1,0)
    pred_probs[:, 2] = gammas[0, 1, :]  # state 2 ↔ (0,1)

    # —— 2. 将 hidden_states 做 One-Hot 编码，得到 y_true_oh (T,3) ——
    y_true_oh = label_binarize(hidden_states, classes=[0, 1, 2])  # shape = (T,3)

    # —— 3. 计算一对其余（One-vs-Rest）下每个类别的 fpr,tpr,auc，并打印 AUC ——
    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}
    for i in range(3):
        fpr_i, tpr_i, _ = roc_curve(y_true_oh[:, i], pred_probs[:, i])
        auc_i = auc(fpr_i, tpr_i)
        fpr_dict[i] = fpr_i
        tpr_dict[i] = tpr_i
        auc_dict[i] = auc_i
        print('')
        print(f"State {i} AUC = {auc_i:.6f}")

    # —— 4. 计算宏平均（macro-average）ROC 曲线并打印 ——
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= 3  # average
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    auc_dict["macro"] = auc(all_fpr, mean_tpr)
    print('')
    print(f"Macro-average AUC = {auc_dict['macro']:.6f}")

    # —— 5. 计算微平均（micro-average）ROC 曲线并打印 ——
    y_true_flat = y_true_oh.ravel()
    pred_flat   = pred_probs.ravel()
    fpr_micro, tpr_micro, _ = roc_curve(y_true_flat, pred_flat)
    auc_micro = auc(fpr_micro, tpr_micro)
    fpr_dict["micro"] = fpr_micro
    tpr_dict["micro"] = tpr_micro
    auc_dict["micro"] = auc_micro
    print(f"Micro-average AUC = {auc_dict['micro']:.6f}", '\n')

    # —— 6. 绘制多类别 ROC 曲线 ——
    plt.figure(figsize=(8, 8), dpi=100)
    plt.plot(fpr_micro, tpr_micro,
             linestyle=':', linewidth=3,
             label=f"Micro-average ROC (AUC = {auc_micro:.3f})")

    plt.plot(all_fpr, mean_tpr,
             linestyle='-.', linewidth=3,
             label=f"Macro-average ROC (AUC = {auc_dict['macro']:.3f})")

    colors = ['aqua', 'darkorange', 'cornflowerblue']
    class_labels = {0: "(Z1=0, Z2=0)", 1: "(Z1=1, Z2=0)", 2: "(Z1=0, Z2=1)"}
    for i, color in zip(range(3), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=2,
                 label=f"Class {i} {class_labels[i]} (AUC = {auc_dict[i]:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("Multi-class_ROC_Curve_Gaussian_copula.png", dpi=450, bbox_inches='tight', pad_inches=0.01)
    plt.show()

# 三维图像，联合调整全局权重 v 与观测值权重 w。F1 Score
def plot_3d_macro_micro_F1(hidden_states, observed_states, weights, numb):
    """
    3D parameter landscapes for FHMM classification:
      - Fig1: Macro-F1 surface
      - Fig2: Micro-F1 surface
      - Fig3: Macro-F1 & Micro-F1 overlaid
    X-axis : global_weight  (transition-vs-emission balance)
    Y-axis : row-sum of weights
    Z-axis : F1 scores
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import f1_score

    # 1. 参数范围
    global_weight_range = np.linspace(0.1, 6.5, numb)    # X 9999
    row_sum_range       = np.linspace(0.7, 3, numb)    # Y

    # global_weight_range = np.linspace(0.1, 5, numb)    # X 9999
    # row_sum_range       = np.linspace(0.5, 5, numb)    # Y

    # 2. 计算 Macro-F1 / Micro-F1
    perf = np.zeros((2, len(row_sum_range), len(global_weight_range)))
    for i, row_sum in enumerate(row_sum_range):
        final_weights = weights * row_sum
        for j, global_weight in enumerate(global_weight_range):
            most, _, _ = H.Viterbi(
                observed_states, final_weights, global_weight, hidden_states
            )
            # decode
            T, pred = hidden_states.shape[0], np.zeros(hidden_states.shape[0], dtype=int)
            for t in range(T):
                pred[t] = 0 if (most[0, t] == 0 and most[1, t] == 0) else (2 if most[0, t] == 0 else 1)
            perf[0, i, j] = f1_score(hidden_states, pred, average="macro")
            perf[1, i, j] = f1_score(hidden_states, pred, average="micro")

    # 3. 统一风格
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'font.size': 14,
        'axes.labelsize': 12,
        'axes.titlesize': 14
    })
    X, Y = np.meshgrid(global_weight_range, row_sum_range)

    # ========== 单曲面绘图函数（供 Fig1 / Fig2 调用） ==========
    # ========== 单曲面绘图函数（修正版） ==========
    def plot_surface(ax, Z, zlabel, cmap, star_color='r', annotate_offset=0.05):
        # 直接在传入的 ax 上绘图
        surf = ax.plot_surface(
            X, Y, Z, cmap=cmap,
            rstride=1, cstride=1,
            edgecolor='none', antialiased=True, alpha=0.85
        )
        # 底部等高线
        zmin, zptp = Z.min(), Z.ptp()
        ax.contourf(X, Y, Z, zdir='z', offset=zmin - 0.1 * zptp,
                    cmap=cmap, alpha=0.3, levels=15)
        # 侧面等高线
        yoff = row_sum_range.max() + 0.05 * row_sum_range.ptp()
        ax.contour(X, Y, Z, zdir='y', offset=yoff,
                   colors='k', alpha=0.4, levels=8)

        # 轴标签与视角
        ax.set_xlabel('Global Weight', labelpad=10)
        ax.set_ylabel('Row-sum of Weights', labelpad=10)
        ax.set_zlabel(zlabel, labelpad=10)
        ax.view_init(elev=28, azim=-124)

        # --- 修正 Colorbar 的创建方式 ---
        # 1. 从传入的 ax 获取它所属的 fig 对象
        fig = ax.get_figure()
        cax = fig.add_axes([0.78, 0.25, 0.02, 0.5])  # 您可以微调这些数值

        # 2. 在这个 fig 上添加 colorbar
        cbar = fig.colorbar(surf, cax=cax)
        cbar.set_label('Performance', rotation=270, labelpad=15)

        # —— 高亮最优点 —— #
        mi, mj = np.unravel_index(Z.argmax(), Z.shape)
        f1_max = Z[mi, mj]
        gw_opt = global_weight_range[mj]
        rs_opt = row_sum_range[mi]
        ax.plot([gw_opt], [rs_opt], [f1_max],
                marker='*', color=star_color, markeredgecolor='k',
                markersize=11)

        # 文字注释（稍微抬高一点，避免遮挡）
        ax.text(gw_opt,
                rs_opt,
                f1_max + annotate_offset * zptp,
                f'F1={f1_max:.3f}\nΣw={rs_opt:.2f}\nv={gw_opt:.2f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
        # 字体调小一点避免重叠

        return surf
    # ========== Fig1：Macro-F1 ==========
    fig1, ax1 = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(12, 8), dpi=120)
    plot_surface(ax1, perf[0], "Macro-F1", cmap='viridis', star_color='r')
    ax1.set_title("Parameter Landscape (Macro-F1)", pad=25, fontsize=16, fontweight='bold', y=1.00)
    fig1.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.88)
    plt.show()

    # ========== Fig2：Micro-F1 ==========
    fig2, ax2 = plt.subplots(subplot_kw={'projection': '3d'}, figsize=(12, 8), dpi=120)
    plot_surface(ax2, perf[1], "Micro-F1", cmap='viridis', star_color='r')
    ax2.set_title("Parameter Landscape (Micro-F1)", pad=25, fontsize=16, fontweight='bold', y=1.00)
    fig2.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.88)
    plt.show()

    # ───────── 叠合图：Macro-F1（蓝）+ Micro-F1（红）─────────
    fig3, ax3 = plt.subplots(subplot_kw={'projection': '3d'},
                             figsize=(12, 8), dpi=120)

    # 绘制两个曲面
    surf_macro = ax3.plot_surface(X, Y, perf[0], cmap='Blues', alpha=0.80,
                                  rstride=1, cstride=1, edgecolor='none')
    surf_micro = ax3.plot_surface(X, Y, perf[1], cmap='Reds', alpha=0.75,
                                  rstride=1, cstride=1, edgecolor='none')

    # --- 高亮两个曲面的最优点 ---
    # 1. 找到并绘制 Macro-F1 的最优点（黄色五角星）
    mi_macro, mj_macro = np.unravel_index(perf[0].argmax(), perf[0].shape)
    f1_max_macro = perf[0][mi_macro, mj_macro]
    gw_opt_macro = global_weight_range[mj_macro]
    rs_opt_macro = row_sum_range[mi_macro]
    ax3.plot([gw_opt_macro], [rs_opt_macro], [f1_max_macro],
             marker='*', color='yellow', markeredgecolor='k', markersize=14,
             label=f'Macro-F1 Opt (Σw={rs_opt_macro:.2f}, v={gw_opt_macro:.2f}, F1={f1_max_macro:.3f})')  # <--- 修改点

    # 2. 找到并绘制 Micro-F1 的最优点（黄色五角星）
    mi_micro, mj_micro = np.unravel_index(perf[1].argmax(), perf[1].shape)
    f1_max_micro = perf[1][mi_micro, mj_micro]
    gw_opt_micro = global_weight_range[mj_micro]
    rs_opt_micro = row_sum_range[mi_micro]
    ax3.plot([gw_opt_micro], [rs_opt_micro], [f1_max_micro],
             marker='*', color='yellow', markeredgecolor='k', markersize=14,
             label=f'Micro-F1 Opt (Σw={rs_opt_micro:.2f}, v={gw_opt_micro:.2f}, F1={f1_max_micro:.3f})')  # <--- 修改点

    # ─── 重新布置并列色标 ───
    cax_macro = fig3.add_axes([0.75, 0.18, 0.020, 0.58])
    cax_micro = fig3.add_axes([0.85, 0.18, 0.020, 0.58])

    cb1 = fig3.colorbar(surf_macro, cax=cax_macro)
    cb1.set_label('Macro-F1', rotation=270, labelpad=12)

    cb2 = fig3.colorbar(surf_micro, cax=cax_micro)
    cb2.set_label('Micro-F1', rotation=270, labelpad=12)

    # 其余格式化
    ax3.set_xlabel('Global Weight', labelpad=10)
    ax3.set_ylabel('Row-sum of Weights', labelpad=10)
    ax3.set_zlabel('F1 Score', labelpad=10)
    ax3.set_title('Parameter Landscape (Macro-F1 & Micro-F1)',
                  pad=25, fontsize=16, fontweight='bold', y=1.00)
    ax3.view_init(elev=28, azim=-124)

    # 添加图例以显示五角星代表的含义
    ax3.legend(loc='upper left', bbox_to_anchor=(0.56, 0.96) ,framealpha=0.95)

    # 左右边距稍缩小，给色标留出空间
    fig3.subplots_adjust(left=0.08, right=0.85, bottom=0.08, top=0.92)
    plt.show()

from typing import Tuple, Optional, Dict
from sklearn.metrics import accuracy_score, f1_score

def forecast_weather_markov(
        final_transition_matrices: np.ndarray,
        last_hidden_state: Tuple[int, int],
        n_hours: int,
        future_hidden_states: Optional[np.ndarray] = None,
        *,
        global_weight: float = 1.0,
        observed_weights: Optional[np.ndarray] = None,
        final_mus: Optional[np.ndarray] = None,
        final_corr: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    """
    根据论文方法使用转移矩阵预测未来 n 小时的天气，并可与真实标签比较。

    Parameters
    ----------
    final_transition_matrices : ndarray, shape (2, 2, 2)
        两条隐藏链 (雾霾链 z1、沙尘链 z2) 的转移矩阵。
        final_transition_matrices[c][i][j] = P(z_{t+1}=j | z_t=i) for chain c.
    last_hidden_state : (int, int)
        时刻 T 的最优隐藏状态 (z1_T, z2_T)。
    n_hours : int
        需要向前预测的小时数 (h)。
    future_hidden_states : ndarray, optional, shape (n_hours,)
        未来 h 小时的真实天气编码 (0/1/2)。若提供将返回评估指标。
    global_weight, observed_weights, final_mus, final_corr :
        保留接口以满足变量传入要求；本函数预测阶段仅使用转移矩阵。

    Returns
    -------
    pred_codes : ndarray, shape (n_hours,)
        预测的天气编码序列 (0 晴朗, 1 雾霾, 2 沙尘)。
    metrics : dict | None
        若给定 future_hidden_states，返回 {'accuracy', 'f1_macro'}；否则为 None。
    """
    # ---------- 1) 初始化 ----------
    z_curr = list(last_hidden_state)            # 当前 (z1, z2)
    pred_codes = np.empty(n_hours, dtype=int)

    # ---------- 2) 逐步递推 ----------
    for t in range(n_hours):
        for chain in (0, 1):                    # 0→雾霾链, 1→沙尘链
            z_curr[chain] = int(np.argmax(
                final_transition_matrices[chain, z_curr[chain], :]
            ))

        # ---------- 3) 状态映射 ----------
        pred_codes[t] = 2 if z_curr[1] else (1 if z_curr[0] else 0)

    # ---------- 4) 评价 (可选) ----------
    metrics = None
    if future_hidden_states is not None:
        acc = accuracy_score(future_hidden_states, pred_codes)
        f1  = f1_score(future_hidden_states, pred_codes,
                       average='macro', zero_division=0)
        metrics = {'accuracy': acc, 'f1_macro': f1}

    return pred_codes, metrics

# 导入所有需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score, classification_report, confusion_matrix, adjusted_rand_score


# 修正后的评估与绘图函数
def evaluate_and_plot(y_true, y_pred, y_prob, model_name: str, all_labels: np.ndarray):
    """
    对模型预测结果进行全面评估，并绘制混淆矩阵和ROC曲线图。
    （此版本修正了宏平均F1分数的计算错误）
    """
    print(f"\n\n{'=' * 25} 评估报告: {model_name} {'=' * 25}")

    # --- 关键修正：在计算 f1_score 时也传入 labels 参数 ---
    # 这确保了宏平均F1的计算会考虑所有类别，即使某些类别在测试集中不存在。
    f1_micro = f1_score(y_true, y_pred, labels=all_labels, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)

    # 打印正确的宏平均F1
    print(f"Micro F1 Score (微平均F1): {f1_micro:.4f}")
    print(f"Macro F1 Score (宏平均F1): {f1_macro:.4f}")  # 这个值现在会和下面的报告一致

    # --- 分类报告 (这部分原本就是正确的) ---
    target_names = [f'状态 {i}' for i in all_labels]
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        labels=all_labels,
        zero_division=0
    )
    print("\n详细分类报告 (Classification Report):\n")
    print(report)

    # --- 混淆矩阵绘图 (代码不变) ---
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.show()

    # --- ROC 曲线绘图 (代码不变) ---
    y_true_bin = label_binarize(y_true, classes=all_labels)
    n_classes = len(all_labels)

    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i, label in enumerate(all_labels):
        # 注意：这里需要确保y_prob的列数和all_labels的长度一致
        # 如果模型因为训练集中没有某个类而输出的概率列数较少，需要处理
        if i < y_prob.shape[1]:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:  # 如果测试集和训练集都没有某个类，y_prob可能没有那一列
            fpr[i], tpr[i], roc_auc[i] = np.array([0]), np.array([0]), 0.0

    # ...后续的ROC绘图代码不变...
    # (为简洁省略，您代码中这部分是正确的)
    # 计算微平均ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 计算宏平均ROC
    all_fpr_vals = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr_vals = np.zeros_like(all_fpr_vals)
    for i in range(n_classes):
        mean_tpr_vals += np.interp(all_fpr_vals, fpr[i], tpr[i])
    mean_tpr_vals /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr_vals, mean_tpr_vals
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # 绘图
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.3f})',
             color='navy', linestyle=':', linewidth=4)
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{target_names[i]} (AUC = {roc_auc[i]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'Multi-class ROC Curve - {model_name}', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True);
    plt.show()
    print(f"{'=' * 60}\n")

# (这里假设您脚本上方的 evaluate_and_plot 函数定义依然存在)
# 如果没有，请将上一轮回答中的该函数定义复制到这里。
# 为完整起见，此处重写一个精简的评估函数，只返回核心指标。
def calculate_metrics(y_true, y_pred, all_labels):
    macro_f1 = f1_score(y_true, y_pred, labels=all_labels, average='macro', zero_division=0)
    return macro_f1


def generate_comprehensive_report(y_true_all, y_pred_all, all_labels, model_name):
    """
    根据所有交叉验证折的累积结果，生成一份详尽的性能报告。
    """
    print(f"\n\n{'=' * 25} 详细评估报告: {model_name} {'=' * 25}")

    # --- 1. 累加混淆矩阵 ---
    cm = confusion_matrix(y_true_all, y_pred_all, labels=all_labels)
    print("累加混淆矩阵 (Aggregated Confusion Matrix):")
    # 为了与您的输出格式一致，我们直接打印数组
    print(cm)
    target_names = [f'状态 {i}' for i in all_labels]
    print(f"标签: {[label_map[l] for l in all_labels]}")  # 使用中文标签

    # --- 2. 调整后兰德指数 (ARI) ---
    ari = adjusted_rand_score(y_true_all, y_pred_all)
    print(f"\nARI = {ari:.4f}")

    # --- 3. 各类别性能指标 (TP, FP, FN, TN) ---
    print("\n--- 各类别性能指标 (TP, FP, FN, TN) ---")
    for i, label in enumerate(all_labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        print(f"类别 {label} ({label_map[label]}): TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # --- 4. F1 分数 (各类别, 宏平均, 微平均) ---
    print("\n--- F1 分数 ---")
    # 直接使用 classification_report 来获取各类别F1和宏/微平均F1
    report = classification_report(y_true_all, y_pred_all, labels=all_labels,
                                   target_names=[label_map[l] for l in all_labels],
                                   zero_division=0, digits=4)
    print(report)

    # 为了单独打印宏/微平均，可以再次计算
    macro_f1 = f1_score(y_true_all, y_pred_all, labels=all_labels, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true_all, y_pred_all, labels=all_labels, average='micro', zero_division=0)
    print(f"宏F1 (Macro-F1) 分数: {macro_f1:.4f}")
    print(f"微F1 (Micro-F1) 分数: {micro_f1:.4f}")

    # --- 5. 可视化混淆矩阵 ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens' if 'Forest' in model_name else 'Purples',
                xticklabels=[label_map[l] for l in all_labels],
                yticklabels=[label_map[l] for l in all_labels])
    plt.title(f'Aggregated Confusion Matrix - {model_name}', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.show()

# === 1. 构造训练集（取历史数据末尾，保证含全部类别） ===
def build_train_set_by_ratio(histr_obs: np.ndarray,
                             histr_hidden: np.ndarray,
                             ratio: float = 0.20):
    """
    histr_obs/histr_hidden 的时间顺序：新 → 旧
    直接取最前面的 ratio 部分（即最新的历史样本）作为训练集。
    """
    total_len = histr_hidden.shape[0]
    n_train   = int(total_len * ratio)
    X_train   = histr_obs[:, :n_train].T      # (n_train, E)
    y_train   = histr_hidden[:n_train]        # (n_train,)
    print(f"训练集选取历史数据前 {n_train}/{total_len} 条（{ratio:.0%}）")
    return X_train, y_train


# # --------------------主函数---------------------------------------------------------------------------------
# # -----------------------------------------------------------------------------------------------------
# # --------------------数据读取---------------------------------------------------------------------------------

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
# print("\nmonths:", months)
print('\nobserved_states =\n', observed_states)
print('\nshape of observed_states = ', observed_states.shape)
print("\nhidden_states = \n", hidden_states)
print("\nshape of hidden_states", hidden_states.shape)

# --------------------参数初始化---------------------------------------------------------------------------------

# 设定参数
n_steps = len(observed_states[0])  # 时间步
E = observed_states.shape[0]  # 观测链个数
M = 2  # 隐藏链个数
K = 2  # 隐藏链范围
precision = 1  # 迭代终止条件

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
    'corr_mode': 1 # 0:无corr, 1:全局高斯coupla, 2:高斯coupla, 3:全局联合正态, 4:联合正态
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

# --------------------EM---------------------------------------------------------------------------------

# 创建实体
F = FullDiscreteFactorialHMM(params=params, n_steps=n_steps, calculate_on_init=True)

# # 运行EM算法
# # 对数似然度 Log Likelihood = log(P(观测数据 | 模型参数))
# final_transition_matrices, final_mus, final_sigmas, new_params, gammas = F.EM(
#     observed_states, likelihood_precision = precision, verbose=True, print_every=1)
# np.save('M1_gammas.npy', gammas)

# 跳过EM算法
final_transition_matrices, final_mus, final_sigmas, final_pi, new_params = jumpEM_710C(params)

print('\nfinal_transition_matrices = \n', final_transition_matrices)
print('\nfinal mus = \n', new_params['mus'])
print('\nfinal sigmas = \n', new_params['sigmas'])
print('\nfinal corr\n', new_params['corr_mats'])


# --------------------权重---------------------------------------------------------------------------------

# # 互信息法计算观测链权重 (将params里（4）的权重（1，1，1，1）变成了new_params里的（4，4）维）
# sigle_weights, new_params['observed_weights'] = Get_observed_weights(histr_observed_states, histr_hidden_states, params)
# print("weights: Clear, Haze, Smog, Smog and Haze =\n", new_params['observed_weights'])
#
# # 标准库
# new_params['observed_weights'] = np.array(
#     [[[0.0831, 0.0182, 0.2794, 0.0689],
#       [0.0125, 0.003,  0.0077, 0.0046]],
#
#      [[0.0765, 0.0146, 0.2708, 0.0723],
#       [1.    , 1.    , 1.    , 1.    ]]]
# )
#
# # 自己的函数
# new_params['observed_weights'] = np.array(
#     [[[0.5244, 0.4416, 0.7771, 0.5137],
#       [0.5122, 0.5005, 0.5052, 0.5028]],
#
#      [[0.5232, 0.445 , 0.7662, 0.5137],
#       [1.    , 1.    , 1.    , 1.    ]]]
# )
#
# # # 权重取倒数
# # new_params['observed_weights'] = 1.0 / new_params['observed_weights']
#
# # 权重归一化
# mi = np.zeros((4))
# for i in range(2):
#     for j in range(2):
#         mi = new_params['observed_weights'][i, j]
#         mi = mi / mi.sum()       # 归一化到和为1
#         # mi = mi * E              # 放大至和为E
#
#         # mi = mi * 1.21
#         # mi = mi * 1.79              # 放大至和为E
#         # mi = mi * 1.47              # 放大至和为E
#         # mi = mi * 1.77              # 放大至和为E
#         new_params['observed_weights'][i, j] = mi

# 设置全局权重
# transition_emission_ratio
global_weight = 1
# global_weight = 4.23
# global_weight =
# global_weight =
# global_weight =

# --------------------Viterbi---------------------------------------------------------------------------------

# 创建实体
H = FullDiscreteFactorialHMM(params=new_params, n_steps=n_steps, calculate_on_init=True)
# 运行Viterbi算法（使用均值权重）
most_likely_hidden_state, back_pointers, lls = H.Viterbi(observed_states, new_params['observed_weights'], global_weight, hidden_states)

# --------------------ROC---------------------------------------------------------------------------------

# 计算ROC

gammas = np.load('M1_gammas.npy')
print('\nfinal gammas\n', gammas)

# # 1. 三个状态各自的子图，如下：
# compute_and_plot_three_ROCs(gammas, hidden_states)

# 2. 多类别的 ROC 图，如下：
compute_and_plot_multiclass_ROC(gammas, hidden_states)

# --------------------画图---------------------------------------------------------------------------------

# 隐藏状态区分结果检验(F1 AUC画图替代)
_ = result_verification(hidden_states, most_likely_hidden_state)
# 隐藏状态区分结果画图(条带)
hidden_state_differentiation_chart(hidden_states, most_likely_hidden_state)

# # 事件区分图（月份条形图）
# hidden_state_monthly_accuracy_chart(hidden_states, most_likely_hidden_state, months, years)

# F1 AUC画图
f1_scores = []
f1_scores = F1_score__Confusion_Matrix__ARI(hidden_states, most_likely_hidden_state)

# # 三维图像，联合调整全局权重 v 与观测值权重 w，F1 Score
# # 设置取样密度
# numb1 = 5
# plot_3d_macro_micro_F1(hidden_states, observed_states, new_params['observed_weights'], numb1)

# --------------------预测---------------------------------------------------------------------------------

# # 预测
# # 读取数据与历史数据
# _, future_hidden_states, _, _, _ = Get_observed_and_hidden_state("D:\Desktop\预测.xls")
# fit_future_hidden_states, report = forecast_weather_markov(
#     final_transition_matrices,
#     last_hidden_state=(0, 0),
#     n_hours=24,
#     future_hidden_states=future_hidden_states,
#     global_weight=global_weight,
#     observed_weights=new_params['observed_weights'],
#     final_mus=final_mus,
# )
#
# print("预测标签:", fit_future_hidden_states)                       # 来自函数返回值
# print("实际标签:", future_hidden_states[:len(fit_future_hidden_states)])  # 确保长度一致
# print("评估结果:", report)

# --------------------随机森林/支持向量机 结果对比---------------------------------------------------------------------------------

# # === 2. 数据准备 ===
# X_train, y_train = build_train_set_by_ratio(histr_observed_states,
#                                             histr_hidden_states,
#                                             ratio=0.30)
#
# X_test  = observed_states.T     # 新数据：时间已是 旧 → 新
# y_test  = hidden_states
# all_possible_labels = np.unique(y_test)
# label_map = {0: '晴朗', 1: '雾霾', 2: '沙尘'}
#
# # === 3. 模型训练与时序外评估 ===
# print("\n########## 开始模型训练与单次时序外（out-of-time）评估 ##########")
#
# # -- 随机森林 --
# rf_model = RandomForestClassifier(
#     n_estimators=100,
#     random_state=42,
#     class_weight='balanced',
#     n_jobs=-1
# )
# rf_model.fit(X_train, y_train)
# y_pred_rf  = rf_model.predict(X_test)
# y_prob_rf  = rf_model.predict_proba(X_test)
# evaluate_and_plot(y_test, y_pred_rf, y_prob_rf, "Random Forest", all_possible_labels)
#
# # -- 支持向量机 --
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)
#
# svm_model = SVC(
#     kernel='rbf',
#     class_weight='balanced',
#     probability=True,
#     random_state=42
# )
# svm_model.fit(X_train_scaled, y_train)
# y_pred_svm  = svm_model.predict(X_test_scaled)
# y_prob_svm  = svm_model.predict_proba(X_test_scaled)
# evaluate_and_plot(y_test, y_pred_svm, y_prob_svm, "Support Vector Machine (SVM)", all_possible_labels)
#
# # === 4. 汇总报告 ===
# generate_comprehensive_report(y_test, y_pred_rf,  all_possible_labels, "Random Forest")
# generate_comprehensive_report(y_test, y_pred_svm, all_possible_labels, "Support Vector Machine (SVM)")

# 显示运行时间
end_time = time.time()
print('\n', f"代码执行时间: {end_time - start_time:.6f} 秒")
