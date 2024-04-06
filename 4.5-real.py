# 去掉断言，去掉方差最低值
# 处理后数据
import copy
import csv
import functools
import itertools
import operator
import warnings
import numpy as np
# numpy=1.21.6，1.26.1不行
# pandas=2.0.0，2,1,4不行
import scipy
from scipy.stats import norm
from scipy.stats import pearsonr
import xlrd
import xlwt
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
import pandas as pd

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

    def GetTransitionMatrixColumn(self, n_step, prev_state):
        MS = self.transition_matrices_tensor[n_step]
        vecs = [M[:, prev_state[i]] for i, M in enumerate(MS)]
        return functools.reduce(np.kron, vecs)

    def GetTransitionMatrix(self, n_step):
        return functools.reduce(np.kron, self.transition_matrices_tensor[n_step])

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
        initial_hidden_state *= self.GetObservedGivenHidden(initial_observed_state, 0)

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
        alphas[..., 0] = self.InitialHiddenStateConditional(observed_states[..., 0])
        scaling_constants[0] = alphas[..., 0].sum()
        alphas[..., 0] /= scaling_constants[0]

        for n_step in range(1, n_steps):
            alphas[..., n_step] = self.GetObservedGivenHidden(tuple(observed_states[..., n_step]), n_step)
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
            vec = self.GetObservedGivenHidden(observed_states[..., n_step + 1], n_step + 1)
            betas[..., n_step] = self.MultiplyTransitionMatrixVector(n_step, vec * betas[..., n_step + 1],
                                                                     transpose=True)
            betas[..., n_step] /= scaling_constants[n_step + 1]

        # Join
        gammas = alphas * betas
        # print("shape = \n",alphas.shape,"\n",betas.shape,"\n",gammas.shape)

        return alphas, betas, gammas, scaling_constants, log_likelihood

    # Notice Errata of page 628 of Bishop 13.2.4!!
    def CalculateXis(self, observed_states, alphas, betas, scaling_constants):
        # Slow, use CalculateAndCollapseXis if possible!!!

        if len(observed_states.shape) == 1:
            observed_states = observed_states.reshape((1, -1))

        n_steps = observed_states.shape[-1]

        xis = np.ones(self.hidden_indices.field_sizes + self.hidden_indices.field_sizes + [n_steps - 1])
        for n_step in range(1, n_steps):
            xis[..., n_step - 1] = (alphas[..., n_step - 1].ravel()[:, np.newaxis]
                                    * self.GetObservedGivenHidden(observed_states[..., n_step], n_step).ravel()[
                                      np.newaxis, :]
                                    * self.GetTransitionMatrix(n_step - 1).T
                                    * betas[..., n_step].ravel()[np.newaxis, :]
                                    / scaling_constants[n_step]).reshape(
                self.hidden_indices.field_sizes + self.hidden_indices.field_sizes)

        return xis

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
            b = self.GetObservedGivenHidden(observed_states[..., n_step], n_step) * betas[..., n_step]

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

    def ViterbiSlower(self, observed_states):
        if len(observed_states.shape) == 1:
            observed_states = observed_states.reshape((1, -1))

        n_steps = observed_states.shape[-1]

        back_pointers = np.ones((self.n_hidden_state_space, n_steps - 1), dtype=int)
        lls = np.ones((self.n_hidden_state_space, n_steps), dtype=float)

        ll = np.log(self.InitialHiddenStateConditional(observed_states[:, 0])).reshape((self.n_hidden_state_space))
        lls[:, 0] = ll

        for n_step in range(1, n_steps):
            M = np.log(self.GetTransitionMatrix(n_step - 1)) + ll[np.newaxis, :]
            back_pointers[:, n_step - 1] = np.argmax(M, axis=1)

            ll = np.log(self.GetObservedGivenHidden(observed_states[..., n_step], n_step)).reshape(
                (self.n_hidden_state_space))
            ll += M[np.arange(self.n_hidden_state_space), back_pointers[:, n_step - 1]]
            lls[:, n_step] = ll

        # Backtrack
        most_likely = np.zeros(n_steps, dtype=int)
        most_likely[n_steps - 1] = np.argmax(ll)
        last = most_likely[n_steps - 1]

        for n_step in range(n_steps - 1, 0, -1):
            most_likely[n_step - 1] = back_pointers[last, n_step - 1]
            last = most_likely[n_step - 1]

        return most_likely, back_pointers, lls

    def Viterbi(self, observed_states, observed_weights):
        if len(observed_states.shape) == 1:
            observed_states = observed_states.reshape((1, -1))

        n_steps = observed_states.shape[-1]

        # 确保权重列表与隐藏链的数量相匹配
        assert len(observed_weights) == self.n_observed_states, "权重列表的长度必须与隐藏链的数量相匹配"

        mgrid_prepared = list(np.mgrid[[range(s) for s in self.hidden_indices.field_sizes]])

        back_pointers = np.ones(self.hidden_indices.field_sizes + [self.n_hidden_states] + [n_steps - 1], dtype=int)
        lls = np.zeros(self.hidden_indices.field_sizes + [n_steps], dtype=float)

        lls[..., 0] = np.log(self.InitialHiddenStateConditional(observed_states[..., 0]))
        ll = lls[..., 0]

        for n_step in range(1, n_steps):
            vector = ll.copy()
            argmaxes = np.zeros(self.hidden_indices.field_sizes + [self.n_hidden_states], dtype=int)

            for i in range(self.n_hidden_states):
                submatrix = np.nan_to_num(np.log(self.transition_matrices_tensor[n_step - 1][i][:, :]))
                idx_a = [slice(None)] + [np.newaxis] * self.n_hidden_states
                idx_a[i + 1] = slice(None)

                B = submatrix[idx_a] + vector[np.newaxis, ...]

                vector = np.moveaxis(B.max(axis=i + 1), 0, i)
                mx = np.moveaxis(B.argmax(axis=i + 1), 0, i)

                argmaxes[..., i] = mx
            back_pointers[..., self.n_hidden_states - 1, n_step - 1] = argmaxes[..., self.n_hidden_states - 1]
            for i in range(self.n_hidden_states - 2, -1, -1):
                indices = mgrid_prepared.copy()
                indices[i + 1] = argmaxes[..., i + 1]
                back_pointers[..., i, n_step - 1] = argmaxes[..., i][indices]

            # 应用权重到观测概率
            observed_probs = self.GetObservedGivenHidden(observed_states[..., n_step], n_step)
            weighted_observed_probs = np.log(observed_probs) * observed_weights[
                n_step % len(observed_weights)]  # 这里用权重调整观测概率的对数
            ll = weighted_observed_probs + vector
            lls[..., n_step] = ll
            # # 检查能见度是否为10
            # if observed_states[ 2, n_step] == 10:
            #     # 如果能见度为10,则强制将最可能的隐藏状态设为(0,0)
            #     ll = np.full(self.hidden_indices.field_sizes, -np.inf)
            #     ll[0, 0] = 0
            # else:
            #     # 否则,正常计算ll
            #     pass

            lls[..., n_step] = ll

        # Backtrack
        most_likely = np.zeros((self.n_hidden_states, n_steps), dtype=int)
        most_likely[..., n_steps - 1] = np.unravel_index(np.argmax(ll.ravel()), self.hidden_indices.field_sizes)
        last = most_likely[..., n_steps - 1]

        for n_step in range(n_steps - 1, 0, -1):
            most_likely[..., n_step - 1] = back_pointers[last.tolist() + [slice(None), n_step - 1]]
            last = most_likely[..., n_step - 1]

        return most_likely, back_pointers, lls

    def CalculateJointLogLikelihood(self, observed_states, hidden_states):
        n_steps = observed_states.shape[-1]

        logp = 0.0
        # p(x0,z0)
        logp += np.log(self.InitialHiddenStateConditional(observed_states[..., 0])[tuple(hidden_states[..., 0])])

        for n_step in range(1, n_steps):
            # p(z_n | z_{n-1})
            col = self.GetTransitionMatrixColumn(n_step - 1, hidden_states[..., n_step - 1])
            logp += np.log(col[np.ravel_multi_index(hidden_states[..., n_step], self.hidden_indices.field_sizes)])

            # p(x_n | z_n)
            logp += np.log(
                self.GetObservedGivenHidden(observed_states[..., n_step], n_step)[tuple(hidden_states[..., n_step])])

        return logp

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
    def __init__(self, params, n_steps, calculate_on_init=False,):

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
        self.mus = np.zeros((params['hidden_alphabet_size'],params['hidden_alphabet_size'], params['n_observed_chains']))
        self.sigmas = np.zeros((params['hidden_alphabet_size'],params['hidden_alphabet_size'], params['n_observed_chains']))

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
        self.mus = params['mus']
        self.sigmas = params['sigmas']

    def GetObservedGivenHidden(self, observed_state, n_step):

        # Initialize array to store probabilities
        probs = np.ones(self.hidden_indices.field_sizes)

        # Loop through all possible hidden states
        for hidden_state in self.all_hidden_states:

            # Get the hidden state indices
            hid_idx = tuple(hidden_state)

            # Calculate probability for this hidden state
            prob = 1.0
            # print("mus_test.shape = ", self.mus.shape)
            for e, obs in enumerate(observed_state):
                mean = self.mus[hid_idx][e]
                std_dev = self.sigmas[hid_idx][e]
                prob *= scipy.stats.norm(loc=mean, scale=std_dev).pdf(obs)

            # Store in array
            probs[hid_idx] = prob

        return probs

    # 根据发射矩阵抽样
    def DrawObservedGivenHidden(self, hidden_states, n_step, random_state):
        observed_state_t = np.zeros((E, ))  # 创建一个形状为(E, 1)的数组，用于存储观测状态

        # 从输入的hidden_states中获取a和b
        # print('shape of hidden_states = ',hidden_states.shape)
        a = hidden_states[0, ]
        b = hidden_states[1, ]

        # 分别抽样E个观测值，每个观测值均值和标准差分别是a * (i+1) 和 b * (i+1)，其中i从0到(E-1)
        for i in range(E):
            mean_i = ((a + 0.5 * b + 1) * (i + 1) * (1 + 1 / (n_step + 1)))/(4 * E)
            std_dev_i = ((b + 0.5 * a + 1) * (i + 1) * (1 + 1 / (n_step + 1)))/(4 * E)

            # 使用正态分布随机抽样
            observed_state_t[i, ] = random_state.normal(loc=mean_i, scale=std_dev_i)

        return observed_state_t

    def MStep(self, observed_states, alphas, betas, gammas, scaling_constants):

        K = self.params['hidden_alphabet_size'] #隐藏值范围
        E = self.params['n_observed_chains'] #观测链个数

        # 初始化initial_hidden_state_estimate和transition_matrices_estimates
        initial_hidden_state_estimate = np.zeros((self.n_hidden_states, K))
        transition_matrices_estimates = np.zeros((self.n_hidden_states, K, K))

        # 更新转移矩阵
        for field in range(self.n_hidden_states):
            collapsed_xis = self.CalculateAndCollapseXis(field, observed_states, alphas, betas, scaling_constants)
            collapsed_gammas = self.CollapseGammas(gammas, field)

            initial_hidden_state_estimate[field, :] = collapsed_gammas[:, 0] / collapsed_gammas[:, 0].sum()
            transition_matrices_estimates[field, :, :] = (
                        collapsed_xis.sum(axis=2) / collapsed_gammas[:, :-1].sum(axis=1)[:, np.newaxis]).T
        # 更新发射矩阵
        # 重置mus和sigmas
        self.mus = np.zeros((K, K, E))
        self.sigmas = np.zeros_like(self.mus)

        # 逐个观测链进行更新
        for e in range(E):
            observed = observed_states[e, :]  # 第e个观测链的数据

            for a in range(K):
                for b in range(K):
                    # 使用gammas来过滤对应于隐藏状态(a, b)的时间点
                    gamma_filtered = gammas[a, b, :]

                    # 如果gamma_filtered全为0，则跳过
                    if not np.any(gamma_filtered):
                        continue

                    # 计算加权均值和方差
                    mean_ab = np.sum(gamma_filtered * observed) / np.sum(gamma_filtered)
                    variance_ab = np.sum(gamma_filtered * (observed - mean_ab) ** 2) / np.sum(gamma_filtered)

                    # 更新mus和sigmas
                    self.mus[a, b, e] = mean_ab
                    self.sigmas[a, b, e] = np.sqrt(variance_ab)
                    # if(self.sigmas[a, b, e] <= 0.01):
                    #     self.sigmas[a, b, e] = 0.01

        # 确保更新的参数被正确返回并在EM方法中使用
        new_params = {
            'hidden_alphabet_size': self.params['hidden_alphabet_size'],
            'n_hidden_chains': self.params['n_hidden_chains'],
            'n_observed_chains': self.params['n_observed_chains'],
            'initial_hidden_state': initial_hidden_state_estimate,
            'transition_matrices': transition_matrices_estimates,
            'mus': self.mus,
            'sigmas': self.sigmas,
        }
        return new_params

    def EM(self, observed_states, likelihood_precision, n_iterations = 100, verbose=False, print_every=1,
           random_seed=None):
        old_log_likelihood = -np.inf
        n_iter = 0

        H = FullDiscreteFactorialHMM(params, self.n_steps, True)

        while True:
            alphas, betas, gammas, scaling_constants, log_likelihood = H.EStep(observed_states)
            # 使用更新后的参数重新初始化模型
            new_params = H.MStep(observed_states, alphas, betas, gammas, scaling_constants)
            H = FullDiscreteFactorialHMM(new_params, self.n_steps, True)
            if verbose and (n_iter % print_every == 0 or n_iter == n_iterations - 1):
                print("Iter: {}\t LL: {}".format(n_iter, log_likelihood))

            n_iter += 1
            if n_iter == n_iterations:
                print("\nFinal Log likelihood:", log_likelihood)
                return new_params['transition_matrices'], new_params['mus'], new_params['sigmas']

            if np.abs(log_likelihood - old_log_likelihood) < likelihood_precision:
                print("\nFinal Log likelihood:", log_likelihood)
                return new_params['transition_matrices'], new_params['mus'], new_params['sigmas']

            old_log_likelihood = log_likelihood

        return H
def Get_observed_and_hidden_state(location):
    # 加载Excel为观测数据:
    #   PM10浓度
    #   观测前十分钟地面高度10-12米处的风速
    #   水平能见度（km，大于10记为10）
    #   地面高度2米处的相对湿度（%）
    df = pd.read_excel(location, skiprows=0)
    # 初始化列表来存储观测状态和隐藏状态
    observed_states = []
    hidden_states = []
    # 遍历DataFrame中的每一行
    for index, row in df.iterrows():
        # 初始化当前行的观测状态列表
        new_row = []
        bob = 0
        # 处理观测状态数据（第3列到第6列）
        for x in row[2:6]:
            if pd.isna(x):
                bob = 1
            else:
                new_row.append(float(x))  # 使用float()将字符串转换为浮点数    observed_states.append(new_row)
        if bob == 0:  # 如果new_row不为空才添加
            observed_states.append(new_row)
        hidden_state = row[6]  # 第7列为隐藏状态
        hidden_states.append(hidden_state)  # 将隐藏状态添加到列表中
    return np.array(observed_states).T, np.array(hidden_states).T

def Get_observed_weights(observed_states, hidden_states):

    # 隐藏状态解码
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

    # 将observed_states转置，因为mutual_info_classif期望每行是一个样本，每列是一个特征
    observed_states_transposed = observed_states.T

    # 计算每个特征与隐藏状态之间的互信息
    mutual_info = mutual_info_classif(observed_states_transposed, hidden_states)

    # 标准化权重，使得总和为1
    weights = mutual_info / mutual_info.sum()

    print("\nobserved_weights = \n", weights, '\n')

    return weights

def initial_transition_matrix(params, location):

    # 创建一个全零的(M, K, K)数组
    transition_matrix_real = np.zeros((params['n_hidden_chains'], params['hidden_alphabet_size'], params['hidden_alphabet_size']))
    # 打开Excel文件
    filename = location
    workbook = xlrd.open_workbook(filename)
    worksheet = workbook.sheet_by_index(0)

    # 从第二行开始读取数据（忽略第一行文字行）
    for row_idx in range(1, worksheet.nrows - 1):
        current_value = int(worksheet.cell_value(row_idx, 6))  # 当前单元格的值（第七列）
        # print("worksheet.cell_value(row_idx + 1, 5) = ",worksheet.cell_value(row_idx + 1, 5))
        next_value = int(worksheet.cell_value(row_idx + 1, 6))  # 下一个单元格的值（第七列）

        # 更新转移矩阵
        if (current_value == 0):
            if (next_value != 1):
                transition_matrix_real[0, 0, 0] += 1
            else:
                transition_matrix_real[0, 0, 1] += 1
            if (next_value != 2):
                transition_matrix_real[1, 0, 0] += 1
            else:
                transition_matrix_real[1, 0, 1] += 1
        if (current_value == 1):
            if (next_value != 1):
                transition_matrix_real[0, 1, 0] += 1
            else:
                transition_matrix_real[0, 1, 1] += 1
            if (next_value != 2):
                transition_matrix_real[1, 0, 0] += 1
            else:
                transition_matrix_real[1, 0, 1] += 1
        if (current_value == 2):
            if (next_value != 1):
                transition_matrix_real[0, 0, 0] += 1
            else:
                transition_matrix_real[0, 0, 1] += 1
            if (next_value != 2):
                transition_matrix_real[1, 1, 0] += 1
            else:
                transition_matrix_real[1, 1, 1] += 1

    # 归一化处理
    for i in range(params['hidden_alphabet_size']):
        for j in range(params['hidden_alphabet_size']):
            sum_values = np.sum(transition_matrix_real[i, j])
            if sum_values != 0:
                transition_matrix_real[i, j] /= sum_values
    for i in range(params['n_hidden_chains']):
        transition_matrix_real[i,...] = transition_matrix_real[i,...].T

    return transition_matrix_real

def initial_ObservedGivenHidden_matrix(params, observed_states):
    # Prepare the matrix of P(X=x|Z=z)

    # 设置打印选项，避免科学计数法，并限制小数位数
    np.set_printoptions(precision=4, suppress=True)

    # 定义聚类中心数
    num_clusters = 4

    # 初始化mus和sigmas
    mus = np.zeros((params['hidden_alphabet_size'], params['hidden_alphabet_size'], params['n_observed_chains']))
    sigmas = np.zeros((params['hidden_alphabet_size'], params['hidden_alphabet_size'], params['n_observed_chains']))

    # 对每一条观测链进行均值和方差聚类
    for obs_chain in range(params['n_observed_chains']):
        # 均值聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(observed_states[:, obs_chain].reshape(-1, 1))
        labels = kmeans.labels_

        # 对每个簇，计算方差
        for cluster_id in range(4):
            cluster_data = observed_states[labels == cluster_id, obs_chain]
            # 计算方差
            cluster_variance = np.var(cluster_data, ddof=0)
            # 如果计算后的方差为0，则设置为0.0001
            if cluster_variance < 0.0001:
                cluster_variance = 0.0001
            # 设置8位小数的精度
            cluster_variance = round(cluster_variance, 5)

            # 假设每个簇对应一个隐藏状态的组合
            hidden_state_1 = cluster_id // 2
            hidden_state_2 = cluster_id % 2

            # 更新mus和sigmas
            mus[hidden_state_1, hidden_state_2, obs_chain] = kmeans.cluster_centers_[cluster_id, 0]
            sigmas[hidden_state_1, hidden_state_2, obs_chain] = np.sqrt(cluster_variance)  # 将方差转换为标准差
            # if(cluster_id == 1 or cluster_id == 3):
            #     sigmas[hidden_state_1, hidden_state_2, obs_chain] /= 10


    # # 打印初始化的mus和sigmas
    # print("\nsup_initial mus:\n", mus)
    # print("\nsup_initial sigmas:\n", sigmas)

    # 根据历史数据预定义的隐状态与聚类匹配顺序
    # (隐状态): [PM10排名, 风速排名, 能见度排名, 相对湿度排名]
    hidden_state_order = {
        (0, 0): [4, 3, 1, 2], # 无
        (1, 0): [3, 4, 3, 1], # 雾霾
        (0, 1): [2, 1, 2, 4], # 沙尘
        (1, 1): [1, 2, 4, 3], # 都有
    }

    # 创建新的mus和sigmas数组，用于存储更新后的值
    new_mus = np.zeros(mus.shape)
    new_sigmas = np.zeros(sigmas.shape)

    # 对每一条观测链进行操作
    for obs_chain in range(params['n_observed_chains']):
        # 获取当前观测链的所有簇的均值，并计算其排序索引（从大到小）
        cluster_means = mus[..., obs_chain].flatten()
        sorted_indices = np.argsort(-cluster_means)  # '-'号表示降序排序

        # 根据排序结果，更新mus和sigmas
        for new_rank, (hidden_state_1, hidden_state_2) in enumerate(itertools.product(range(K), repeat=M)):
            # 根据预定义的顺序找到当前隐状态对应的排名，并映射到排序后的索引
            rank = hidden_state_order[(hidden_state_1, hidden_state_2)][obs_chain] - 1  # 从0开始索引
            sorted_idx = sorted_indices[rank]

            # 计算原始索引位置
            original_idx = np.unravel_index(sorted_idx, (K, K))

            # 更新新的mus和sigmas
            new_mus[hidden_state_1, hidden_state_2, obs_chain] = mus[original_idx][obs_chain]
            new_sigmas[hidden_state_1, hidden_state_2, obs_chain] = sigmas[original_idx][obs_chain]

    # 更新mus和sigmas
    mus = new_mus
    sigmas = new_sigmas

    # 打印更新后的mus和sigmas
    # 横坐标为指标，纵坐标为簇
    print("\nshape of mus and sigmas : ", mus.shape)
    print("\ninitial mus:\n", mus)
    print("\ninitial sigmas:\n", sigmas)

    return mus, sigmas

def predict_future_hidden_states(transition_matrices, most_likely_states, n_future_steps, random_state=None):
    """
    根据概率随机选择未来的隐藏状态。

    Parameters:
    - transition_matrices: 转移概率矩阵，形状为(M, K, K)，M是隐藏链的数量，K是每条隐藏链的状态数。
    - most_likely_states: 维特比算法得到的最可能的隐藏状态序列，形状为(M, T)，T是时间步数。
    - n_future_steps: 要预测的未来时间步数。
    - random_state: 随机数种子，用于可重复性的随机选择。

    Returns:
    - future_states: 预测的未来隐藏状态序列，形状为(M, n_future_steps)。
    """
    if random_state is not None:
        np.random.seed(random_state)  # 设置随机数种子

    M, T = most_likely_states.shape
    future_states = np.zeros((M, n_future_steps), dtype=int)

    # 从最后一个已知的隐藏状态开始
    last_states = most_likely_states[:, -1]

    for future_step in range(n_future_steps):
        next_states = np.zeros(M, dtype=int)
        for chain_index in range(M):
            transition_matrix = transition_matrices[chain_index]
            last_state = last_states[chain_index]
            state_probabilities = transition_matrix[:, last_state]
            # 根据概率随机选择下一个状态
            next_state = np.random.choice(
                np.arange(len(state_probabilities)), p=state_probabilities)
            next_states[chain_index] = next_state

        future_states[:, future_step] = next_states
        last_states = next_states

    return future_states

# 读取数据
# 设置地址
location = 'D:/new/桌面/合并数据.xls'
observed_states, hidden_states = Get_observed_and_hidden_state(location)
print('\nobserved_states =\n', observed_states)
print('\nshape of observed_states = ', observed_states.shape)
print("\nhidden_states = \n", hidden_states)
print("\nshape of hidden_states", hidden_states.shape)

# 设定参数
n_steps = len(observed_states[0]) # 时间步
E = observed_states.shape[0] # 观测链个数
M = 2 # 隐藏链个数
K = 2 # 隐藏链范围
precision = 0.01 # 迭代终止条件

# 字典便于传递和更新参数
params = {
    'hidden_alphabet_size': K, # 隐藏值范围
    'n_hidden_chains': M, # 隐藏链个数
    'n_observed_chains': E, # 观测链个数
    'initial_hidden_state': np.zeros((M, K)), # 初始隐藏状态
    'transition_matrices': np.zeros((M, K, K)), # 转移矩阵
    'mus': np.zeros((K, ) * M + (E, )), # 均值
    'sigmas': np.zeros((K, ) * M + (E, )), # 标准差
    'wells': 0, # 打印第一次循环中的均值标准差，作为跳出循环的判定值
    'observed_weights' : [1, 1, 1, 1] # 观测链权重
}

# 参数初始化
# 设置初始隐藏值
params['initial_hidden_state'][0, :] = [548 / 8477] * K
params['initial_hidden_state'][1, :] = [76 / 8477] * K
# 设置初始转移矩阵
transition_matrix_real = initial_transition_matrix(params, location)
params['transition_matrices'] = transition_matrix_real
print('\nshape of transition_matrices:', params['transition_matrices'].shape)
print('\ninitial transition_matrices:\n', params['transition_matrices'])
# 设置初始mus和sigmas
params['mus'], params['sigmas'] = initial_ObservedGivenHidden_matrix(params, observed_states.T)
# 互信息法计算观测链权重
params['observed_weights'] = Get_observed_weights(observed_states, hidden_states)

# 创建实体
F = FullDiscreteFactorialHMM(params=params, n_steps=n_steps, calculate_on_init=True)

# 运行Forward-Backward算法
alphas, betas, gammas, scaling_constants, log_likelihood = F.EStep(observed_states)
# gammas: 隐状态被占据的概率(在给定观测序列下，特定时间步的特定隐藏状态的后验概率,alphas * betas)
# scaling_constants: 尺度因子
# log_likelihood: 对数似然度

# 运行EM算法
# 对数似然度 Log Likelihood = log(P(观测数据 | 模型参数))
final_transition_matrices, final_mus, final_sigmas = F.EM(
    observed_states, likelihood_precision = precision, verbose=True, print_every=1)
print('\nfinal_transition_matrices = \n', final_transition_matrices)
print('\nfinal mus = \n', final_mus)
print('\nfinal sigmas = \n', final_sigmas)

# 运行Viterbi算法
most_likely_hidden_state, back_pointers, lls = F.Viterbi(observed_states
                                                         , params['observed_weights'])
# most_likely: 最可能的隐状态序列
# back_pointers: 回溯指针
# lls: 各timestep的最大log likelihood
print("\nViterbi most_likely_hidden_state = \n", most_likely_hidden_state)

# 保存维特比结果
# 创建一个DataFrame，指定列名为“雾霾估计”和“沙尘估计”
hidden_states_df = pd.DataFrame(most_likely_hidden_state.T, columns=['雾霾估计', '沙尘估计'])
# 读取原始Excel文件
df = pd.read_excel(location)
# 将隐藏状态序列DataFrame添加到原始DataFrame中
df['雾霾估计'] = hidden_states_df['雾霾估计']
df['沙尘估计'] = hidden_states_df['沙尘估计']
# 保存更新后的DataFrame回Excel文件
df.to_excel(location.replace('.xls', '.xlsx'), index=False)

# 结果检验
# 假设你的新Excel文件保存在以下位置
file_path = 'D:/new/桌面/合并数据.xlsx'  # 请根据实际保存路径修改
# 读取Excel文件
df = pd.read_excel(file_path)
# 添加结果检验列
def check_result(row):
    if row['1=雾/霾，2=沙/尘'] == 0 and row['雾霾估计'] == 0 and row['沙尘估计'] == 0:
        return 1
    elif row['1=雾/霾，2=沙/尘'] == 1 and row['雾霾估计'] == 1:
        return 1
    elif row['1=雾/霾，2=沙/尘'] == 2 and row['沙尘估计'] == 1:
        return 1
    else:
        return 0
# 应用函数
df['结果检验'] = df.apply(check_result, axis=1)
# 保存回Excel
df.to_excel(file_path, index=False)
print("结果列及检验列已成功添加到", file_path)

# 未来隐藏状态预测
n_future_steps = 100
future_hidden_states = predict_future_hidden_states(
    final_transition_matrices, most_likely_hidden_state, n_future_steps)
print('\nfuture_hidden_states = \n', future_hidden_states)