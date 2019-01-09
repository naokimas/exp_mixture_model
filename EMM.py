import numpy as np
from sklearn.utils import shuffle
from scipy.misc import logsumexp
from scipy.special import gammaln

DEFAULT_EM_ITERATION_NUM = 1000
DEFAULT_EM_TRIAL_NUM = 10


class SC_MIX_EXP:
    def __init__(self, nmax, kmax):
        self.n = nmax
        self.kmax = kmax

    def calc_SC(self):
        self.logI = np.zeros(self.n + 1)  # I(0) cannot be defined

        ns = np.arange(1, self.n + 1)
        self.logI[0] = -np.inf
        self.logI[1:] = ns * np.log(ns / np.e) - gammaln(ns)

        self.logC = np.zeros((self.kmax + 1, self.n + 1)) - np.inf

        for k in range(1, self.kmax + 1):
            self.logC[k, 0] = 0

        self.logC[1, 1:] = self.logI[1:]

        for k in range(1, self.kmax):
            for j in range(2, self.n + 1):
                logc = np.zeros(j - 1)  # [logc(r1=1,r2=j-1),...,logc(r1=j-1,r2=1)]

                r1_arange = np.arange(len(logc)) + 1
                r2_arange = r1_arange[::-1]

                logc += gammaln(j + 1)
                logc -= gammaln(r1_arange + 1)
                logc -= gammaln(r2_arange + 1)

                logc += r1_arange * np.log(r1_arange)
                logc += r2_arange * np.log(r2_arange)
                logc -= j * np.log(j)

                logc += self.logC[k, 1:j]
                logc += self.logI[j - 1:0:-1]
                self.logC[k + 1, j] = logsumexp(logc)

    def getLogSC_without_boundary_code_length(self, K, n):
        return self.logC[K, n]


class SC_Mult:
    def __init__(self, ns, kmax):
        self.ns = ns
        self.kmax = kmax

    def calc_SC(self):
        self.Cs = {}
        for n in self.ns:
            C = np.zeros(self.kmax + 1)
            C[1] = 1

            logc = np.zeros(n - 1)
            ts = np.arange(n - 1) + 1

            logc += gammaln(n + 1)
            logc -= gammaln(ts + 1)
            logc -= gammaln(n - ts + 1)

            logc += ts * np.log(ts)
            logc += (n - ts) * np.log(n - ts)
            logc -= n * np.log(n)

            C[2] = np.sum(np.exp(logc))

            if self.kmax >= 3:
                for k in range(3, self.kmax + 1):
                    C[k] = C[k - 1] + n * C[k - 2] / (k - 2)
            self.Cs[n] = C

    def getSC(self, n, K):
        return self.Cs[n][K]
        

def integer_codelen(k):
    l = np.log(2.865) + np.log(2)
    t = np.log(np.abs(k) + 1)
    while t > 0:
        l += t
        t = np.log(t)
    return l


def boundary_2stage(mu_max, mu_min):
    a = np.ceil(np.log(mu_max))
    b = min(np.floor(np.log(mu_min)), a - 1)
    return np.log(a - b), integer_codelen(a) + integer_codelen(b)


def NML_EXP(n, mu):
    if n == 0:
        return 0
    boundary1, boundary2 = boundary_2stage(mu, mu)
    return n * np.log(mu) + n * np.log(n) - gammaln(n) + boundary1 + boundary2


def NML_EXP_without_boundary_code_length(n, mu):
    if n == 0:
        return 0
    return n * np.log(mu) + n * np.log(n) - gammaln(n)


class EMM:
    def __init__(self, k, n_EM=DEFAULT_EM_TRIAL_NUM, n_iter=DEFAULT_EM_ITERATION_NUM, rnd=np.random.RandomState(1234)):
        self.k_init = k
        self.rnd = rnd
        self.n_EM = n_EM
        self.n_iter = n_iter

        # parameter
        self.pis = None
        self.mus = None

        # latent variable
        self.z = None
        self.hs = None
        self.k_final = None

        # data
        self.n = None
        self.x = None

        # emp_log_likelihood
        self.emp_log_marginal_likelihood = None
        self.emp_log_joint_likelihood = None

    def fit(self, x):
        self.n = len(x)
        self.x = x
        if self.n_EM < 2:
            (self.pis, self.mus, self.z, self.hs), self.emp_log_marginal_likelihood = EMM.__run_EM(x, self.k_init,
                                                                                                       n_iter=self.n_iter)
            self.k_final = len(self.hs)
            self.emp_log_joint_likelihood = EMM.__log_joint_likelihood(self.n, self.k_final, self.hs, self.pis, self.mus)
        else:
            max_log_joint_likelihood = - np.inf
            for i in range(self.n_EM):
                mu0 = 10 ** self.rnd.uniform(np.log10(min(x)), np.log10(max(x)), self.k_init)
                (temp_pis, temp_mus, temp_z, temp_hs), temp_log_marginal_likelihood = EMM.__run_EM(x, self.k_init,
                                                                                                       n_iter=self.n_iter, mu0=mu0)
                temp_k_final = len(temp_hs)
                temp_log_joint_likelihood = EMM.__log_joint_likelihood(self.n, temp_k_final, temp_hs, temp_pis, temp_mus)
                if temp_log_joint_likelihood > max_log_joint_likelihood:
                    max_log_joint_likelihood = temp_log_joint_likelihood
                    self.pis = temp_pis
                    self.mus = temp_mus
                    self.z = temp_z
                    self.hs = temp_hs
                    self.k_final = temp_k_final
                    self.emp_log_marginal_likelihood = temp_log_marginal_likelihood
                    self.emp_log_joint_likelihood = temp_log_joint_likelihood
        return self.pis, self.mus

    def generate(self, n, pis=None, mus=None):
        self.n = n
        self.k_final = self.k_init
        self.mus = 10**self.rnd.uniform(0, 5, self.k_init) if mus is None else mus
        self.pis = (1 / self.mus) / np.sum(1 / self.mus) if pis is None else pis

        self.x, self.z, self.hs = EMM.__generate(self.n, self.pis, self.mus, self.rnd)

        self.emp_log_marginal_likelihood = EMM.__log_marginal_likelihood(self.x, self.pis, self.mus)
        self.emp_log_joint_likelihood = EMM.__log_joint_likelihood(self.n, self.k_final, self.hs, self.pis, self.mus)

        return self.x

    def get_info(self, sc_mix_exp=None, sc_mult=None):
        return {
            'n': self.n,
            'k_init': self.k_init,
            'k_final': self.k_final,
            'log_marginal_likelihood': self.emp_log_marginal_likelihood,
            'log_joint_likelihood': self.emp_log_joint_likelihood,
            'AIC': self.AIC(),
            'BIC': self.BIC(),
            'AIC_LVC': self.AIC_LVC(),
            'BIC_LVC': self.BIC_LVC(),
            'NML': self.NML(sc_mix_exp),
            'DNML': self.DNML(sc_mult)
        }

    @staticmethod
    def pdf(x, pis, mus):
        try:
            y = np.zeros(len(x))
        except TypeError:
            y = 0
        for pi, mu in zip(pis, mus):
            y += pi / mu * np.exp(- x / mu)
        return y

    @staticmethod
    def cdf(x, pis, mus):
        try:
            y = np.zeros(len(x))
        except TypeError:
            y = 0
        for pi, mu in zip(pis, mus):
            y += pi * (1 - np.exp(-x / mu))
        return y

    @staticmethod
    def __log_marginal_likelihood(x, pis, mus):
        pi = np.array(pis)
        mu = np.array(mus)
        log_gamma = np.log(pi[np.newaxis, :]) - np.log(mu[np.newaxis, :]) - x[:, np.newaxis] / mu[np.newaxis, :]
        log_each_likelihood = logsumexp(log_gamma, axis=1)
        return np.sum(log_each_likelihood)

    @staticmethod
    def __log_joint_likelihood(n, k, h, pi, mu):
        temp = 0
        for j in range(k):
            if h[j] > 0:
                temp += h[j] * (np.log(pi[j]) - np.log(mu[j]))
        return temp - n

    @staticmethod
    def __calc_joint_mle(x, z, k):
        z_onehot = np.eye(k)[z.astype('int32')]
        h = np.sum(z_onehot, axis=0)

        pi = h / len(x)
        mu = np.zeros(len(h))
        total = np.dot(z_onehot.T, x)
        for j in range(k):
            if h[j] == 0:
                mu[j] = 1
            else:
                mu[j] = total[j] / h[j]

        nonzero_index = np.argwhere(h > 0).flatten()

        return pi[nonzero_index], mu[nonzero_index], z, h[nonzero_index]

    @staticmethod
    def __run_EM(x, k, mu0=None, n_iter=DEFAULT_EM_TRIAL_NUM):
        if mu0 is None:
            mu0 = np.logspace(np.log10(min(x)), np.log10(max(x)), k)
        n = len(x)
        pi = np.ones(k) / k
        mu = np.array(mu0)
        log_pi = np.log(pi)
        log_mu = np.log(mu)
        log_x = np.log(x[:, np.newaxis])

        for i in range(n_iter):
            # E step
            log_gamma = log_pi[np.newaxis, :] - log_mu[np.newaxis, :] - x[:, np.newaxis] / np.exp(log_mu)[np.newaxis, :]
            log_gamma = log_gamma - logsumexp(log_gamma, axis=1, keepdims=True)

            # M step
            log_pi = logsumexp(log_gamma, axis=0) - np.log(n)
            log_mu = logsumexp(log_gamma + log_x, axis=0) - logsumexp(log_gamma, axis=0)

        emp_log_marginal_likelihood = EMM.__log_marginal_likelihood(x, np.exp(log_pi), np.exp(log_mu))

        log_gamma = log_pi[np.newaxis, :] - log_mu[np.newaxis, :] - x[:, np.newaxis] / np.exp(log_mu)[np.newaxis, :]
        z = np.argmax(log_gamma, axis=1)

        return EMM.__calc_joint_mle(x, z, k), emp_log_marginal_likelihood

    @staticmethod
    def __generate(n, pi, mu, rnd=np.random.RandomState(1234)):
        k = len(pi)
        hs = rnd.multinomial(n, pi)
        xs = []
        for j in range(k):
            xs.append(rnd.exponential(scale=mu[j], size=hs[j]))
        x = np.concatenate(xs, axis=0)
        z = np.concatenate([np.ones(hs[i]) * i for i in range(k)])
        x, z = shuffle(x, z, random_state=rnd)
        return x, z, hs

    def AIC(self):
        return - self.emp_log_marginal_likelihood + 2 * self.k_init - 1

    def BIC(self):
        return - self.emp_log_marginal_likelihood + (self.k_init - 1) / 2 * np.log(self.n) + np.sum(np.log(self.hs)) / 2
    
    def AIC_LVC(self):
        return - self.emp_log_joint_likelihood + 2 * self.k_final - 1

    def BIC_LVC(self):
        return - self.emp_log_joint_likelihood + (self.k_final - 1) / 2 * np.log(self.n) + np.sum(np.log(self.hs)) / 2

    def NML(self, sc_mix_exp=None):
        if sc_mix_exp is None:
            sc_mix_exp = SC_MIX_EXP(self.n, self.k_final)
            sc_mix_exp.calc_SC()

        if self.k_final == 1:
            return NML_EXP(self.n, self.mus[0])

        boundary1, boundary2 = boundary_2stage(np.max(self.mus), np.min(self.mus))
        boundary = self.k_final * boundary1 + boundary2

        return (np.sum(-self.hs * np.log(self.hs)) + self.n * np.log(self.n)
                + np.sum(self.hs * np.log(self.mus)) + self.n
                + sc_mix_exp.getLogSC_without_boundary_code_length(self.k_final, self.n) + boundary
                )

    def DNML(self, sc_mult=None):
        if sc_mult is None:
            sc_mult = SC_Mult([self.n], self.k_final)
            sc_mult.calc_SC()

        if self.k_final == 1:
            return NML_EXP(self.n, self.mus[0])

        l_z = (np.sum(-self.hs * np.log(self.hs)) + self.n * np.log(self.n) 
               + np.log(sc_mult.getSC(self.n, self.k_final))
               )

        boundary1, boundary2 = boundary_2stage(np.max(self.mus), np.min(self.mus))
        l_x_z = (np.sum([NML_EXP_without_boundary_code_length(n_i, mu_i) for n_i, mu_i in zip(self.hs, self.mus)])
                 + self.k_final * boundary1 + boundary2)

        return l_x_z + l_z


def find_optimal_model(x, k_candidates, ic_name="DNML",
                       n_EM=DEFAULT_EM_TRIAL_NUM, n_iter=DEFAULT_EM_ITERATION_NUM,
                       print_flag=False, rnd_seed=1234):
    sc_mix_exp = SC_MIX_EXP(len(x), np.max(k_candidates))
    sc_mix_exp.calc_SC()
    sc_mult = SC_Mult([len(x)], np.max(k_candidates))
    sc_mult.calc_SC()

    models = [EMM(k, n_EM=n_EM, rnd=np.random.RandomState(rnd_seed + k * 101)) for k in k_candidates]

    best_ic = np.inf
    best_model = None
    best_model_info = None
    for model in models:
        model.fit(x)
        info = model.get_info(sc_mix_exp, sc_mult)

        if print_flag:
            print("k:%d k_final:%d %s:%f" % (model.k_init, info["k_final"], ic_name, info[ic_name]))

        if info[ic_name] < best_ic:
            best_ic = info[ic_name]
            best_model = model
            best_model_info = info

    return best_model, best_model_info


if __name__ == "__main__":
    x = EMM(k=10).generate(n=1000)

    model1 = EMM(k=5)
    model1.fit(x)
    info1 = model1.get_info()
    print("--------- model1 ---------")
    print("k:", info1["k_init"])
    print("k_final:", info1["k_final"])
    mu_asc = np.argsort(model1.mus)
    print("pis:", model1.pis[mu_asc])
    print("mus:", model1.mus[mu_asc])
    print("log_marginal_likelihood:", info1["log_marginal_likelihood"])
    print("--------------------------")

    print()

    model2, info2 = find_optimal_model(x, [1, 2, 4, 8, 16], ic_name="DNML",  n_EM=5, print_flag=True)
    print("--------- model2 ---------")
    for name, info in info2.items():
        print("%s:" % name, info)
    print("--------------------------")
