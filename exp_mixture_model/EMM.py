# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.special import gammaln
import matplotlib.pyplot as plt

from distutils.version import StrictVersion
import scipy as sp
if StrictVersion(sp.__version__) >= StrictVersion("1.0.0"):
    from scipy.special import logsumexp
else:
    from scipy.misc import logsumexp


class EMM:
    """
    Exponential mixture model. EM algorithm is used for fitting.
    The EM algorithm is run several times to avoid local maxima
    as much as possible.

    Parameters
    ----------
    k : int, optional
        Number of components initially given.
    n_init : int, optional
        EM algorithm is run n_init times with different initial conditions.
        The estimated parameters that maximize the joint likelihood
        are selected.
    n_iter : int, optional
        Number of iterations in the EM algorithm.
    random_state : int, RandomState instance or None(default)
        Seed of random number generator.
    """

    def __init__(self, k=10, n_init=10, n_iter=1000, random_state=None):
        self.k = k
        self.n_init = n_init
        self.n_iter = n_iter
        self.rnd = random_state if type(random_state) is np.random.RandomState else np.random.RandomState(random_state)

        # parameters
        self.pi = None  # mixing weight
        self.mu = None  # mean of the exponential distribution

        # latent variables
        self.z = None
        self.n_each_exp = None  # number of samples that belong to each component
        self.k_final = None  # effective number of components

        # data
        self.n = None  # length of data
        self.x = None  # values of data

        # empirical log likelihood
        self.marginal_log_likelihood = None  # empirical marginal log likelihood
        self.joint_log_likelihood = None  # empirical joint log likelihood

    def fit(self, x):
        """
        Fit an EMM to data 'x'.

        Parameters
        ----------
        x : list or array
            Data.

        Returns
        -------
        pi : array
            Mixing weight of the individual exponential distribution in
            the estimated EMM. The array length is k_final.
        mu : array
            Mean parameter of the individual exponential distribution in
            the estimated EMM. The array length is k_final.
        """
        if 0 in x:
            x = np.array(x)
            self.x = x[x > 0]
            print("'x' contains 0, which has been removed.")
        else:
            self.x = x
        self.n = len(self.x)

        max_log_likelihood = - np.inf
        for i in range(self.n_init):
            # generate initial mu randomly
            mu0 = 10 ** self.rnd.uniform(np.log10(min(self.x)), np.log10(max(self.x)), self.k)

            z, temp_marginal_log_likelihood = _run_em(self.x, self.k, n_iter=self.n_iter, mu0=mu0)
            temp_results = _calc_joint_mle(self.x, z)

            # use the estimated parameter set that maximizes joint log likelihood
            if temp_results[-1] > max_log_likelihood:
                max_log_likelihood = temp_results[-1]
                self.marginal_log_likelihood = temp_marginal_log_likelihood
                self.pi, self.mu, self.z, self.n_each_exp, self.k_final, self.joint_log_likelihood = temp_results

        return self.pi, self.mu

    def generate(self, n, pi, mu):
        """
        Generate 'n' samples from EMM that has 'k' components.
        This method gives the instance the "fitted" status.
        The estimation of the parameters and likelihoods are
        based on the generated data.

        Parameters
        ----------
        n : int
            Sample size.
        pi : array or list
            Mixing weight of the individual exponential distribution in
            the EMM.
        mu : array or list, optional
            Mean of the individual exponential distribution in the EMM.

        Returns
        -------
        x : array
            Generated samples.
        """
        # set parameter values
        if len(pi) != self.k:
            raise ValueError("Length of 'pi' must be k=%d." % self.k)
        self.pi = np.array(pi)
        if len(mu) != self.k:
            raise ValueError("Length of 'mu' must be k=%d." % self.k)
        self.mu = np.array(mu)

        self.x, self.z, self.n_each_exp = _generate(n, self.pi, self.mu, self.rnd)

        # give the instance the "fitted" status
        self.k_final = self.k
        self.n = n
        self.marginal_log_likelihood = _marginal_log_likelihood(self.x, self.pi, self.mu)
        self.joint_log_likelihood = _joint_log_likelihood(n, self.n_each_exp, self.pi, self.mu)

        return self.x

    def pdf(self, x):
        """
        Probability density function value, p(x), at each element,
        x_i, of x.

        Parameters
        ----------
        x : float or array
            Data.

        Returns
        -------
        y: float or array
            Value of p(x).
        """
        self._check_fitted()
        try:
            y = np.zeros(len(x))
        except TypeError:
            # x is not array
            y = 0
        for pi_j, mu_j in zip(self.pi, self.mu):
            y += pi_j / mu_j * np.exp(- x / mu_j)
        return y

    def cdf(self, x):
        """
        Cumulative distribution function value, F(x), at each element,
        x_i, of x.

        Parameters
        ----------
        x : float or array
            Data.
        """
        self._check_fitted()
        try:
            y = np.zeros(len(x))
        except TypeError:
            # x is not array
            y = 0
        for pi_j, mu_j in zip(self.pi, self.mu):
            y += pi_j * (1 - np.exp(-x / mu_j))
        return y

    def ccdf(self, x):
        """
        Complementary cumulative distribution function value, 1 - F(x),
        at each element, x_i, of x.

        Parameters
        ----------
        x : float or array
            Data.
        """
        self._check_fitted()
        return 1 - self.cdf(x)

    def aic(self):
        """
        Akaike information criterion for the estimated EMM.
        """
        self._check_fitted()
        return - self.marginal_log_likelihood + 2 * self.k - 1

    def bic(self):
        """
        Baysian information criterion for the estimated EMM.
        """
        self._check_fitted()
        return - self.marginal_log_likelihood + (2 * self.k - 1) * np.log(self.n) / 2

    def aic_lvc(self):
        """
        AIC for the estimated EMM with latent variable completion.
        """
        self._check_fitted()
        return - self.joint_log_likelihood + 2 * self.k_final - 1

    def bic_lvc(self):
        """
        BIC for the estimated EMM with latent variable completion.
        """
        self._check_fitted()
        return (- self.joint_log_likelihood
                + (self.k_final - 1) * np.log(self.n) / 2 + np.sum(np.log(self.n_each_exp)) / 2
                )

    def nml_lvc(self, pc_emm=None):
        """
        Normalized maximum likelihood codelength for the estimated EMM
        with latent variable completion.

        Parameters
        -------
        pc_emm : PC_EMM instance, optional
            Parametric complexity of the EMM, calculated in advance.
        """
        self._check_fitted()
        if self.k_final == 1:
            return _nml_exp(self.n, self.mu[0])

        if pc_emm is None:
            pc_emm = PC_EMM(self.n, self.k_final)
            pc_emm.calc_pc()

        return - self.joint_log_likelihood + pc_emm.get_log_pc(self.n, self.k_final, np.min(self.mu), np.max(self.mu))

    def dnml(self, pc_mult=None):
        """
        Decomposed normalized maximum likelihood codelength for the
        estimated EMM.

        Parameters
        -------
        pc_mult : PC_mult instance, optional
            Parametric complexity of the multinomial distribution,
            calculated in advance.
        """
        self._check_fitted()
        if self.k_final == 1:
            return _nml_exp(self.n, self.mu[0])

        if pc_mult is None:
            pc_mult = PC_mult([self.n], self.k_final)
            pc_mult.calc_pc()

        l_z = (np.sum(- self.n_each_exp * np.log(self.pi))
               + np.log(pc_mult.get_pc(self.n, self.k_final))
               )
        l_x_z = (np.sum([_nml_exp_without_boundary_codelen(h_j, mu_j) for h_j, mu_j in zip(self.n_each_exp, self.mu)])
                 + _boundary_codelen_2stage(self.k_final, np.min(self.mu), np.max(self.mu))
                 )
        return l_x_z + l_z

    def print_result(self):
        """
        Print parameters and k_final of the estimated EMM.
        """
        self._check_fitted()
        head = "------- EMM(k=%d, k_final=%d) -------" % (self.k, self.k_final)
        print(head)
        for j in range(self.k_final):
            print("component %d: (pi, mu)=(%0.3f, %0.3f)" % (j+1, self.pi[j], self.mu[j]))
        print("-" * len(head))

    def plot_survival_probability(self, ax=None, show_flag=True):
        """
        Plot survival probability (= CCDF) for the estimated EMM and the
        given data.

        Parameters
        -------
        ax : matplotlib axis, optional
            Axis on which to plot. If None, a new figure is created.
        show_flag : bool, optional
            If True, a figure window shows up.
        """
        self._check_fitted()
        if ax is None:
            fig, ax = plt.subplots()
            _plot_survival_probability(ax, self)
            if show_flag:
                plt.show()
            return fig, ax
        else:
            _plot_survival_probability(ax, self)
            return ax

    def plot_odds_ratio(self, ax=None, show_flag=True):
        """
        Plot odds ratio for the estimated EMM and the given data.

        Parameters
        -------
        ax : matplotlib axis, optional
            Axis on which to plot. If None, a new figure is created.
        show_flag : bool, optional
            If True, a figure window shows up.
        """
        self._check_fitted()
        if ax is None:
            fig, ax = plt.subplots()
            _plot_odds_ratio(ax, self)
            if show_flag:
                plt.show()
            return fig, ax
        else:
            _plot_odds_ratio(ax, self)
            return ax

    def _check_fitted(self):
        """
        Check if this instance is fitted.
        """
        if self.k_final is None:
            raise AttributeError("This EMM instance is not fitted yet. Call 'fit' or 'generate' beforehand.")


class EMMs:
    """
    EMMs with different numbers of components. For each estimated EMM,
    six model selection criteria are calculated.

    Parameters
    ----------
    k_candidates : list or array, optional
        Candidates of the number of components.
    n_init : int, optional
        EM algorithm is run n_init times with different initial conditions.
        The estimated parameters that maximize the joint likelihood are
        selected.
    n_iter : int, optional
        Number of iterations in the EM algorithm.
    random_state : int, RandomState instance or None(default)
        Seed of random number generator.
    """
    def __init__(self, k_candidates=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100],
                 n_init=10, n_iter=1000, random_state=None):
        self.rnd = random_state if type(random_state) is np.random.RandomState else np.random.RandomState(random_state)

        self.model_candidates = {k: EMM(k, n_init, n_iter, self.rnd) for k in k_candidates}

        self.n = None # length of data
        self.x = None # values of data

        self.result_table = pd.DataFrame(index=self.model_candidates.keys(),
                                         columns=["k_final", "marginal_log_likelihood", "joint_log_likelihood",
                                                  "AIC", "BIC", "AIC_LVC", "BIC_LVC", "NML_LVC", "DNML"
                                                  ],
                                         dtype="float"
                                         )
        self.result_table["k_final"] = np.zeros(len(k_candidates), dtype='int32')
        self.result_table.index.rename('k', inplace=True)
        self.calculated_column = []

    def fit(self, x, verbose=False):
        """
        EMMs with different k values are fitted to data 'x'.

        Parameters
        ----------
        x : list or array
            Input data.

        verbose : bool, optional
            If True, message is printed upon completion of the calculation
            of each EMM.
        """
        if 0 in x:
            x = np.array(x)
            self.x = x[x > 0]
            print("'x' contains 0, which has been removed.")
        else:
            self.x = x
        self.n = len(self.x)

        for k, model in self.model_candidates.items():
            model.fit(self.x)
            self.result_table.loc[k, "k_final"] = model.k_final
            self.result_table.loc[k, "marginal_log_likelihood"] = model.marginal_log_likelihood
            self.result_table.loc[k, "joint_log_likelihood"] = model.joint_log_likelihood

            if verbose:
                print("Calculation of EMM(k=%d) has been completed." % k)

        self.calculated_column = ["k_final", "marginal_log_likelihood", "joint_log_likelihood"]

    def select(self, criterion="DNML"):
        """
        Calculate 'criterion' for each k value and returns the estimated
        EMM model.

        Parameters
        ----------
        criterion : str
            The criterion used for selecting the best EMM. Choose from
            'marginal_log_likelihood', 'joint_log_likelihood', 'AIC', 'BIC', 'AIC_LVC',
            'BIC_LVC', 'NML_LVC', or 'DNML'. The default is 'DNML'.

        Returns
        -------
        best_model: EMM instance
            Selected EMM model.
        """
        self._check_fitted()

        if criterion in ["marginal_log_likelihood", "joint_log_likelihood"]:
            best_k = self.result_table[criterion].idxmax()
        elif criterion in self.calculated_column:
            print("'%d' is already calculated.")
            best_k = self.result_table[criterion].idxmin()
        elif criterion == "AIC":
            best_k = self._calc_criterion(criterion, lambda model: model.aic())
        elif criterion == "BIC":
            best_k = self._calc_criterion(criterion, lambda model: model.bic())
        elif criterion == "AIC_LVC":
            best_k = self._calc_criterion(criterion, lambda model: model.aic_lvc())
        elif criterion == "BIC_LVC":
            best_k = self._calc_criterion(criterion, lambda model: model.bic_lvc())
        elif criterion == "NML_LVC":
            pc_emm = PC_EMM(self.n, np.max(list(self.model_candidates.keys())))
            pc_emm.calc_pc()
            best_k = self._calc_criterion(criterion, lambda model: model.nml_lvc(pc_emm))
        elif criterion == "DNML":
            pc_mult = PC_mult([self.n], np.max(list(self.model_candidates.keys())))
            pc_mult.calc_pc()
            best_k = self._calc_criterion(criterion, lambda model: model.dnml(pc_mult))
        else:
            raise ValueError(
                """Choose 'criterion' from ['marginal_log_likelihood', 'joint_log_likelihood',
                'AIC', 'BIC', 'AIC_LVC', 'BIC_LVC', 'NML_LVC', 'DNML'].
                """)

        best_model = self.model_candidates[best_k]
        return best_model

    def print_result_table(self, print_columns=None):
        """
        Print the result table for the estimated EMMs.

        Parameters
        ----------
        print_columns : str or list
            Columns to print in the result table. Choose from
            'k_final', 'marginal_log_likelihood', 'joint_log_likelihood', 'AIC', 'BIC',
            'AIC_LVC', 'BIC_LVC', 'NML_LVC', or 'DNML'.
            If None, all columns are printed.
        """
        self._check_fitted()
        if print_columns is None:
            print(self.result_table[self.calculated_column])
        else:
            print(self.result_table[print_columns])

    def _calc_criterion(self, criterion, calc_func):
        """
        Calculate 'criterion' for each EMM using 'calc_func',
        update self.result_table and self.calculated_column, and
        return best_k.

        Parameters
        ----------
        criterion : str
            Criterion to be calculated. Choose from 'AIC', 'BIC',
            'AIC_LVC', 'BIC_LVC', 'NML_LVC', or 'DNML'.
        calc_func : function
            Function to be used for calculating the model selection criterion.
        """
        for k, model in self.model_candidates.items():
            self.result_table.loc[k, criterion] = calc_func(model)
        self.calculated_column.append(criterion)
        return self.result_table[criterion].idxmin()  # return best_k

    def _check_fitted(self):
        """
        Check if this instance has been fitted.
        """
        if self.x is None:
            raise AttributeError("This EMMs instance is not in the 'fitted' state yet. Call 'fit' beforehand.")


def generate_emm(n, k, pi=None, mu=None):
    """
    Generate 'n' samples from EMM that has 'k' components.
    If you want to estimate the parameters or latent variables,
    use method 'generate' of class 'EMM', as in the sample code below.

    model = EMM(k=10)
    x = emm.generate(1000)
    z = emm.z
    pi, mu = emm.pi, emm.mu

    Parameters
    ----------
    n : int
        Sample size.
    k : int
        Number of components.
    pi : array or list, optional
        Mixture weight of the individual exponential distribution in the EMM.
        If None, each mixing weight, pi_i (i.e. i-th element of pi) is
        proportional to 1/mu_i, where mu_i is the i-th element of mu.
    mu : array or list, optional
        Mean of the individual exponential distribution in the EMM.
        If None, each element of mu, denoted by mu_j, is generated so that
        log10(mu_j) independently obeys the uniform density on [0, 6].

    Returns
    -------
    x : array
        Generated samples.
    """
    if mu is None:
        mu = _generate_mu(k)
    if pi is None:
        pi = _calculate_pi(mu)

    return EMM(k=k).generate(n, pi, mu)


class PC_EMM:
    """
    Parametric complexity of the EMM with (n, k), where n <= 'n_max'
    and k <= 'k_max'.

    Parameters
    ----------
    n_max : int
        Max of sample size.
    k_max : int
        Max of the number of components.
    """

    def __init__(self, n_max, k_max):
        self.n_max = n_max
        self.k_max = k_max

        # self.logC[k, n] is the logarithm of parametric complexity
        # of EMM whose number of components
        # and sample size are equal to 'k' and 'n', respectively.
        self.logC = None

    def calc_pc(self):
        """
        Calculate parametric complexity.
        """
        ns = np.arange(1, self.n_max + 1)
        self.logC = np.zeros((self.k_max + 1, self.n_max + 1)) - np.inf

        self.logC[1, 1:] = ns * np.log(ns / np.e) - gammaln(ns)

        for k in range(1, self.k_max + 1):
            self.logC[k, 0] = 0

        for k in range(1, self.k_max):
            for j in range(2, self.n_max + 1):
                logc = np.zeros(j - 1)

                r1_arange = np.arange(len(logc)) + 1
                r2_arange = r1_arange[::-1]

                logc += gammaln(j + 1)
                logc -= gammaln(r1_arange + 1)
                logc -= gammaln(r2_arange + 1)

                logc += r1_arange * np.log(r1_arange)
                logc += r2_arange * np.log(r2_arange)
                logc -= j * np.log(j)

                logc += self.logC[k, 1:j]
                logc += self.logC[1, j - 1:0:-1]
                self.logC[k + 1, j] = logsumexp(logc)

    def get_log_pc(self, n, k, mu_min, mu_max):
        """
        Logarithm of parametric complexity of the EMM specified by
        'n' and 'k'.

        Parameters
        ----------
        n : int
            Sample size.
        k : int
            Number of components.
        mu_min : float
            Min of the mean of the individual exponential distribution
            in the EMM.
        mu_max : float
            Max of the mean of the individual exponential distribution
            in the EMM.
        """
        if self.logC is None:
            raise AttributeError("Parametric complexity has not been calculated. Call 'calc' beforehand.")
        if n > self.n_max:
            raise ValueError(
                """Parametric complexities only for n <= 'n_max' (= %d) are calculated. 
                In the constructor, specify 'n_max', which has to be larger than or equal to %d.""" % (self.n_max, n)
            )
        if k > self.k_max:
            raise ValueError(
                """Parametric complexities only for k <= 'k_max' (= %d) are calculated. 
                In the constructor, specify 'k_max', which has to be larger than or equal to %d.""" % (self.k_max, k)
            )

        log_pc = self.logC[k, n] + _boundary_codelen_2stage(k, mu_min, mu_max)

        return log_pc


class PC_mult:
    """
       Parametric complexity of the multinomial distribution with (n, k),
       where n is in 'ns' and k <= 'k_max'.

       Parameters
       ----------
       ns : array
           Candidates of sample size.
       k_max : int
           Maximum number of components.
       """

    def __init__(self, ns, k_max):
        self.ns = ns
        self.k_max = k_max

        # self.C[n][k] is parametric complexity of the multinomial distribution
        # whose number of components
        # and sample size are equal to 'k' and 'n', respectively.
        self.Cs = None

    def calc_pc(self):
        """
        Calculate parametric complexity.
        """
        self.Cs = {}
        for n in self.ns:
            C = np.zeros(self.k_max + 1)  # 'C' is self.C[n].
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

            if self.k_max >= 3:
                for k in range(3, self.k_max + 1):
                    C[k] = C[k - 1] + n * C[k - 2] / (k - 2)
            self.Cs[n] = C

    def get_pc(self, n, k):
        """
        Parametric complexity of the multinomial distribution specified
        by 'n' and 'k'.

        Parameters
        ----------
        n : int
            Sample size.
        k : int
            Number of components.
        """
        if self.Cs is None:
            raise AttributeError("Parametric complexity has not been calculated. Call 'calc' beforehand.")
        if n not in self.ns:
            raise ValueError(
                """Parametric complexities only for 'ns' (= %s) are calculated. 
                In the constructor, specify 'ns', which is composed of %d.""" % (self.ns, n)
            )
        if k > self.k_max:
            raise ValueError(
                """Parametric complexities only for k <= 'k_max' (= %d) are calculated. 
                In the constructor, specify 'k_max', which has to be larger than or equal to %d.""" % (self.k_max, k)
            )
        return self.Cs[n][k]


def _integer_codelen(m):
    """
    Calculate the codelength of integer 'm'.
    """
    codelen = np.log(2.865) + np.log(2)
    t = np.log(np.abs(m) + 1)
    while t > 0:
        codelen += t
        t = np.log(t)
    return codelen


def _boundary_codelen_2stage(k, mu_min, mu_max):
    """
    Calculate the part of codelength of the EMM contributed by mu_min
    and mu_max.

    Parameters
    ----------
    k : int
        Number of components.
    mu_min : float
        Min of the mean of the individual exponential distribution in the EMM.
    mu_max : float
        Max of the mean of the individual exponential distribution in the EMM.
    """
    m_max = np.ceil(np.log(mu_max))
    m_min = min(np.floor(np.log(mu_min)), m_max - 1)
    return k * np.log(m_max - m_min) + _integer_codelen(m_min) + _integer_codelen(m_max)


def _nml_exp(n, mu):
    """
    Calculate the normalized maximum likelihood codelength of a single
    exponential distribution.

    Parameters
    ----------
    n : int
        Sample size.
    mu : int
        Mean of the exponential distribution.
    """
    if n == 0:
        return 0
    return n * np.log(mu) + n * np.log(n) - gammaln(n) + _boundary_codelen_2stage(1, mu, mu)


def _nml_exp_without_boundary_codelen(n, mu):
    """
    Calculate the normalized maximum likelihood codelength of the
    exponential distribution without the effect of constraining mu.

    Parameters
    ----------
    n : int
        Sample size.
    mu : int
        Mean of the exponential distribution.
    """
    if n == 0:
        return 0
    else:
        return n * np.log(mu) + n * np.log(n) - gammaln(n)


def _calc_joint_mle(x, z):
    """
    Calculate the maximum likelihood estimators of EMM whose latent
    variables completed.

    Parameters
    ----------
    x : array
        Data.
    z : array
        Estimated latent variables.

    Returns
    ----------
    pi : array
        Mixing weight of the individual exponential distribution in the EMM.
        The length of the array = k_final.
    mu : array
        Mean of the individual exponential distribution in the EMM.
        The length of the array = k_final. Sorted in ascending order.
    z : array
        Latent variables whose labels are swapped.
    n_each_exp : array
        Number of samples that belong to each component.
    k_final : int
        Number of components that are used at least once under the estimated
        latent variable values.
    joint_log_likelihood : float
        Logarithm of empirical joint likelihood.
    """
    n = len(x)  # length of data

    counts = np.bincount(z)
    z = (np.cumsum(counts > 0) - 1)[z]
    n_each_exp = counts[counts > 0]  # number of samples that belong to each component
    k_final = len(n_each_exp)  # effective number of components

    pi = n_each_exp / float(n)
    mu = np.dot(np.eye(k_final)[z].T, x) / n_each_exp

    # sort components' indices in ascending order of mu
    mu_ascend = np.argsort(mu)
    z = mu_ascend[z]
    n_each_exp = n_each_exp[mu_ascend]
    pi = pi[mu_ascend]
    mu = mu[mu_ascend]

    joint_log_likelihood = _joint_log_likelihood(n, n_each_exp, pi, mu)

    return pi, mu, z, n_each_exp, k_final, joint_log_likelihood


def _run_em(x, k, mu0=None, n_iter=1000):
    """
    Fit EMM to data 'x' by the EM algorithm.

    Parameters
    ----------
    x : array
        Data.
    k : int
        Number of components.
    mu0 : array, optional
        Initial values of mu.
    n_iter : int, optional
        Number of iterations in the EM algorithm.

    Returns
    ----------
    z : array
        Estimated latent variables.
    marginal_log_likelihood : float
        Empirical marginal log likelihood.
    """
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

    # estimate z
    log_gamma = log_pi[np.newaxis, :] - log_mu[np.newaxis, :] - x[:, np.newaxis] / np.exp(log_mu)[np.newaxis, :]
    z = np.argmax(log_gamma, axis=1).astype('int32')

    marginal_log_likelihood = _marginal_log_likelihood(x, np.exp(log_pi), np.exp(log_mu))

    return z, marginal_log_likelihood


def _marginal_log_likelihood(x, pi, mu):
    """
    Calculate the logarithm of the marginal likelihood.

    Parameters
    ----------
    x : array
        Data.
    pi : array
        Mixing weight of the individual component in the estimated EMM.
    mu : array
        Mean of the individual exponential distribution in the estimated EMM.
    """
    log_gamma = np.log(pi[np.newaxis, :]) - np.log(mu[np.newaxis, :]) - x[:, np.newaxis] / mu[np.newaxis, :]
    each_log_likelihood = logsumexp(log_gamma, axis=1)
    return np.sum(each_log_likelihood)


def _joint_log_likelihood(n, n_each_exp, pi, mu):
    """
    Calculate the logarithm of the joint likelihood.

    Parameters
    ----------
    n : int
        Sample size.
    n_each_exp : array
        Number of samples that belong to each component.
    pi : array
        Mixing weight of the individual exponential distribution in the
        estimated EMM.
    mu : array
        Mean of the individual exponential distribution in the estimated EMM.
    """
    temp = 0
    for j in range(len(n_each_exp)):
        if n_each_exp[j] > 0:
            temp += n_each_exp[j] * (np.log(pi[j]) - np.log(mu[j]))
    return temp - n


def _generate(n, pi, mu, random_state=None):
    """
    Generate samples from an EMM.

    Parameters
    ----------
    n : int
        Sample size.
    pi : array
        Mixing weight of the individual exponential distribution in the EMM.
    mu : array
        Mean of the individual exponential distribution in the EMM.
    random_state : int, RandomState instance or None(default)
        Seed of random number generator.

    Returns
    ----------
    x : array
        Generated sapmles.
    z : array
        Latent variable of each sample.
    n_each_exp : array
        Number of samples that belong to each component.
    """
    rnd = random_state if type(random_state) is np.random.RandomState else np.random.RandomState(random_state)
    k = len(pi)
    n_each_exp = rnd.multinomial(n, pi)  # number of samples in each component
    xs = []  # samples of each component
    for j in range(k):
        xs.append(rnd.exponential(scale=mu[j], size=n_each_exp[j]))
    x = np.concatenate(xs, axis=0)
    z = np.concatenate([np.ones(n_each_exp[i]) * i for i in range(k)])
    shuffled_indexes = rnd.permutation(n)
    x = x[shuffled_indexes]
    z = z[shuffled_indexes]
    return x, z, n_each_exp


def _generate_mu(k):
    """
    Generate mean of each exponential distribution in the EMM. Each element
    in mu, denoted by mu_j, is generated so that log10(mu_j) independently
    obeys the uniform density on [0, 6].

    Parameters
    ----------
    k : int
        Number of components.
    """
    rnd = np.random.RandomState(None)
    return np.sort(10 ** rnd.uniform(0, 6, k))


def _calculate_pi(mu):
    """
    Calculate a particular mixing weight vector. Each mixing weight,
    pi_i (i.e. i-th element of the output) is proportional to 1/mu_i,
    where mu_i is the i-th element of mu.

    Parameters
    ----------
    mu : array
        Mean of the individual exponential distribution in the EMM.
    """
    return (1 / mu) / np.sum(1 / mu)


def _plot_survival_probability(ax, emm):
    """
    Plot survival probability (= CCDF) for the estimated EMM and the
    given data.

    Parameters
    -------
    ax : matplotlib axis, optional
        Axis on which to plot.
    emm : EMM instance
        EMM fitted to data emm.x.
    """
    bins = np.logspace(np.log10(min(emm.x)) - 0.5, np.log10(max(emm.x)) + 0.5, 100000)
    ccdf = emm.ccdf(bins)

    # plot the survival probability for data emm.x
    ax.plot(np.concatenate(([bins[0]], np.kron(np.sort(emm.x), [1, 1]))),
            np.concatenate((np.kron(np.arange(emm.n, 0, -1, dtype="float64") / emm.n, [1, 1]),
                            [ccdf[-1]])),
            c="orange", label="empirical")

    # plot the survival probability for the EMM
    ax.plot(bins, ccdf, c="blue", label="EMM ($k$,$k^*$)=(%d,%d)" % (emm.k, emm.k_final))

    ax.set_xscale("log")
    ax.set_ylim((-0.02, 1.02))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel('x')
    ax.set_ylabel('Survival probability')
    ax.legend()


def _plot_odds_ratio(ax, emm):
    """
    Plot odds ratio for the estimated EMM and the given data.

    Parameters
    -------
    ax : matplotlib axis, optional
        Axis on which to plot.
    emm : EMM instance
        EMM fitted to data emm.x.
    """
    bins = np.logspace(np.log10(min(emm.x)) - 0.5, np.log10(max(emm.x)) + 0.5, 100000)
    cdf = emm.cdf(bins)

    # plot the odds ratio for data emm.x
    y = (np.arange(1, len(emm.x) + 1) - 0.5) / float(len(emm.x))
    ax.scatter(np.sort(emm.x), y / (1 - y), c="orange", label="empirical")

    # plot the odds ratio for the EMM
    ax.plot(bins, cdf / (1 - cdf), c="blue", label="EMM ($k$,$k^*$)=(%d,%d)" % (emm.k, emm.k_final))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('x')
    ax.set_ylabel('Odds ratio')
    ax.legend()
