import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import t


def output_formatter(b_dict, p_dict):
    keys = b_dict.keys()
    print('{:20}{:20}{:20}'.format('Parameter', 'Value', 'P-value'))
    print('-----------------------------------------------')
    for key in keys:
        print('{:20}{:<20.3f}{:<20.3f}'.format(key, b_dict[key], p_dict.get(key, np.nan)))
    print('-----------------------------------------------')


class BayesianRegressor:
    def __init__(self, beta_pior_sig, gibbs_drift_rate):
        self.beta_pior_sig = beta_pior_sig
        self.g_drift = gibbs_drift_rate
        self.coef = {}
        self.p_vals = {}
        self.gibbs_sampler = Gibbs(self.posterior, self.g)
        self.coef_names = []
        self.X = None
        self.y = None
        self.N = None
        self.M = None

    @staticmethod
    def scale_prior(s):
        if s <= 0:
            return 0
        return 1/s

    def normal_beta_prior(self, b):
        return 1 / self.beta_pior_sig * np.exp(-np.sum(b**2) / (2 * self.beta_pior_sig**2))

    def prior(self, s, b):
        return self.scale_prior(s) * self.normal_beta_prior(b)

    def likelihood(self, s, b):
        return 1 / s ** self.N * np.exp(-np.sum((np.matmul(self.X, b) - self.y)**2) / (2 * s**2))

    def posterior(self, x):
        s, b = self.split_x(x)
        return self.prior(s, b) * self.likelihood(s, b)

    def g(self, x):
        return np.random.normal(loc=x, scale=self.g_drift)

    @staticmethod
    def split_x(x):
        return x[0], x[1:]

    @staticmethod
    def comb_x(s, b):
        return np.hstack([s, b])

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.N, self.M = X.shape
        start_x = np.zeros(self.M + 1)
        start_x[0] = 1
        self.gibbs_sampler.sample(start_x)
        self.gibbs_sampler.calc_kdes()
        self.gibbs_sampler.calc_map()
        self.coef_names = ['s'] + ['b_' + str(i) for i in range(self.N)]
        self.coef.update(zip(self.coef_names, self.gibbs_sampler.map.values()))
        self.p_vals.update(zip(self.coef_names, self.calc_p_vals()))

    def calc_p_vals(self):
        pvals = np.empty(self.M + 1)
        for i, key in zip(range(self.N + 1), self.gibbs_sampler.g_kdes.keys()):
            beta = self.gibbs_sampler.map[key]
            if beta >= 0:
                lower = np.min(self.gibbs_sampler.g_kdes[key].dataset)
                pvals[i] = 2 * np.abs(self.gibbs_sampler.g_kdes[key].integrate_box_1d(lower, 0))
            else:
                high = np.max(self.gibbs_sampler.g_kdes[key].dataset)
                pvals[i] = 2 * np.abs(self.gibbs_sampler.g_kdes[key].integrate_box_1d(0, high))
        return pvals

    def summary(self):
        print('Bayesian Regression')
        print('-----------------------------------------------')
        output_formatter(self.coef, self.p_vals)

    def plot(self):
        cols = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        i = 0
        coef_vals = list(self.coef.values())
        x = np.linspace(np.min(coef_vals) - 1,  np.max(coef_vals) + 1, 1000)
        for c_key, g_key in zip(self.coef.keys(), self.gibbs_sampler.g_kdes.keys()):
            plt.plot(x, self.gibbs_sampler.g_kdes[g_key].evaluate(x), label=c_key + ' Bayes', c=cols[i])
            i += 1

        plt.legend()


class Gibbs:
    def __init__(self, posterior_function, sample_function):
        self.posterior_f = posterior_function
        self.sample_function = sample_function
        self.samples = None
        self.g_kdes = {}
        self.map = {}
        self.p_vals = {}

    def sample(self, start_x, n_samples=5000):
        self.samples = np.empty([n_samples, len(start_x)], dtype='float')
        indices = range(len(start_x))
        current_x = np.array(start_x, dtype='float')
        for i in range(n_samples):
            candidate_xs = self.sample_function(current_x)
            order = np.random.permutation(indices)
            for ind in order:
                candidate_x = current_x.copy()
                candidate_x[ind] = candidate_xs[ind]
                den = self.posterior_f(current_x)
                if den == 0:
                    p = 1
                else:
                    p = self.posterior_f(candidate_x) / den

                if p > np.random.uniform(0, 1):
                    current_x = candidate_x
            self.samples[i, :] = current_x

    def plot_dists(self):
        x = np.linspace(np.min(self.samples), np.max(self.samples), 1000)
        for i in self.g_kdes.keys():
            plt.plot(x, self.g_kdes[i].evaluate(x), label=i)

    def calc_kdes(self):
        for i in range(self.samples.shape[1]):
            self.g_kdes['b_' + str(i)] = gaussian_kde(self.samples[:, i])

    def calc_map(self):
        x = np.linspace(np.min(self.samples), np.max(self.samples), 1000)
        for key in self.g_kdes.keys():
            self.map[key] = x[self.g_kdes[key].evaluate(x) == np.max(self.g_kdes[key].evaluate(x))][0]

        return self.map


class OLSRegressor:
    def __init__(self):
        self.coef = {}
        self.coef_s = {}
        self.A = None
        self.N = None
        self.M = None
        self.s2 = None
        self.t_s = None
        self.p_s = {}
        self.b = None

    def fit(self, X, y):
        self.N, self.M = X.shape
        self.A = np.linalg.inv(np.matmul(X.transpose(), X))
        coef = np.matmul(self.A, np.matmul(X.transpose(), y))
        self.s2 = sum((np.matmul(X, coef) - y) ** 2) / (self.N - 1)
        coef_s = np.sqrt(self.s2 * np.diag(self.A))
        self.t_s = np.abs(coef) / coef_s
        b_names = ['b_' + str(i) for i in range(self.N)]
        self.coef['s'] = np.sqrt(self.s2)
        self.coef.update(zip(b_names, coef))
        self.coef_s.update(zip(b_names, coef_s))
        self.p_s.update(zip(b_names, self.calc_p_value()))
        self.b = coef

    def calc_p_value(self):
        df = self.N - self.M
        p = []
        for ts in self.t_s:
            pv = 2*(1 - t.cdf(x=ts, df=df))
            p.append(pv)

        return p

    def summary(self):
        print('OLS Regression')
        print('-----------------------------------------------')
        output_formatter(self.coef, self.p_s)

    def plot(self):
        cols = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
        i = 1
        coef_vals = list(self.coef.values())
        x = np.linspace(np.min(coef_vals) - 1,  np.max(coef_vals) + 1, 1000)
        plt.plot([self.coef['s'], self.coef['s']], [0, 3], label='s' + ' OLS', c=cols[0], linestyle='--')
        for key in self.coef_s.keys():
            plt.plot(x, t.pdf(x, df=self.N - 2, loc=self.coef[key], scale=self.coef_s[key]), label=key + ' OLS',
                     c=cols[i], linestyle='--')
            i += 1
        plt.legend()

    def predict(self, X):
        return np.matmul(X, self.b)
