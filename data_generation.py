import numpy as np


def generate_linear_data(parameters, n_samples, std_error):
    x = np.concatenate([np.ones([n_samples, 1]),
                        np.random.uniform(low=0, high=5, size=[n_samples, len(parameters) - 1])]
                       , axis=1)
    y = np.matmul(x, parameters) + np.random.normal(loc=0, scale=std_error, size=n_samples)

    return x, y


# # have intercept, trend, and period of year
# N = 12
# x = np.linspace(0, 10, N)
# y1 = 3 + np.sin(x) + 0.5*x
#
# one_hot = np.eye(N)
# X1 = np.hstack([one_hot, x[:, np.newaxis]])
#
# X = np.tile(X1, [10, 1])
# X = X + np.random.normal(loc=0, scale=0.01, size=X.shape)
# y = np.tile(y1[:, np.newaxis], [10, 1]) + np.random.normal(loc=0, scale=0.1, size=[N*10, 1])
#
# from mcmc import OLSRegressor
#
# r = OLSRegressor()
# r.fit(X, y)
#
# yp = r.predict(X1)
#
# import matplotlib.pyplot as plt
#
# plt.plot(x, y1, label='truth')
# plt.plot(x, yp, label='predict')
# plt.legend()
# plt.show()
