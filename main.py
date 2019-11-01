from data_generation import generate_linear_data
from mcmc import BayesianRegressor, OLSRegressor
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)
# intercept - b_0, b_1, b_2, b_3 ...
PARAMETERS = np.array([0, .3, .5])
X, Y = generate_linear_data(PARAMETERS, n_samples=30, std_error=1)

bayesian_reg = BayesianRegressor(beta_pior_sig=50, gibbs_drift_rate=.15)
bayesian_reg.fit(X, Y)
bayesian_reg.plot()

ols_reg = OLSRegressor()
ols_reg.fit(X, Y)
ols_reg.plot()

bayesian_reg.summary()
ols_reg.summary()

plt.show()
