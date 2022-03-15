from plotly.subplots import make_subplots

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"

TRUE_EXPECTATION = 10

def test_univariate_gaussian():

    # Question 1 - Draw samples and print fitted model
    gaussian_1000_samples = np.random.normal(TRUE_EXPECTATION, 1, size=1000)
    sample_1 = UnivariateGaussian()
    sample_1.fit(gaussian_1000_samples)
    print(sample_1.mu_, sample_1.var_)

    # Question 2 - Empirically showing sample mean is consistent
    samples_2 = np.arange(10,1010,10)
    samples_2_distance = np.zeros(100)
    samples_2_amount = np.arange(10,1010,10)
    for amount in samples_2_amount:
        sample_1.fit(gaussian_1000_samples[amount])
        samples_2_distance[amount/10-1] = np.abs(sample_1.mu_-TRUE_EXPECTATION)


    # Question 3 - Plotting Empirical PDF of fitted model
    raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
