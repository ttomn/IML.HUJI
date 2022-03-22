from plotly.subplots import make_subplots

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
from numpy import unravel_index
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"

SAMPLES_AMOUNT = 1000
TRUE_EXPECTATION = 10
EXPECTATION_AMOUNT_Q5 = 200


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    gaussian_1000_samples = np.random.normal(TRUE_EXPECTATION, 1, size=SAMPLES_AMOUNT)
    univariate_gaussian = UnivariateGaussian()
    univariate_gaussian.fit(gaussian_1000_samples)
    print("(", univariate_gaussian.mu_, ", ", univariate_gaussian.var_, ")", sep="")

    # Question 2 - Empirically showing sample mean is consistent
    samples_2_distance = np.zeros(100)
    samples_2_amount = np.arange(10, 1010, 10)
    for amount in samples_2_amount:
        univariate_gaussian.fit(gaussian_1000_samples[:amount])
        samples_2_distance[amount // 10 - 1] = abs(univariate_gaussian.mu_ - TRUE_EXPECTATION)
    fig = make_subplots(rows=1, cols=1) \
        .add_traces(
        [go.Scatter(x=samples_2_amount, y=samples_2_distance, mode='lines', marker=dict(color="black"),
                    showlegend=False)]) \
        .update_layout(title_text=r"$\text{bias as a function of sample size}$", height=300) \
        .update_xaxes(title_text="Number of samples") \
        .update_yaxes(title_text="bias")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_samples_3 = univariate_gaussian.pdf(gaussian_1000_samples)
    go.Figure([go.Scatter(x=gaussian_1000_samples, y=pdf_samples_3, mode='markers')],
              layout=go.Layout(title=r"$\text{PDF as a function of the samples values}$",
                               xaxis_title="samples values", yaxis_title="PDF", height=300)).show()
    # in the plot above I am expecting to see that as more samples we are using the smaller the bias will be.
    return


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    gaussian_1000_multi_samples = np.random.multivariate_normal(mu, cov, size=SAMPLES_AMOUNT)
    multivariate_gaussian = MultivariateGaussian()
    multivariate_gaussian.fit(gaussian_1000_multi_samples)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    X_axis = np.linspace(-10, 10, EXPECTATION_AMOUNT_Q5)
    Y_axis = np.linspace(-10, 10, EXPECTATION_AMOUNT_Q5)
    multivariate_gaussian_likelihood = np.zeros([EXPECTATION_AMOUNT_Q5, EXPECTATION_AMOUNT_Q5])
    for i in range(EXPECTATION_AMOUNT_Q5):
        for j in range(EXPECTATION_AMOUNT_Q5):
            multivariate_gaussian_likelihood[i][j] = multivariate_gaussian.log_likelihood(
                np.array([X_axis[i], 0, Y_axis[j], 0]), cov, gaussian_1000_multi_samples)
    px.imshow(multivariate_gaussian_likelihood, x=X_axis, y=Y_axis,
              title=r"$\text{log likelihood as a function of expectation change}$",
              labels=dict(x="first feature expectation", y="third feature expectation",
                          color="log likelihood of the samples")).show()
    # we can see in the graph above that the maximal likelihood is when f1~4.5, f3~0. as you go further
    # from this point the likelihood decrease more and more.

    # Question 6 - Maximum likelihood
    indexes = unravel_index(multivariate_gaussian_likelihood.argmax(), multivariate_gaussian_likelihood.shape)
    print(round(X_axis[indexes[0]], 4))
    print(round(Y_axis[indexes[1]], 4))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
