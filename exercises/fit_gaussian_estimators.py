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
    samples_2_distance = np.zeros(100)
    samples_2_amount = np.arange(10, 1010, 10)
    for amount in samples_2_amount:
        sample_1.fit(gaussian_1000_samples[:amount])
        samples_2_distance[amount // 10 - 1] = abs(sample_1.mu_ - TRUE_EXPECTATION)
    fig = make_subplots(rows=1, cols=1) \
        .add_traces(
        [go.Scatter(x=samples_2_amount, y=samples_2_distance, mode='lines', marker=dict(color="black"),
                    showlegend=False)]) \
        .update_layout(title_text=r"$\text{bias as a function of sample size}$", height=300) \
        .update_xaxes(title_text="Number of samples") \
        .update_yaxes(title_text="bias")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_samples_3 = sample_1.pdf(gaussian_1000_samples)
    go.Figure([go.Scatter(x=gaussian_1000_samples, y=pdf_samples_3, mode='markers')],
              layout=go.Layout(title=r"$\text{PDF as a function of the samples values}$",
                               xaxis_title="samples values", yaxis_title="PDF", height=300)).show()
    return


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
