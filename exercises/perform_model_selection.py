from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.random.uniform(-1.2, 2, n_samples)
    f_x = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    epsilon = np.random.normal(0, noise, n_samples)
    y = f_x + epsilon
    df_X = pd.DataFrame(X, columns=['x'])
    series_y = pd.Series(y)
    series_real_y = pd.Series(f_x)
    train_real_X_df, train_real_y_df, test_real_X_df, test_real_y_df = split_train_test(X=df_X,
                                                                                        y=series_real_y,
                                                                                        train_proportion=1 / 3)
    train_real_X = train_real_X_df.to_numpy().flatten()
    train_real_y = train_real_y_df.to_numpy().flatten()
    test_real_X = test_real_X_df.to_numpy().flatten()
    test_real_y = test_real_y_df.to_numpy().flatten()

    fig = make_subplots(rows=1, cols=1)

    fig.update_layout(title=rf"$\textbf{{The Noiseless Model}}$")

    fig.add_trace(go.Scatter(x=train_real_X, y=train_real_y, mode="markers", showlegend=True,
                             marker=dict(color="blue"), name="train"))
    fig.add_trace(go.Scatter(x=test_real_X, y=test_real_y, mode="markers", showlegend=True,
                             marker=dict(color="red"), name="test"))
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="f(x)")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    # train_X, train_y, test_X, test_y = split_train_test(X=df_X, y=series_y, train_proportion=1 / 3)
    train_X_df, train_y_df, test_X_df, test_y_df = split_train_test(X=df_X, y=series_y,
                                                                    train_proportion=1 / 3)
    train_X = train_X_df.to_numpy()
    train_y = train_y_df.to_numpy().flatten()
    test_X = test_X_df.to_numpy()
    test_y = test_y_df.to_numpy().flatten()

    for k in range(11):
        learner = PolynomialFitting(k)
        train_score, validation_score = cross_validate(learner, train_X, train_y, mean_square_error)
        print("for k=" + str(k) + " the scores are:\n")
        print("train score=" + str(train_score) + "\n")
        print("validation score=" + str(validation_score) + "\n")

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k = 5
    learner = PolynomialFitting(k)
    learner.fit(train_X, train_y)
    print("test error for k=" + str(k) + " is " + str(round(mean_square_error(test_y, learner.predict(
        test_X)), 2)))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(n_samples=100, noise=5)
