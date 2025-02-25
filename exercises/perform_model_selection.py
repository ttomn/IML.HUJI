from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn
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
    X = np.linspace(-1.2, 2, n_samples)
    f_x = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    epsilon = np.random.normal(0, noise, n_samples)
    y = f_x + epsilon
    df_X = pd.DataFrame(X, columns=['x'])
    series_y = pd.Series(y)
    train_X_df, train_y_df, test_X_df, test_y_df = split_train_test(X=df_X, y=series_y,
                                                                    train_proportion=2 / 3)
    train_X = train_X_df.to_numpy()
    train_y = train_y_df.to_numpy().flatten()
    test_X = test_X_df.to_numpy()
    test_y = test_y_df.to_numpy().flatten()

    fig = make_subplots(rows=1, cols=1).update_layout(
        title=rf"$\textbf{{Samples Before Model With Sample Size of {n_samples} and Noise of {noise}}}$")
    fig.add_traces([go.Scatter(x=X, y=f_x, mode="markers", showlegend=True,
                               marker=dict(color="black"), name="noiseless"),
                    go.Scatter(x=train_X.flatten(), y=train_y, mode="markers", showlegend=True,
                               marker=dict(color="blue"), name="train with noise"),
                    go.Scatter(x=test_X.flatten(), y=test_y, mode="markers", showlegend=True,
                               marker=dict(color="red"), name="test with noise")])
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    # train_X, train_y, test_X, test_y = split_train_test(X=df_X, y=series_y, train_proportion=1 / 3)
    min_k = 0
    min_validation_score = 10000000000
    train_score_array = np.zeros(11)
    validation_score_array = np.zeros(11)
    for k in range(0, 11):
        learner = PolynomialFitting(k)
        train_score, validation_score = cross_validate(learner, train_X, train_y, mean_square_error)
        if validation_score < min_validation_score:
            min_k, min_validation_score = k, validation_score
        train_score_array[k] = train_score
        validation_score_array[k] = validation_score

    fig = make_subplots(rows=1, cols=1).update_layout(
        title=rf"$\textbf{{Cross Validation Model With Sample Size of {n_samples} and Noise of {noise}}}$")
    fig.add_traces([go.Scatter(x=list(range(0, 11)), y=train_score_array, mode="lines", showlegend=True,
                               marker=dict(color="blue"), name="train score"),
                    go.Scatter(x=list(range(0, 11)), y=validation_score_array, mode="lines", showlegend=True,
                               marker=dict(color="red"), name="validation score")])
    fig.update_xaxes(title_text="degree of the polynom")
    fig.update_yaxes(title_text="scores")
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    learner = PolynomialFitting(min_k)
    learner.fit(train_X, train_y)
    print("test error for k=" + str(min_k) + " is " + str(round(mean_square_error(test_y, learner.predict(
        test_X)), 2)) + "\n")


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
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, y, train_size=n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso
    # regressions
    model_names = ["Ridge", "Lasso"]
    lambdas = np.linspace(start=0, stop=2, num=n_evaluations)
    min_validation_error_dict = dict()
    for model in model_names:
        learner_train_score = np.zeros(n_evaluations)
        learner_validation_score = np.zeros(n_evaluations)
        for i, lam in enumerate(lambdas):
            if model == "Lasso":
                learner = Lasso(alpha=lam, fit_intercept=True)
            else:
                learner = RidgeRegression(lam=lam, include_intercept=True)
            train_score, validation_score = cross_validate(learner, train_X, train_y, mean_square_error, cv=5)
            learner_train_score[i] = train_score
            learner_validation_score[i] = validation_score

        min_validation_error_dict[model] = lambdas[learner_validation_score.argmin()]
        fig = make_subplots(rows=1, cols=1)

        fig.update_layout(
            title=rf"$\textbf{{Cross Validation {model} Model With Sample Size of}}$")

        fig.add_trace(
            go.Scatter(x=lambdas, y=learner_train_score, mode="lines", showlegend=True,
                       marker=dict(color="blue"), name="train error"))
        fig.add_trace(
            go.Scatter(x=lambdas, y=learner_validation_score, mode="lines", showlegend=True,
                       marker=dict(color="red"), name="validation error"))

        fig.update_xaxes(title_text="regularization parameter")
        fig.update_yaxes(title_text="error")
        fig.show()
    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    for lam in min_validation_error_dict.values():
        ridge = RidgeRegression(lam=lam, include_intercept=True)
        ridge.fit(train_X, train_y)
        print("ridge regressor with lambda of ", lam, "has test error of ", ridge.loss(test_X, test_y))
        lasso = Lasso(alpha=lam, fit_intercept=True)
        lasso.fit(train_X, train_y)
        print("lasso regressor with lambda of ", lam, "has test error of ",
              mean_square_error(lasso.predict(test_X), test_y))
    linear = LinearRegression(include_intercept=True)
    linear.fit(train_X, train_y)
    print("linear regressor has test error of ", linear.loss(test_X, test_y))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(n_samples=100, noise=5)
    select_polynomial_degree(n_samples=100, noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter(n_samples=50, n_evaluations=500)
