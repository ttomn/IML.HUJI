import numpy as np
from typing import Tuple
from IMLearn.metalearners import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def decision_surface_particial(predict, xrange, yrange, t, density=120, dotted=False, colorscale=custom,
                               showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()], t)
    if dotted:
        return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers",
                          marker=dict(color=pred, size=1, colorscale=colorscale, reversescale=False),
                          hoverinfo="skip", showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=colorscale, reversescale=False,
                      opacity=.7, connectgaps=True, hoverinfo="skip", showlegend=False, showscale=showscale)


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500, isq5=False):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    learner = AdaBoost([[], DecisionStump], n_learners)
    learner.fit(train_X, train_y)
    losses_train = np.zeros(n_learners)
    losses_test = np.zeros(n_learners)
    learners_amount = np.arange(1, n_learners + 1)
    for i, t in enumerate(learners_amount):
        losses_train[i] = learner.partial_loss(train_X, train_y, t)
        losses_test[i] = learner.partial_loss(test_X, test_y, t)
    fig = go.Figure(layout=go.Layout(
        title=rf"$\textbf{{Loss of Adaboost Running With Decision Stump As a Function of Amount "
              rf"of Base Estimators With Noise of {noise}}}$",
        margin=dict(t=100)))
    fig.add_trace(go.Scatter(x=learners_amount, y=losses_train, mode="lines", name="train_samples"))
    fig.add_trace(go.Scatter(x=learners_amount, y=losses_test, mode="lines", name="test_samples"))
    fig.show()

    symbols = np.array(["circle", "x"])
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    if not isq5:  # in q5 we are asked to plot the graphs of q1 and q4
        # Question 2: Plotting decision surfaces
        T = [5, 50, 100, 250]
        fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{m} iterations}}$" for m in T],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        for i, t in enumerate(T):
            fig.add_traces(
                [decision_surface_particial(learner.partial_predict, lims[0], lims[1], t=t, showscale=False),
                 go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                            marker=dict(color=train_y, symbol=symbols[0], colorscale=[custom[0], custom[-1]],
                                        line=dict(color="black", width=1))),
                 go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                            marker=dict(color=test_y, symbol=symbols[1], colorscale=[custom[0], custom[-1]],
                                        line=dict(color="black", width=1)))],
                rows=(i // 2) + 1, cols=(i % 2) + 1)

        fig.update_layout(title=rf"$\textbf{{Decision Boundaries Using AdaBoost}}$",
                          margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig.show()

        # Question 3: Decision surface of best performing ensemble
        best_t = losses_test.argmin()
        go.Figure(
            data=[decision_surface_particial(learner.partial_predict, lims[0], lims[1], t=best_t + 1,
                                             showscale=False),
                  go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,

                             marker=dict(color=test_y, symbol=symbols[1], colorscale=[custom[0], custom[-1]],
                                         line=dict(color="black", width=1)))],
            layout=go.Layout(title=rf"$\textbf{{Best Ensemble Size is {best_t + 1} With Loss "
                                   rf"{losses_test[best_t]}}}$")).update_xaxes(
            visible=False).update_yaxes(
            visible=False).show()

    # Question 4: Decision surface with weighted samples
    go.Figure(data=[decision_surface_particial(learner.partial_predict, lims[0], lims[1], t=n_learners,
                                               showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(size=learner.D_ / np.max(learner.D_) * 5, color=train_y,
                                           symbol=symbols[0], colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))],
              layout=go.Layout(title=rf"$\textbf{{Decision Boundaries of Training Set With Point Size "
                                     rf"Proportional to Its Weight In the Last Iteration "
                                     rf"With Noise of {noise}}}$")).update_xaxes(
        visible=False).update_yaxes(
        visible=False).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0, n_learners=250, train_size=5000, test_size=500)  # q1-4
    fit_and_evaluate_adaboost(noise=0.4, n_learners=250, train_size=5000, test_size=500, isq5=True)  # q5
