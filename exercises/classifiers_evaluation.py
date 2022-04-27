from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
from IMLearn.learners.classifiers import Perceptron
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from math import atan2, pi
from IMLearn.metrics import accuracy

from utils import decision_surface, custom

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, [0, 1]], data[:, 2]


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_func(p: Perceptron, X_inner: np.ndarray, y_inner: int):
            losses.append(p.loss(X, y))

        p = Perceptron(callback=loss_func)
        p.fit(X, y)

        # Plot figure
        fig = make_subplots(rows=1, cols=1) \
            .add_traces(
            [go.Scatter(x=list(range(len(losses))), y=losses, mode='lines',
                        showlegend=False)]) \
            .update_layout(
            title_text=r"$\text{training loss values as a function of the training iterations}$", height=300) \
            .update_xaxes(title_text="training iterations") \
            .update_yaxes(title_text="training loss")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    models = [GaussianNaiveBayes(), LDA()]
    model_names = ["Gaussian Naive Bias", "LDA"]
    symbols = np.array(["circle", "x", "square"])
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        X, y = load_dataset(f)
        y = y.astype(int)
        accuracies = [accuracy(y, model.fit(X, y).predict(X)) for model in models]

        fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{model_names[i]} with accuracy of "
                                                            rf"{accuracies[i]}}}$" for i in range(len(
            model_names))], horizontal_spacing=0.01)

        fig.update_layout(title=rf"$\textbf{{Predictions of {f} Dataset}}$",
                          margin=dict(t=100))

        for i, model in enumerate(models):
            fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                     marker=dict(color=model.predict(X), symbol=symbols[y],
                                                 colorscale=[custom[0], custom[2]],
                                                 line=dict(color="black", width=1))), row=1, col=i + 1)

            fig.add_trace(go.Scatter(x=model.mu_[:, 0], y=model.mu_[:, 1], mode="markers", showlegend=False,
                                     marker=dict(size=20, color="black", symbol="x")), row=1, col=i + 1)

            for j in range(len(symbols)):
                if model_names[i] == "LDA":
                    fig.add_trace(get_ellipse(model.mu_[j], model.cov_), row=1, col=i + 1)
                else:
                    fig.add_trace(get_ellipse(model.mu_[j], np.diag(model.vars_[j])), row=1, col=i + 1)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
