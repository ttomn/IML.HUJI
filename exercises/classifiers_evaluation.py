from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
from IMLearn.learners.classifiers import Perceptron
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

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


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    models = [GaussianNaiveBayes(), LDA()]
    model_names = ["Gaussian Naive Bias", "LDA"]
    symbols = np.array(["circle", "x", "square"])
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)
        y = y.astype(int)
        losses = [model.fit(X, y).loss(X, y) for model in models]

        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
        fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{model_names[i]} with accuracy of "
                                                            rf"{1 - losses[i]}}}$" for i in range(len(
            model_names))], horizontal_spacing=0.01, vertical_spacing=.03)

        for i, model in enumerate(models):
            # Fit models and predict over training set
            # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
            # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
            from IMLearn.metrics import accuracy

            fig.add_traces([
                go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                           marker=dict(color=model.predict(X), symbol=symbols[y],
                                       colorscale=[custom[0], custom[-1]],
                                       line=dict(color="black", width=1)))],
                rows=1, cols=i + 1)

            fig.update_layout(title=rf"$\textbf{{predictions of  {f} Dataset}}$",
                              margin=dict(t=100)) \
                .update_xaxes(visible=False).update_yaxes(visible=False)
            fig.add_traces([go.Scatter(x=model.mu_[:, 0], y=model.mu_[:, 1], mode="markers", showlegend=False,
                                       marker=dict(size=20, color="black", symbol="x"))], rows=1, cols=i + 1)
        fig.show()



if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron() todo
    compare_gaussian_classifiers()
