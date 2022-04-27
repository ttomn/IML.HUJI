from typing import NoReturn

from numpy.dual import inv

from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, n_k = np.unique(y, return_counts=True)
        self.pi_ = n_k / y.shape[0]
        X_and_y = np.concatenate((np.reshape(y, (y.size, 1)), X), axis=1)
        self.mu_ = np.array([np.mean(X_and_y[X_and_y[:, 0] == i, 1:], axis=0) for i in self.classes_])
        self.vars_ = np.zeros([self.classes_.shape[0], X.shape[1]])
        for k in range(self.classes_.shape[0]):
            temp = X[np.where(y == self.classes_[k])] - self.mu_[k]
            temp = np.sum(temp**2, axis=0)
            self.vars_[k] = temp / (n_k[k] - 1)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        classes_indexes = np.argmax(self.likelihood(X), axis=1)
        y = (self.classes_[classes_indexes])
        return y

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = np.zeros([X.shape[0], self.classes_.shape[0]])
        for k in range(0, self.classes_.shape[0]):
            likelihoods[k] = np.sum(np.log(self.vars_[k])) * (-0.5) + np.log(self.pi_[k])
            sigma_k_minus_1 = inv(np.diag(self.vars_[k]))
            temp = X - self.mu_[k]
            for i in range(X.shape[0]):
                likelihoods[i][k] -= ((temp[i].T @ sigma_k_minus_1 @ temp[i]) * 0.5)
        return likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self._predict(X))
