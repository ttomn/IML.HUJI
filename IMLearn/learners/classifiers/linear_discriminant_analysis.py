from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

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
        cov_sum = np.zeros([X.shape[1], X.shape[1]])
        for i in range(y.shape[0]):
            temp = X[i] - self.mu_[np.where(self.classes_ == y[i])]
            cov_sum += temp.T @ temp
        self.cov_ = cov_sum / (y.shape[0] - self.classes_.shape[0])
        self._cov_inv = inv(self.cov_)

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
        # y = np.zeros(X.shape[0])
        # for i in range(X.shape[0]):
        #     y[i] = (self.classes_[classes_indexes[i]])

        # a = self.mu_ @ self._cov_inv
        # b = []
        # for k in range(self.classes_.shape[0]):
        #     b.append(np.log(self.pi_[k]) - 0.5 * self.mu_[k] @ self._cov_inv @ self.mu_[k].T)
        # y = np.zeros(X.shape[0])
        # for i in range(X.shape[0]):
        #     max_index = 0
        #     max_value = a[0] @ X[i] + b[0]
        #     for k in range(1, self.classes_.shape[0]):
        #         curr_value = a[k] @ X[i] + b[k]
        #         if curr_value > max_value:
        #             max_index = k
        #             max_value = curr_value
        #     y[i] =(self.classes_[max_index])
        # return y

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
        a = self.mu_ @ self._cov_inv
        b = []
        for k in range(self.classes_.shape[0]):
            b.append(np.log(self.pi_[k]) - 0.5 * self.mu_[k] @ self._cov_inv @ self.mu_[k].T)
        for i in range(X.shape[0]):
            for k in range(0, self.classes_.shape[0]):
                likelihoods[i][k] = a[k] @ X[i] + b[k]
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
