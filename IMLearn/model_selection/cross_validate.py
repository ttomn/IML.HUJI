from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[
    float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    indexes = np.random.permutation(X.shape[0])
    X_folds = list()
    y_folds = list()
    for i in range(cv):
        from_index = int(i * X.shape[0] / cv)
        to_index = int((i + 1) * X.shape[0] / cv)
        fold_indexes = indexes[from_index:to_index]
        X_folds.append(X[fold_indexes, :])
        y_folds.append(y[fold_indexes])

    train_score = 0
    validation_score = 0
    for i in range(cv):
        train_x = np.concatenate(X_folds[:i] + X_folds[i + 1:], axis=0)
        train_y = np.concatenate(y_folds[:i] + y_folds[i + 1:], axis=0)
        validation_X = X_folds[i]
        validation_y = y_folds[i]
        estimator.fit(train_x, train_y)
        train_score += scoring(train_y, estimator.predict(train_x))
        validation_score += scoring(validation_y, estimator.predict(validation_X))
    train_score /= cv
    validation_score /= cv
    return train_score, validation_score
