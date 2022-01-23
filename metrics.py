"""
Computes GLM metrics.
@author: JingyiGF
"""

from .family import *


def get_deviance(y, yhat, family, theta):
    """Deviance metrics.
    Parameters
    ----------
    y : array
        Target labels of shape (n_samples, )
    yhat : array
        Predicted labels of shape (n_samples, )
    distr: str
        distribution
    Returns
    -------
    score : float
        Deviance of the predicted labels.
    """
    if family in ['poisson', 'neg-binomial']:
        LS = loglik(family, y, None, y, theta=theta)
    else:
        LS = 0

    L1 = loglik(family, y, None, yhat, theta=theta)
    score = -2 * (L1 - LS)
    return score


def get_pseudo_R2(y, yhat, ynull_, distr, theta):
    """Pseudo-R2 metric.
    Parameters
    ----------
    y : array
        Target labels of shape (n_samples, )
    yhat : array
        Predicted labels of shape (n_samples, )
    ynull_ : float
        Mean of the target labels (null model prediction)
    distr: str
        distribution
    Returns
    -------
    score : float
        Pseudo-R2 score.
    """
    if distr in ['poisson', 'neg-binomial']:
        LS = loglik(distr, y, None, y, theta=theta)
    else:
        LS = 0

    L0 = loglik(distr, y, None, ynull_, theta=theta)
    L1 = loglik(distr, y, None, yhat, theta=theta)

    if distr in ['poisson', 'neg-binomial']:
        score = (1 - (LS - L1) / (LS - L0))
    else:
        score = (1 - L1 / L0)
    return score


def get_llr(y, yhat, ynull_, distr, theta):
    """Pseudo-R2 metric.
    Parameters
    ----------
    y : array
        Target labels of shape (n_samples, )
    yhat : array
        Predicted labels of shape (n_samples, )
    ynull_ : float
        Mean of the target labels (null model prediction)
    distr: str
        distribution
    Returns
    -------
    score : float
        Pseudo-R2 score.
    """

    L0 = loglik(distr, y, None, ynull_, theta=theta)
    L1 = loglik(distr, y, None, yhat, theta=theta)

    score = 2 * (L1 - L0)
    return score


def get_accuracy(y, yhat):
    """Accuracy as ratio of correct predictions.
    Parameters
    ----------
    y : array
        Target labels of shape (n_samples, )
    yhat : array
        Predicted labels of shape (n_samples, )
    Returns
    -------
    accuracy : float
        Accuracy score.
    """
    return float(np.sum(y == yhat)) / yhat.shape[0]


def get_auc(x, y):
    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % x.shape)

    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is neither increasing nor decreasing "
                             ": {}.".format(x))

    area = direction * np.trapz(y, x)
    if isinstance(area, np.memmap):
        area = area.dtype.type(area)
    return area
