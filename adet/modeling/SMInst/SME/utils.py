# coding:utf-8

import numpy as np
import torch
import math

def direct_sigmoid(x):
    """Apply the sigmoid operation.
    """
    y = 1./(1.+1./np.exp(x))
    dy = y*(1-y)
    return y


def inverse_sigmoid(x):
    """Apply the inverse sigmoid operation.
            y = -ln(1-x/x)
    """
    y = -1 * np.log((1-x)/x)
    return y


def transform(X, components_, explained_variance_, mean_=None, whiten=False):
    """Apply dimensionality reduction to X.
    X is projected on the first principal components previously extracted
    from a training set.
    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        New data, where n_samples is the number of samples
        and n_features is the number of features.
    components_: array-like, shape (n_components, n_features)
    mean_: array-like, shape (n_features,)
    explained_variance_: array-like, shape (n_components,)
                        Variance explained by each of the selected components.
    whiten : bool, optional
        When True (False by default) the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.
    Returns
    -------
    X_new : array-like, shape (n_samples, n_components)
    """

    if mean_ is not None:
        X = X - mean_
    X_transformed = np.dot(X, components_.T)
    if whiten:
        X_transformed /= np.sqrt(explained_variance_)
    return X_transformed


def inverse_transform(X, components_, explained_variance_, mean_=None, whiten=False):
    """Transform data back to its original space.
    In other words, return an input X_original whose transform would be X.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_components)
        New data, where n_samples is the number of samples
        and n_components is the number of components.
    components_: array-like, shape (n_components, n_features)
    mean_: array-like, shape (n_features,)
    explained_variance_: array-like, shape (n_components,)
                        Variance explained by each of the selected components.
    whiten : bool, optional
        When True (False by default) the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.

    Returns
    -------
    X_original array-like, shape (n_samples, n_features)
    """
    if whiten:
        X_transformed = np.dot(X, np.sqrt(explained_variance_[:, np.newaxis]) * components_)
    else:
        X_transformed = np.dot(X, components_)

    if mean_ is not None:
        X_transformed = X_transformed + mean_

    return X_transformed


class IOUMetric(object):
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc


def soft_thresholding(x, lm):
    ze_ = torch.zeros(size=x.size(), device=x.device)
    return torch.sign(x) * torch.maximum(torch.abs(x) - lm, ze_)


@torch.no_grad()
def fast_ista(b, A, lmbda, max_iter):
    """
    This is the fast Iterative Shrinkage-Thresholding Algorithm to solve the following objective:
    min: {L2_norm(Ax - b) + L1_norm(x)}
    :param b: input data with shape: [n_samples, n_features]
    :param A: a pre-learned Dictionary, with shape: [n_coeffs, n_features]
    :param lmbda: sparsity term to control the importance of the L1 term
    :param max_iter:
    :return: sparse codes with shape: [n_samples, n_coeffs]
    """
    n_coeffs, n_feats = A.size()
    n_samples = b.size()[0]
    x = torch.zeros(size=(n_samples, n_coeffs), device=b.device)
    t = 1.
    z = torch.zeros(size=(n_samples, n_coeffs), device=b.device)
    L = torch.linalg.norm(A, ord=2) ** 2  # Lipschitz constant, 2-norm (largest sing. value)

    for k in range(max_iter):
        x_old = x.clone()
        z = z + torch.matmul(b - torch.matmul(z, A), A.T) / L
        x = soft_thresholding(z, lmbda / L)
        t0 = t
        t = (1. + math.sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - x_old)

    return x
