"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
import math
from scipy import stats


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    E_array = []
    log_likelihood = 0
    for n in range(X.shape[0]):
        prob_K_list = []
        sum_prob_K = 0
        x = X[n]
        for K in range(len(mixture.mu)):
            mu_K, var_K, p_K = mixture.mu[K], mixture.var[K], mixture.p[K]
            prob_K = p_K * stats.multivariate_normal.pdf(x, mean=mu_K, cov=var_K)
            sum_prob_K += prob_K
            prob_K_list.append(prob_K)
        correct_prob_K_list = [y / sum_prob_K for y in prob_K_list]
        E_array.append(correct_prob_K_list)
        log_likelihood += math.log(sum_prob_K)
    E_array = np.array(E_array)
    return E_array, log_likelihood
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    E_array_transpose = np.transpose(post)  # Shape(K,n)
    mu_array = (E_array_transpose @ X) / np.reshape(np.sum(E_array_transpose, axis=1), (-1, 1))
    p_array = np.sum(E_array_transpose, axis=1) / X.shape[0]  # Shape(K,)
    var_array = []
    for k in range(E_array_transpose.shape[0]):
        var_array_K = (E_array_transpose[k] @ np.square(np.linalg.norm(X - mu_array[k], axis=1))) / (X.shape[1] * np.sum(E_array_transpose[k]))
        var_array.append(var_array_K)
    var_array = np.array(var_array)
    return GaussianMixture(mu_array, var_array, p_array)
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log_likelihood = 0
    new_log_likelihood = 0
    while old_log_likelihood == 0 or new_log_likelihood - old_log_likelihood >= 10 ** -6 * abs(new_log_likelihood):
        old_log_likelihood = new_log_likelihood
        post, new_log_likelihood = estep(X, mixture)
        mixture = mstep(X, post)
    return mixture, post, new_log_likelihood
    raise NotImplementedError

