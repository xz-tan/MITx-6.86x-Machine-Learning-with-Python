"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
import math
from scipy import stats


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    log_E_array = []
    log_likelihood = 0
    for n in range(X.shape[0]):
        prob_K_list = []
        sum_prob_K = 0
        x = X[n]
        if not(np.all(x == 0)):
            index_val_not_0 = np.where(x != 0)  # To get index of non-zero values
            x = x[index_val_not_0]  # To get non-zero values
            for K in range(len(mixture.mu)):
                mu_K, var_K, p_K = mixture.mu[K][index_val_not_0], mixture.var[K], mixture.p[K]  # Extract values of Mu that are non-zero
                prob_K = math.log(p_K + 10 ** -16) + math.log(stats.multivariate_normal.pdf(x, mean=mu_K, cov=var_K) + 10 ** -16)
                sum_prob_K += (p_K + 10 ** -16) * stats.multivariate_normal.pdf(x, mean=mu_K, cov=var_K)
                prob_K_list.append(prob_K)
            prob_K_list = np.array(prob_K_list)
            correct_prob_K_list = prob_K_list - logsumexp(prob_K_list)
            log_E_array.append(correct_prob_K_list)
            log_likelihood += math.log(sum_prob_K + 10 ** -16)
        else:
            log_E_array.append(np.log(mixture.p))

    E_array = np.exp(np.array(log_E_array))

    return E_array, log_likelihood
    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    E_array_transpose = np.transpose(post)  # Shape(K,n)
    X_array = np.where(X > 0, 1, X)  # Replace user rating that is more than 0 with 1, Shape(n,d)
    C_u = np.sum(X_array, axis=1)  # Shape(n, 1)
    mu_array = []

    p_array = np.sum(E_array_transpose, axis=1) / X.shape[0]  # Shape(K,)

    for k in range(E_array_transpose.shape[0]):
        criteria_array =  E_array_transpose[k] @ X_array  # Shape(1,d)
        not_pass_criteria_index = np.where(criteria_array < 1)
        mu_k = (E_array_transpose[k] @ np.multiply(X_array, X)) / (E_array_transpose[k] @ X_array)
        mu_k[not_pass_criteria_index] = mixture.mu[k][not_pass_criteria_index]
        mu_array.append(mu_k)
    mu_array = np.array(mu_array)

    var_array = []
    for k in range(E_array_transpose.shape[0]):
        summation_part = 0
        for n in range(X.shape[0]):
            x = X[n]
            if not(np.all(x == 0)):
                index_val_not_0 = np.where(x != 0)  # To get index of non-zero values
                x = x[x != 0]  # To get non-zero values
                summation_part += E_array_transpose[k][n] * np.square(np.linalg.norm(x - mu_array[k][index_val_not_0]))
        var_k = summation_part / (E_array_transpose[k] @ C_u)
        if var_k >= min_variance:
            var_array.append(var_k)
        else:
            var_array.append(min_variance)
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
        mixture = mstep(X, post, mixture)
    return mixture, post, new_log_likelihood
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
