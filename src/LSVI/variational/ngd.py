from typing import Callable

import jax
import jax.numpy as jnp


def ngd(OP_key: jax.Array, sampling: Callable, sufficient_statistic: Callable, tgt_log_density: Callable,
        eta_init: jnp.ndarray, n_iter: int, n_samples: int,
        lr_schedule=1.0, sanity=lambda _: False):
    """
    Natural gradient descent algorithm for variational inference within exponential families
    NOTE : the objective is the MC version of \int \bar{q}(\log(q / \pi) - 1), WHICH IS NOT the un-normalised KL div nor the KL.
    :param OP_key: PRNGKey, needed to generate samples from both the target and current fitted distribution
    :param sampling: sampling method from the variational family
    :param sufficient_statistic: sufficient statistic of the variational family
    :param tgt_log_density: log-density of the target distribution
    :param eta_init: initial parameter characterizing the initial variational distribution
    :param n_iter: number of iterations of the fixed-point scheme
    :param n_samples: number of samples to draw at each iteration, used to replace the exact expectations by empirical
        expectations
    :param lr_schedule: float or array of floats, learning rate schedule
    :param sanity: callable, function to check whether a natural parameter defines a valid distribution, if set then call momentum_backtracking
    """

    vmapped_sampling = jax.vmap(sampling, in_axes=(None, 0))
    vmapped_sufficient_statistic = jax.vmap(sufficient_statistic)
    vmapped_tgt_log_density = jax.vmap(tgt_log_density)
    iter_keys = jax.random.split(OP_key, n_iter)

    def momentum_backtracking(lr, eta, next_eta):
        """
        Momentum backtracking to ensure that the natural parameter defines a valid distribution
        This function divides by two the learning rate until the natural parameter defines a valid distribution
        """
        lr = jax.lax.while_loop(lambda _lr: sanity(- next_eta * _lr + eta),
                                lambda _lr: _lr / 2, lr)
        return lr

    def l_or_ukl(eta, keys):
        theta = eta.at[:-1].get()
        samples = vmapped_sampling(theta, keys)
        logq = vmapped_sufficient_statistic(samples) @ eta
        integrand = logq - vmapped_tgt_log_density(samples) - 1
        ukl = jnp.mean(integrand)
        return ukl, samples

    def routine_iter(eta, inps):
        key, lr = inps
        keys = jax.random.split(key, n_samples)
        val, grad = jax.value_and_grad(l_or_ukl, has_aux=True)(eta, keys)
        _, samples = val
        X = vmapped_sufficient_statistic(samples)
        next_eta = jnp.linalg.inv(X.T @ X / n_samples) @ grad
        lr = momentum_backtracking(lr, eta, next_eta)
        next_eta = eta - lr * next_eta
        return next_eta

    def fun_iter(eta, inps):
        next_eta = routine_iter(eta, inps)
        return next_eta, next_eta

    if isinstance(lr_schedule, float):
        lr_schedule = jnp.full(n_iter, lr_schedule)

    _, etas = jax.lax.scan(fun_iter, eta_init, (iter_keys, lr_schedule))
    etas = jnp.insert(etas, 0, eta_init, axis=0)
    return etas


from LSVI.variational.exponential_family import GenericNormalDistribution, GenericMeanFieldNormalDistribution


def ngd_on_gaussian_kl(OP_key: jax.Array, tgt_log_density: Callable,
                       eta_init: jnp.ndarray, n_iter: int, n_samples: int,
                       lr_schedule=1.0, sanity=lambda _: False):
    r"""
    Natural gradient descent algorithm for variational inference within exponential families
    NOTE : the objective is the MC version of \int \bar{q}(\log(\bar{q} / \pi) = KL(\bar{q}\mid \bar{\pi}) up to some additive constant
    :param OP_key: PRNGKey, needed to generate samples from both the target and current fitted distribution
    :param tgt_log_density: log-density of the target distribution
    :param eta_init: initial parameter characterizing the initial variational distribution
    :param n_iter: number of iterations of the fixed-point scheme
    :param n_samples: number of samples to draw at each iteration, used to replace the exact expectations by empirical
        expectations
    :param lr_schedule: float or array of floats, learning rate schedule
    :param sanity: callable, function to check whether a natural parameter defines a valid distribution, if set then call momentum_backtracking
    """
    dimension = int(jnp.sqrt(len(eta_init) - 3 / 4) - 1 / 2)
    gaussian = GenericNormalDistribution(dimension)

    def make_logpdf(eta):
        def logpdf(x):
            return jax.scipy.stats.multivariate_normal.logpdf(x, *gaussian.get_mean_cov(eta))

        return logpdf

    sampling = gaussian.sampling_method
    sufficient_statistic = gaussian.sufficient_statistic
    vmapped_sampling = jax.vmap(sampling, in_axes=(None, 0))
    vmapped_sufficient_statistic = jax.vmap(sufficient_statistic)
    vmapped_tgt_log_density = jax.vmap(tgt_log_density)
    iter_keys = jax.random.split(OP_key, n_iter)

    def momentum_backtracking(lr, eta, next_eta):
        """
        Momentum backtracking to ensure that the natural parameter defines a valid distribution
        This function divides by two the learning rate until the natural parameter defines a valid distribution
        """
        lr = jax.lax.while_loop(lambda _lr: sanity(- next_eta * _lr + eta),
                                lambda _lr: _lr / 2, lr)
        return lr

    def kl(eta, keys):
        theta = eta.at[:-1].get()
        samples = vmapped_sampling(theta, keys)
        logq = make_logpdf(eta)(samples)
        integrand = logq - vmapped_tgt_log_density(samples)
        estimate_of_kl = jnp.sum(integrand)
        return estimate_of_kl, samples

    def routine_iter(eta, inps):
        key, lr = inps
        keys = jax.random.split(key, n_samples)
        val, grad = jax.value_and_grad(kl, has_aux=True)(eta, keys)
        _, samples = val
        X = vmapped_sufficient_statistic(samples)
        next_eta = jnp.linalg.pinv(X.T @ X) @ grad
        lr = momentum_backtracking(lr, eta, next_eta)
        next_eta = eta - lr * next_eta
        return next_eta

    def fun_iter(eta, inps):
        next_eta = routine_iter(eta, inps)
        return next_eta, next_eta

    if isinstance(lr_schedule, float):
        lr_schedule = jnp.full(n_iter, lr_schedule)

    _, etas = jax.lax.scan(fun_iter, eta_init, (iter_keys, lr_schedule))
    etas = jnp.insert(etas, 0, eta_init, axis=0)
    return etas


def ngd_on_mf_gaussian_kl(OP_key: jax.Array, tgt_log_density: Callable,
                          eta_init: jnp.ndarray, n_iter: int, n_samples: int,
                          lr_schedule=1.0, sanity=lambda _: False):
    r"""
    Natural gradient descent algorithm for variational inference within exponential families
    NOTE : the objective is the MC version of \int \bar{q}(\log(\bar{q} / \pi) = KL(\bar{q}\mid \bar{\pi}) up to some additive constant
    :param OP_key: PRNGKey, needed to generate samples from both the target and current fitted distribution
    :param tgt_log_density: log-density of the target distribution
    :param eta_init: initial parameter characterizing the initial variational distribution
    :param n_iter: number of iterations of the fixed-point scheme
    :param n_samples: number of samples to draw at each iteration, used to replace the exact expectations by empirical
        expectations
    :param lr_schedule: float or array of floats, learning rate schedule
    :param sanity: callable, function to check whether a natural parameter defines a valid distribution, if set then call momentum_backtracking
    """
    dimension = int((len(eta_init) - 1) / 2)
    gaussian = GenericMeanFieldNormalDistribution(dimension)

    def make_logpdf(eta):
        def logpdf(x):
            mean, cov = gaussian.get_mean_cov(eta.at[:-1].get())
            L = cov ** 0.5
            y = (x - mean) / L
            return (-1 / 2 * jnp.einsum('...i,...i->...', y, y) - dimension / 2 * jnp.log(2 * jnp.pi)
                    - jnp.log(L).sum(-1))

        return logpdf

    sampling = gaussian.sampling_method
    sufficient_statistic = gaussian.sufficient_statistic
    vmapped_sampling = jax.vmap(sampling, in_axes=(None, 0))
    vmapped_sufficient_statistic = jax.vmap(sufficient_statistic)
    vmapped_tgt_log_density = jax.vmap(tgt_log_density)
    iter_keys = jax.random.split(OP_key, n_iter)

    def momentum_backtracking(lr, eta, next_eta):
        """
        Momentum backtracking to ensure that the natural parameter defines a valid distribution
        This function divides by two the learning rate until the natural parameter defines a valid distribution
        """
        lr = jax.lax.while_loop(lambda _lr: sanity(- next_eta * _lr + eta),
                                lambda _lr: _lr / 2, lr)
        return lr

    def kl(eta, keys):
        theta = eta.at[:-1].get()
        samples = vmapped_sampling(theta, keys)
        logq = make_logpdf(eta)(samples)
        integrand = logq - vmapped_tgt_log_density(samples)
        estimate_of_kl = jnp.sum(integrand)
        return estimate_of_kl, samples

    def routine_iter(eta, inps):
        key, lr = inps
        keys = jax.random.split(key, n_samples)
        val, grad = jax.value_and_grad(kl, has_aux=True)(eta, keys)
        _, samples = val
        X = vmapped_sufficient_statistic(samples)
        next_eta = jnp.linalg.pinv(X.T @ X) @ grad
        lr = momentum_backtracking(lr, eta, next_eta)
        next_eta = eta - lr * next_eta
        return next_eta

    def fun_iter(eta, inps):
        next_eta = routine_iter(eta, inps)
        return next_eta, next_eta

    if isinstance(lr_schedule, float):
        lr_schedule = jnp.full(n_iter, lr_schedule)

    _, etas = jax.lax.scan(fun_iter, eta_init, (iter_keys, lr_schedule))
    etas = jnp.insert(etas, 0, eta_init, axis=0)
    return etas


def ngd_on_gaussian_kl(OP_key: jax.Array, tgt_log_density: Callable,
                       eta_init: jnp.ndarray, n_iter: int, n_samples: int,
                       lr_schedule=1.0, sanity=lambda _: False):
    r"""
    Natural gradient descent algorithm for variational inference within exponential families
    NOTE : the objective is the MC version of \int \bar{q}(\log(\bar{q} / \pi) = KL(\bar{q}\mid \bar{\pi}) up to some additive constant
    :param OP_key: PRNGKey, needed to generate samples from both the target and current fitted distribution
    :param tgt_log_density: log-density of the target distribution
    :param eta_init: initial parameter characterizing the initial variational distribution
    :param n_iter: number of iterations of the fixed-point scheme
    :param n_samples: number of samples to draw at each iteration, used to replace the exact expectations by empirical
        expectations
    :param lr_schedule: float or array of floats, learning rate schedule
    :param sanity: callable, function to check whether a natural parameter defines a valid distribution, if set then call momentum_backtracking
    """
    dimension = int(jnp.sqrt(len(eta_init) - 3 / 4) - 1 / 2)
    gaussian = GenericNormalDistribution(dimension)

    def make_logpdf(eta):
        def logpdf(x):
            mean, cov = gaussian.get_mean_cov(eta.at[:-1].get())
            return jax.scipy.stats.multivariate_normal.logpdf(x, mean, cov)

        return logpdf

    sampling = gaussian.sampling_method
    sufficient_statistic = gaussian.sufficient_statistic
    vmapped_sampling = jax.vmap(sampling, in_axes=(None, 0))
    vmapped_sufficient_statistic = jax.vmap(sufficient_statistic)
    vmapped_tgt_log_density = jax.vmap(tgt_log_density)
    iter_keys = jax.random.split(OP_key, n_iter)

    def momentum_backtracking(lr, eta, next_eta):
        """
        Momentum backtracking to ensure that the natural parameter defines a valid distribution
        This function divides by two the learning rate until the natural parameter defines a valid distribution
        """
        lr = jax.lax.while_loop(lambda _lr: sanity(- next_eta * _lr + eta),
                                lambda _lr: _lr / 2, lr)
        return lr

    def kl(eta, keys):
        theta = eta.at[:-1].get()
        samples = vmapped_sampling(theta, keys)
        logq = make_logpdf(eta)(samples)
        integrand = logq - vmapped_tgt_log_density(samples)
        estimate_of_kl = jnp.sum(integrand)
        return estimate_of_kl, samples

    def routine_iter(eta, inps):
        key, lr = inps
        keys = jax.random.split(key, n_samples)
        val, grad = jax.value_and_grad(kl, has_aux=True)(eta, keys)
        _, samples = val
        X = vmapped_sufficient_statistic(samples)
        next_eta = jnp.linalg.pinv(X.T @ X) @ grad
        lr = momentum_backtracking(lr, eta, next_eta)
        next_eta = eta - lr * next_eta
        return next_eta

    def fun_iter(eta, inps):
        next_eta = routine_iter(eta, inps)
        return next_eta, next_eta

    if isinstance(lr_schedule, float):
        lr_schedule = jnp.full(n_iter, lr_schedule)

    _, etas = jax.lax.scan(fun_iter, eta_init, (iter_keys, lr_schedule))
    etas = jnp.insert(etas, 0, eta_init, axis=0)
    return etas
