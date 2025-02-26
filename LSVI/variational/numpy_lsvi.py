import numpy as np


def get_residual(y, X, beta):
    """
    Compute the residual of the regression
    """
    return np.var(y - beta @ X.T)


def lsvi(sampling, sufficient_statistic, tgt_log_density, upsilon_init, n_iter, n_samples, lr_schedule=1.0,
         return_all=False, sanity=lambda _: False, target_residual_schedule=np.inf):
    """
    Fixed-point scheme for Variational Inference problem on exponential families, given some regression estimators.
    :param sampling: sampling method from the variational family
    :param sufficient_statistic: sufficient statistic of the variational family
    :param tgt_log_density: log-density of the target distribution, can loop over the first dimension for the different samples
    :param upsilon_init: initial parameter characterizing the initial variational distribution
    :param n_iter: number of iterations of the fixed-point scheme
    :param n_samples: number of samples to draw at each iteration, used to replace the exact expectations by empirical
        expectations
    :param lr_schedule: float or array of floats, learning rate schedule
    :param return_all: bool, whether to return all the intermediate results, only the residual variances //including samples and evaluation of log-density
    :param sanity: callable, function to check whether a natural parameter defines a valid distribution, if set then call momentum_backtracking
    :param target_residual_schedule: float or array of floats, desired variance for the residuals
    """

    def momentum_backtracking(lr, upsilon, next_upsilon, y, X, target_residual):
        """
        Momentum backtracking to ensure that the natural parameter defines a valid distribution
        This function divides by two the learning rate until the natural parameter defines a valid distribution
        """
        while sanity(next_upsilon * lr + (1 - lr) * upsilon):
            lr /= 2
        current_residual = get_residual(y, X, next_upsilon * lr + (1 - lr) * upsilon)
        if current_residual <= target_residual:
            return lr, current_residual
        else:
            lr = min(np.sqrt(target_residual / current_residual), lr)
        new_residual = get_residual(y, X, next_upsilon * lr + (1 - lr) * upsilon)
        return lr, new_residual

    upsilons = np.array([upsilon_init] * (n_iter + 1))
    if isinstance(lr_schedule, float):
        lr_schedule = np.full(n_iter, lr_schedule)
    if isinstance(target_residual_schedule, float):
        target_residual_schedule = np.full(n_iter, target_residual_schedule)
    if return_all:
        ys = np.empty((n_iter, n_samples), dtype=np.float64)
        _sample = sampling(upsilon_init[:-1], n_samples)
        _X = sufficient_statistic(_sample)
        Xs = np.empty((n_iter, *_X.shape), dtype=_X.dtype)
        sampless = np.empty((n_iter, *_sample.shape), dtype=_sample.dtype)

    for i_iter in range(n_iter):
        lr = lr_schedule[i_iter]
        target_residual = target_residual_schedule[i_iter]
        current_upsilon = upsilons[i_iter]
        theta = current_upsilon[:-1]
        samples = sampling(theta, n_samples)
        X = sufficient_statistic(samples)
        y = tgt_log_density(samples)
        next_upsilon = np.linalg.lstsq(X, y, rcond=None)[0]
        lr, residual = momentum_backtracking(lr, current_upsilon, next_upsilon, y, X, target_residual)
        next_upsilon = next_upsilon * lr + (1 - lr) * current_upsilon
        upsilons[i_iter + 1] = next_upsilon
        if return_all:
            ys[i_iter] = y
            Xs[i_iter] = X
            sampless[i_iter] = samples

    if return_all:
        return upsilons, (Xs, ys, sampless)
    return upsilons, None
