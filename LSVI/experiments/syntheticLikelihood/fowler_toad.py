from functools import partial

import jax.numpy as jnp
import jax.random
import levy_stable_jax


@partial(jax.jit, static_argnums=(1, 2))
def simulation(OP_key: jax.Array, num_days: int, num_tods: int, params: jnp.ndarray):
    """
    Simulate from the Toad's stochastic displacement model
    """
    alpha, gamma, p0 = params
    p0 = jax.lax.max(1e-4, p0)
    overnight_displacements = levy_stable_jax.rvs(
        alpha=alpha,
        beta=0.0,
        scale=gamma,
        loc=0.0,
        shape=(num_tods, num_days),
        prng=OP_key
    )
    return_bool = jax.random.bernoulli(OP_key, p=1 - p0, shape=(num_tods, num_days))
    keys = jax.random.split(OP_key, num_tods)

    @jax.vmap
    def to_be_vmapped_along_tods(key: jax.Array, overnight_displacements: jnp.ndarray, return_bool: bool):
        Y = jnp.zeros(num_days)
        Y = Y.at[0].set(overnight_displacements.at[0].get())

        def fun_loop(day_idx, _y):
            _y = jax.lax.cond(return_bool.at[day_idx - 1].get(), lambda _: _y.at[day_idx].set(
                _y.at[jax.random.randint(key, (1,), 0, day_idx - 1).at[0].get()].get()),
                              lambda _: _y.at[day_idx].set(
                                  _y.at[day_idx - 1].get() + overnight_displacements.at[day_idx - 1].get()), None)
            return _y

        Y = jax.lax.fori_loop(1, num_days, lambda day_idx, _y: fun_loop(day_idx, _y), Y)
        return Y

    res = to_be_vmapped_along_tods(keys, overnight_displacements, return_bool)
    return res


@jax.jit
def summary_statistic(Y: jnp.ndarray):
    qs = jnp.linspace(0, 1, 11)

    def compute_statistic_for_one_lag(lag: int):
        displacements = jnp.abs(jnp.diff(Y, lag))
        displacements = jnp.where(displacements < 10, 0, displacements)
        return_counts = (displacements == 0).sum()
        proportion_of_zero = return_counts / (displacements.shape[1] * displacements.shape[0])
        quantile_displacements = jax.vmap(lambda q: jnp.quantile(displacements, q))(
            (1 - proportion_of_zero) * qs + proportion_of_zero)
        logdiffquantiles = jnp.log(jnp.diff(quantile_displacements)).reshape(
            -1)
        median = jnp.quantile(displacements, 0.5 * (1 - proportion_of_zero) + proportion_of_zero)
        statistic = jnp.concatenate([logdiffquantiles, median.reshape(1, ), return_counts.reshape(1, )])
        return statistic

    ss = jnp.concatenate([compute_statistic_for_one_lag(1),
                          compute_statistic_for_one_lag(2),
                          compute_statistic_for_one_lag(4),
                          compute_statistic_for_one_lag(8)])
    return ss


def log_prior(params: jnp.ndarray):
    """
    Uniform prior on [1, 2] x [0, 100] x [0, 0.9]
    """
    alpha, gamma, p0 = params
    return jnp.select([alpha < 1, alpha > 2, gamma < 0, gamma > 100, p0 < 0, p0 > 0.9],
                      [-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf],
                      default=-jnp.log(100 * 0.9))


def shrinkage_cov(ss: jnp.ndarray, shrinkage: float):
    C = jnp.corrcoef(ss, rowvar=False)
    C = jnp.where(C == jnp.nan, 1.0, C)
    sqD = jnp.diag(jnp.var(ss, axis=0) ** 0.5)
    return sqD @ (shrinkage * C + (1 - shrinkage) * jnp.eye(C.shape[0])) @ sqD


def get_tgt_density(sdata: jnp.ndarray, n_samples: int, shrinkage=0.1, transform=False, scales2=None):
    """
    Given params, simulate and compute the summary statistic, compute the log-likelihood estimate and log-posterior function
    """
    num_days, num_tods = sdata.shape
    ssdata = summary_statistic(sdata.reshape((num_tods, num_days)))

    if not transform:
        _simulation = simulation
        _log_prior = log_prior
    elif transform:
        offsets = jnp.array([1., 0., 0.])
        scales = jnp.array([1., 100., 0.9])

        def from_params_transform_to_params(params_transform: jnp.ndarray) -> jnp.ndarray:
            """
            Mapping the transformed parameters to the real paramter: a + expit(x*b)
            """
            params = jax.scipy.special.expit(params_transform * scales2)
            params = offsets + scales * params
            return params

        def log_jac_inverse_transform(params_transform: jnp.ndarray) -> jnp.ndarray:
            """
            Implementing the log. of the jacobian of the previous inverse mapping
            """

            def log_d(x):
                """
                D[1 / (1 + exp(-x))] = exp(-x) / (1 + exp(-x))^2
                """
                return -x * scales2 - 2 * jnp.log1p(jnp.exp(-x * scales2))

            return jnp.log(scales) + jnp.log(scales2) + log_d(params_transform)

        def _simulation(keys: jax.Array, num_days: int, num_tods: int, params_transform: jnp.ndarray) -> jnp.ndarray:
            """
            Given the transformed paramaters, simulate from the model with the corresponding params.
            """
            params = from_params_transform_to_params(params_transform)
            return simulation(keys, num_days, num_tods, params)

        def _log_prior(params_transform: jnp.ndarray) -> jnp.ndarray:
            """
            log_prior of the transformed parameters + log_jac_inverse_transform
            """
            params = from_params_transform_to_params(params_transform)
            return log_prior(params) + log_jac_inverse_transform(params_transform).sum()

    vmapped_simulation = lambda keys, params: jax.vmap(_simulation, in_axes=(0, None, None, None))(keys, num_days,
                                                                                                   num_tods, params)

    @partial(jax.vmap, in_axes=(None, 0))
    def sample_and_compute_posterior(OP_key: jax.Array, params: jnp.ndarray):
        """
        Samples trajectories, compute the statistics, evaluate the estimated Gaussian likelihood and posterior
        """
        keys = jax.random.split(OP_key, n_samples)
        Ys = vmapped_simulation(keys, params)
        ss = jax.vmap(summary_statistic)(Ys)
        ss = jnp.where(ss == jnp.nan, 0., ss)
        estimated_mean = ss.mean(axis=0)
        estimated_cov = shrinkage_cov(ss, shrinkage)
        ll_estimate = - 0.5 * (ssdata - estimated_mean).T @ jnp.linalg.pinv(estimated_cov) @ (
                ssdata - estimated_mean) - 0.5 * jnp.linalg.slogdet(estimated_cov)[1]
        posterior = _log_prior(params) + ll_estimate
        posterior = jnp.where(posterior == jnp.nan, 0., posterior)
        return posterior

    return sample_and_compute_posterior
