import jax
import jax.numpy as jnp


def make_get_batch(flipped_predictors, P):
    """
    Returns a function that given a key, select randomly P rows from the flipped_predictors matrix
    """
    N = flipped_predictors.shape[0]

    def get_batch(key):
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, jnp.arange(N), shape=(P,), replace=False)
        return flipped_predictors.at[idx].get()

    return get_batch


def get_tgt_log_density(P, flipped_predictors, my_prior_log_density):
    """
    Define the log target density of the posterior distribution of the logistic regression model,
    assuming a Gaussian prior, with sub-sampling procedure
    """
    get_batch = make_get_batch(flipped_predictors, P)

    def tgt_log_density(key, beta):
        _flipped_predictors = get_batch(key)
        logcdf = - jnp.log1p(jnp.exp(-_flipped_predictors @ beta.T))
        logcdf = jnp.nan_to_num(logcdf, False, nan=0.0, posinf=0.0, neginf=0.0)
        log_likelihood = jnp.sum(logcdf, axis=-1)
        return log_likelihood + my_prior_log_density(beta)

    return tgt_log_density
