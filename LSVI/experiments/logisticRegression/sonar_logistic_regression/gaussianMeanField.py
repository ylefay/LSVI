import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.utils import get_dataset, get_tgt_log_density
from variational.exponential_family import GenericMeanFieldNormalDistribution, MeanFieldNormalDistribution
from variational.laplace import laplace_approximation
from variational.lsvi import lsvi

OUTPUT_PATH = "./output_mean_field"
OP_key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    flipped_predictors = get_dataset(dataset="Sonar")
    N, dim = flipped_predictors.shape

    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_covariance = jnp.diag(my_prior_covariance)
    my_prior_log_density = MeanFieldNormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior_log_density)

    # Mean Field Gaussian Variational Family
    my_variational_family = GenericMeanFieldNormalDistribution(dimension=dim)
    sampling = my_variational_family.sampling_method
    sufficient_statistic = my_variational_family.sufficient_statistic

    # Laplace Approximation for the initialisation
    _, laplace_mean, laplace_cov = laplace_approximation(tgt_log_density, jnp.zeros(dim))
    laplace_cov = 1 / jnp.diag(jnp.linalg.inv(laplace_cov))
    upsilon_init = my_variational_family.get_upsilon(laplace_mean, laplace_cov)
    # upsilon_init = my_variational_family.get_upsilon(jnp.zeros(dim), jnp.ones(dim))

    n_iter = 100
    n_samples = int(1e5)
    lr_schedule = jnp.full(n_iter, 1.0)


    def sanity(upsilon):
        mean, cov = my_variational_family.get_mean_cov(upsilon.at[:-1].get())
        res = jnp.any(cov <= 0)
        return res


    res, res_all = lsvi(OP_key, sampling, sufficient_statistic, tgt_log_density, upsilon_init, n_iter, n_samples,
                        lr_schedule=lr_schedule, return_all=False, sanity=sanity)

    PARAMS = {'n_iter': n_iter, 'n_samples': n_samples, 'lr': lr_schedule}
    desc = "PIMA dataset, standard initialization, mean field Gaussian"
    with open(
            f"{OUTPUT_PATH}/MeanFieldGaussian_{n_iter}_{n_samples}_{lr if isinstance(lr, float) else "Seq"}.pkl",
            "wb") as f:
        pickle.dump({'desc': desc, 'PARAMS': PARAMS, 'res': res, 'all': res_all}, f)
