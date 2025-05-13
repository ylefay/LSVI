import pickle

import jax
import jax.numpy as jnp

from experiments.logisticRegression.utils import get_dataset, get_tgt_log_density
from variational.exponential_family import GenericNormalDistribution, NormalDistribution
from variational.lsvi import lsvi

OUTPUT_PATH = "./output"
OP_key = jax.random.PRNGKey(0)
jax.config.update("jax_enable_x64", True)

if __name__ == "__main__":
    flipped_predictors = get_dataset(dataset="Sonar")
    N, dim = flipped_predictors.shape

    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_log_density = NormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior_log_density)

    # Gaussian Variational Family
    my_variational_family = GenericNormalDistribution(dimension=dim)
    sampling = my_variational_family.sampling_method
    sufficient_statistic = my_variational_family.sufficient_statistic

    upsilon_init = my_variational_family.get_upsilon(jnp.zeros(dim), jnp.identity(dim))

    n_iter = 5
    n_samples = int(1e4)
    lr = None
    lr_schedule = jnp.full(n_iter, 1.0)
    res, res_all = lsvi(OP_key, sampling, sufficient_statistic, tgt_log_density, upsilon_init, n_iter, n_samples,
                        lr_schedule=lr_schedule,
                        return_all=False)
    for idx, k in enumerate(res):
        mean_cov = my_variational_family.get_mean_cov(k[:-1])
        print(f"{idx}: {mean_cov}")

    PARAMS = {'n_iter': n_iter, 'n_samples': n_samples, 'lr': lr_schedule}
    desc = "PIMA dataset, standard initialization, full cov. Gaussian"
    with open(
            f"{OUTPUT_PATH}/gaussian_{n_iter}_{n_samples}_{lr if isinstance(lr, float) else "Seq"}.pkl",
            "wb") as f:
        pickle.dump({'desc': desc, 'PARAMS': PARAMS, 'res': res, 'all': res_all}, f)
