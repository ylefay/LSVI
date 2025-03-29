import jax.numpy as jnp
import jax.random
import numpy as np
import pymc as pm

from experiments.time_wrapper import timer

n_runs = 5

OP_key = jax.random.PRNGKey(0)

jax.config.update("jax_enable_x64", True)

from experiments.logisticRegression.utils import get_dataset, get_tgt_log_density
from variational.exponential_family import GenericNormalDistribution, NormalDistribution
from variational.gaussian_lsvi import gaussian_lsvi

OUTPUT_PATH = "./output"
OP_key = jax.random.PRNGKey(4)
jax.config.update("jax_enable_x64", True)


@timer(runs=n_runs)
def experiment_fc_lsvi(key, n_samples=100000, n_iter=100, lr_schedule=None, target_residual_schedule=None):
    flipped_predictors = get_dataset(dataset="Sonar")
    N, dim = flipped_predictors.shape

    # Gaussian Prior
    my_prior_covariance = 25 * jnp.identity(dim)
    my_prior_covariance = my_prior_covariance.at[0, 0].set(400)
    my_prior_log_density = NormalDistribution(jnp.zeros(dim), my_prior_covariance).log_density
    tgt_log_density = get_tgt_log_density(flipped_predictors, my_prior_log_density)

    # Gaussian Variational Family
    my_variational_family = GenericNormalDistribution(dimension=dim)

    upsilon_init = my_variational_family.get_upsilon(jnp.zeros(dim), jnp.identity(dim))

    def f(key):
        res, res_all = gaussian_lsvi(key, tgt_log_density, upsilon_init, n_iter, n_samples, lr_schedule=lr_schedule,
                                     target_residual_schedule=target_residual_schedule,
                                     return_all=False)
        return res, res_all

    _ = f(key)
    return None


@timer(runs=n_runs)
def experiment(n_iter, n_samples=None):
    flipped_predictors, response = get_dataset(dataset="Sonar", flip=False)
    dim = flipped_predictors.shape[1]
    response += 1
    response /= 2

    with pm.Model() as logistic_model:
        cov = np.identity(dim) * 25
        cov[0, 0] = 400
        beta = pm.MvNormal('beta', mu=np.zeros(dim), cov=cov)
        logit_theta = pm.Deterministic('logit_theta', flipped_predictors @ beta)
        y = pm.Bernoulli("y", logit_p=logit_theta, observed=response)
        with logistic_model:
            logistic_model.debug()
            pm.find_MAP()
            callback = pm.variational.callbacks.CheckParametersConvergence(diff='absolute')
            start_means = {'beta': pm.find_MAP()['beta']}
            start_means = {'beta': np.zeros(dim)}
            start_sigma = {'beta': np.identity(dim)}
            approx = pm.fit(n=n_iter, callbacks=[callback], obj_n_mc=n_samples, method='fullrank_advi',
                            start=start_means)
            approx.mean.eval(), approx.cov.eval(), approx.hist, approx.means, approx.covs
    return None


if __name__ == "__main__":
    """
    Running n_runs time with timeit decorator the experiment mean field lsvi for different n_samples
    """
    n_iter = 100
    Seq = jnp.ones(n_iter)
    target_residual_schedule = jnp.full(n_iter, 10)
    n_samples_arr = [100, 1000, 10000]
    print("FC LSVI (sch. 3)")
    for n_samples in n_samples_arr:
        print(n_samples)
        experiment_fc_lsvi(OP_key, n_samples, n_iter, Seq, target_residual_schedule)

    Seq = 1 / jnp.arange(1, n_iter + 1)
    target_residual_schedule = jnp.inf
    print("FC LSVI (sch. 1)")
    for n_samples in n_samples_arr:
        print(n_samples)
        experiment_fc_lsvi(OP_key, n_samples, n_iter, Seq, target_residual_schedule)

    """
    Doing the same but using pyMC3 default ADVI implementation.
    """
    print("FC ADVI")
    for n_samples in n_samples_arr:
        print(n_samples)
        experiment(n_iter, n_samples)
