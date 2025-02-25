import logging

import jax.numpy as jnp
import jax.random
from timeit_decorator import timeit

from experiments.logisticRegression.sonar_logistic_regression.gaussian_Nicolas import experiment as gaussian_Nicolas
from experiments.logisticRegression.sonar_logistic_regression.heuristic_gaussianMeanField_Nicolas import experiment as heuristic_gaussianMeanField_Nicolas
from experiments.logisticRegression.sonar_logistic_regression.ADVI.gaussian_ADVI import experiment as gaussian_ADVI
from experiments.logisticRegression.sonar_logistic_regression.ADVI.gaussianMeanField_ADVI import experiment as gaussianMeanField_ADVI
from experiments.logisticRegression.sonar_logistic_regression.gaussianMeanField_Nicolas import experiment as gaussianMeanField_Nicolas
from experiments.logisticRegression.sonar_logistic_regression.heuristic_gaussian_Nicolas import experiment as heuristic_gaussian_Nicolas
from experiments.logisticRegression.sonar_logistic_regression.ngd import experiment as ngd
from experiments.logisticRegression.sonar_logistic_regression.ngd_diagonal import experiment as ngd_diagonal

logging.basicConfig(level=logging.INFO)
jax.config.update("jax_enable_x64", True)
OUTPUT_PATH = "./output_timeit"


@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_gaussian_Nicolas_seq2():
    n_iter = 1e2
    Seq_title = 'Seq2'
    interval = jnp.arange(1, n_iter + 1)
    Seq = 1 / interval
    n_samples = 1e5
    OP_key = jax.random.PRNGKey(0)
    gaussian_Nicolas(n_samples=int(n_samples), n_iter=int(n_iter), lr_schedule=Seq, title_seq=Seq_title,
               OP_key=OP_key, OUTPUT_PATH=OUTPUT_PATH)


@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_heuristic_gaussian_Nicolas_u10():
    n_iter = 1e2
    Seq_title = 'Seq1_u10'
    Seq = jnp.ones(n_iter)
    target_residual_schedule = jnp.full(n_iter, 10)

    n_samples = 1e5
    OP_key = jax.random.PRNGKey(0)
    heuristic_gaussian_Nicolas(n_samples=int(n_samples), n_iter=int(n_iter), lr_schedule=Seq,
               target_residual_schedule=target_residual_schedule, title_seq=Seq_title,
               OP_key=OP_key, OUTPUT_PATH=OUTPUT_PATH)

@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_heuristic_gaussian_Nicolas_u1():
    n_iter = 1e2
    Seq_title = 'Seq1_u1'
    Seq = jnp.ones(n_iter)
    target_residual_schedule = jnp.full(n_iter, 1)

    n_samples = 1e5
    OP_key = jax.random.PRNGKey(0)
    heuristic_gaussian_Nicolas(n_samples=int(n_samples), n_iter=int(n_iter), lr_schedule=Seq,
               target_residual_schedule=target_residual_schedule, title_seq=Seq_title,
               OP_key=OP_key, OUTPUT_PATH=OUTPUT_PATH)

@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_heuristic_gaussianMeanField_Nicolas_u10():
    n_iter = 1e2
    Seq_title = 'Seq1_u10'
    Seq = jnp.ones(n_iter)
    target_residual_schedule = jnp.full(n_iter, 10)

    n_samples = 1e4
    OP_key = jax.random.PRNGKey(0)
    heuristic_gaussianMeanField_Nicolas(n_samples=int(n_samples), n_iter=int(n_iter), lr_schedule=Seq,
               target_residual_schedule=target_residual_schedule, title_seq=Seq_title,
               OP_key=OP_key, OUTPUT_PATH=OUTPUT_PATH)

@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_heuristic_gaussianMeanField_Nicolas_u1():
    n_iter = 1e2
    Seq_title = 'Seq1_u1'
    Seq = jnp.ones(n_iter)
    target_residual_schedule = jnp.full(n_iter, 1)

    n_samples = 1e4
    OP_key = jax.random.PRNGKey(0)
    heuristic_gaussianMeanField_Nicolas(n_samples=int(n_samples), n_iter=int(n_iter), lr_schedule=Seq,
               target_residual_schedule=target_residual_schedule, title_seq=Seq_title,
               OP_key=OP_key, OUTPUT_PATH=OUTPUT_PATH)


@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_gaussianMeanField_Nicolas_seq2():
    n_iter = 1e2
    Seq_title = 'Seq2'
    interval = jnp.arange(1, n_iter + 1)
    Seq = 1 / interval
    n_samples = 1e4
    OP_key = jax.random.PRNGKey(0)
    gaussianMeanField_Nicolas(n_samples=int(n_samples), n_iter=int(n_iter), lr_schedule=Seq, title_seq=Seq_title,
               OP_key=OP_key, OUTPUT_PATH=OUTPUT_PATH)



@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_gaussianMeanField_ADVI():
    n_iter = 1e5
    n_samples = None
    gaussianMeanField_ADVI(int(n_iter), n_samples,  OUTPUT_PATH=OUTPUT_PATH)


@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_gaussian_ADVI():
    n_iter = 1e4
    n_samples = None
    gaussian_ADVI(int(n_iter), n_samples, OUTPUT_PATH=OUTPUT_PATH)


@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_gaussian_ngd():
    n_iter = 1e2
    n_samples = 1e4
    lr = 1 / jnp.arange(1, n_iter + 1)
    OP_key = jax.random.PRNGKey(0)
    ngd(OP_key, n_iter, n_samples, lr, OUTPUT_PATH)


@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_gaussian_ngd_diagonal():
    n_iter = 1e3
    n_samples = 1e4
    lr = 1 / jnp.arange(1, n_iter + 1)
    OP_key = jax.random.PRNGKey(0)
    ngd_diagonal(OP_key, n_iter, n_samples, lr, OUTPUT_PATH)

if __name__ == "__main__":
    #time_gaussian_Nicolas_seq2()
    #time_heuristic_gaussian_Nicolas_u10()
    #time_heuristic_gaussian_Nicolas_u1()
    #time_heuristic_gaussianMeanField_Nicolas_u10()
    #time_heuristic_gaussianMeanField_Nicolas_u1()
    #time_gaussianMeanField_Nicolas_seq2()
    #time_gaussianMeanField_ADVI()
    #time_gaussian_ADVI()
    time_gaussian_ngd_diagonal()
    time_gaussian_ngd()
    