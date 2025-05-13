import logging

import jax.numpy as jnp
import jax.random
from timeit_decorator import timeit

from experiments.logisticRegression.subsampling_review.heuristic_and_non_heuristic_gaussianMeanField_NIcolas import \
    experiment as heuristic_gaussianMeanField_Nicolas
from experiments.logisticRegression.subsampling_review.ngd import experiment as ngd

logging.basicConfig(level=logging.INFO)
jax.config.update("jax_enable_x64", True)
OUTPUT_PATH = "./output_timeit"

n_runs = 5
n_repetitions = 1
OP_key = jax.random.PRNGKey(0)
keys = jax.random.split(OP_key, n_repetitions)


@timeit(runs=n_runs, log_level=logging.INFO, detailed=True)
def time_heuristic_gaussianMeanField_Nicolas_sch5():
    n_iter = int(1e4)
    Seq_title = 'sch5'
    Seq = jnp.ones(n_iter) * 1e-3
    target_residual_schedule = jnp.inf
    n_samples = 1e4
    heuristic_gaussianMeanField_Nicolas(keys, n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq,
                                        target_residual_schedule=target_residual_schedule, title_seq=Seq_title,
                                        OUTPUT_PATH=OUTPUT_PATH)


@timeit(runs=n_runs, log_level=logging.INFO, detailed=True)
def time_heuristic_gaussianMeanField_Nicolas_sch6():
    n_iter = int(1e4)
    Seq_title = 'sch6'
    Seq = jnp.ones(n_iter) * 1e-3
    target_residual_schedule = jnp.full(n_iter, 10)
    n_samples = 1e4
    heuristic_gaussianMeanField_Nicolas(keys, n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq,
                                        target_residual_schedule=target_residual_schedule, title_seq=Seq_title,
                                        OUTPUT_PATH=OUTPUT_PATH)


@timeit(runs=n_runs, log_level=logging.INFO, detailed=True)
def time_ngd_diagonal():
    Seq_title = 'sch5'
    n_iter = int(1e4)
    n_samples = int(1e4)
    lr = 1e-3
    ngd(keys, n_iter, n_samples, lr, OUTPUT_PATH=OUTPUT_PATH)


if __name__ == "__main__":
    time_ngd_diagonal()
    time_heuristic_gaussianMeanField_Nicolas_sch5()
    time_heuristic_gaussianMeanField_Nicolas_sch6()
