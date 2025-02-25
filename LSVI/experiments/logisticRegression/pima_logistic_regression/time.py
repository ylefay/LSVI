import logging

import jax.numpy as jnp
import jax.random
from timeit_decorator import timeit

from experiments.logisticRegression.pima_logistic_regression.gaussian import experiment as gaussian
from experiments.logisticRegression.pima_logistic_regression.gaussian_Nicolas import experiment as gaussian_Nicolas
from experiments.logisticRegression.pima_logistic_regression.ADVI.gaussian_ADVI import experiment as gaussian_ADVI
from experiments.logisticRegression.pima_logistic_regression.ngd import experiment as ngd

logging.basicConfig(level=logging.INFO)
jax.config.update("jax_enable_x64", True)
OUTPUT_PATH = "./output_timeit"


@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_gaussian():
    OP_key = jax.random.PRNGKey(0)
    n_iter = 100
    n_samples = int(1e4)
    lr = 1.0
    gaussian(OP_key, n_iter, n_samples, lr, OUTPUT_PATH)


@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_gaussian_Nicolas_seq1():
    n_iter = 100
    Seq_title = 'Seq1'
    Seq = jnp.ones(n_iter)
    n_samples = 1e4
    OP_key = jax.random.PRNGKey(0)
    gaussian_Nicolas(n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq, title_seq=Seq_title,
               OP_key=OP_key, OUTPUT_PATH=OUTPUT_PATH)


@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_gaussian_Nicolas_seq2():
    n_iter = 100
    Seq_title = 'Seq2'
    interval = jnp.arange(1, n_iter + 1)
    Seq = 1 / interval
    n_samples = 1e4
    OP_key = jax.random.PRNGKey(0)
    gaussian_Nicolas(n_samples=int(n_samples), n_iter=n_iter, lr_schedule=Seq, title_seq=Seq_title,
               OP_key=OP_key, OUTPUT_PATH=OUTPUT_PATH)

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


if __name__ == "__main__":
    """time_gaussian()
    time_gaussian_Nicolas_seq1()
    time_gaussian_Nicolas_seq2()
    time_gaussian_ADVI()"""
    time_gaussian_ngd()