import logging

import jax.random
from timeit_decorator import timeit

from experiments.bayesianSelectionVariable.concrete.Bernoulli_vs_ConcreteStrengthDataset import \
    experiment as lsvi_concrete
from experiments.bayesianSelectionVariable.concrete.concrete import experiment as smc_concrete

logging.basicConfig(level=logging.INFO)
jax.config.update("jax_enable_x64", True)
OUTPUT_PATH = "./output_timeit"


@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_smc_concrete():
    N = 10 ** 5
    P = 1_000
    nruns = 3
    smc_concrete(N, P, nruns, OUTPUT_PATH=OUTPUT_PATH)


@timeit(runs=5, log_level=logging.INFO, detailed=True)
def time_lsvi_concrete():
    N_repeat = 3  # number of repetitions
    # LSVI parameter
    lr = 0.5
    n_samples = 5 * 10 ** 4
    n_iter = 25
    lsvi_concrete(n_samples, n_iter, lr, N_repeat, OUTPUT_PATH=OUTPUT_PATH)


if __name__ == "__main__":
    time_smc_concrete()
    time_lsvi_concrete()
