import os
import pickle

import numpy as np
import pandas as pd

from experiments.bayesianSelectionVariable.utils import get_tgt_log_density
from variational.exponential_family import GenericBernoulliDistributionNumpy as GenericBernoulliDistribution
from variational.numpy_lsvi import lsvi

init_smc = np.array([9.02903333e-01, 5.17910000e-01, 8.40920000e-01, 8.91340000e-01,
                     1.46850000e-01, 5.03170000e-01, 5.12710000e-01, 4.67800000e-02,
                     4.89126667e-01, 8.62066667e-02, 4.80730000e-01, 9.87990000e-01,
                     4.99603333e-01, 1.08910000e-01, 0.00000000e+00, 2.97333333e-03,
                     2.44700000e-02, 3.06000000e-02, 3.83333333e-04, 6.27666667e-03,
                     4.80233333e-02, 1.13000000e-03, 5.13633333e-02, 8.98456667e-01,
                     5.04826667e-01, 8.36420000e-01, 1.90753333e-01, 7.41333333e-02,
                     4.21833333e-02, 9.04316667e-01, 1.00000000e+00, 1.53400000e-02,
                     9.53666667e-03, 6.85250000e-01, 1.66666667e-05, 1.33000000e-03,
                     4.00000000e-05, 8.00000000e-05, 1.55000000e-03, 1.75166667e-02,
                     1.00000000e+00, 1.53133333e-02, 3.13533333e-01, 3.29400000e-02,
                     9.11800000e-02, 1.25143333e-01, 3.57723333e-01, 9.94293333e-01,
                     2.12900000e-02, 1.90000000e-04, 3.34666667e-03, 1.03266667e-02,
                     2.66666667e-04, 9.50166667e-02, 6.26666667e-02, 1.52333333e-03,
                     6.66666667e-06, 2.34700000e-02, 1.95570000e-01, 3.07960000e-01,
                     8.36750000e-01, 8.15833333e-01, 1.28093333e-01, 4.14390000e-01,
                     9.88400000e-01, 3.63000000e-03, 3.33700000e-01, 3.81633333e-02,
                     6.33333333e-05, 8.57303333e-01, 8.56660000e-01, 3.55833333e-02,
                     3.12333333e-03, 5.90800000e-02, 8.03333333e-04, 2.00600000e-02,
                     5.60166667e-02, 1.00000000e+00, 1.00000000e+00, 3.87666667e-03,
                     1.00000000e+00, 1.00000000e+00, 7.03013333e-01, 5.52066667e-02,
                     9.52780000e-01, 2.81366667e-02, 2.58000000e-03, 3.95840000e-01,
                     5.04333333e-03, 4.73873333e-01, 4.36000000e-03, 1.00000000e+00])

"""
    The target distribution is the conjugated posterior of the Hierarchical Bayesian model for Bayesian selection variable problem
    as described in Schäfer, Chopin 2013.
    The dataset is the Concrete Strength dataset.
    The variational family is the set of products of Binomial distributions.
"""


def experiment(n_samples, n_iter, lr, N_repeat=3, OUTPUT_PATH="./output"):
    prefix = "Ber_Concrete_"
    suffix = "_epsilon1em4"
    lr_schedule = np.full(n_iter, lr)
    title = lambda _: f"{prefix}_{_}_{n_samples}_{lr}_{n_iter}{suffix}.pkl"
    epsilon = 1e-4  # Ensures the probabilities stay in [epsilon, 1-epsilon] for numerical stability reasons.

    desc = "Hierarchical Bayesian linear regression, Ber variational family, Concrete Strength dataset."
    data = pd.read_csv("./concrete_from_particles.csv", header=None)
    obs = np.array(data.iloc[:, 0].to_numpy())
    reg = np.array(data.iloc[:, 1:].to_numpy())
    dim = reg.shape[1]

    tgt_log_density = get_tgt_log_density(reg, obs)

    init_ps = np.ones(dim) * 0.5
    init_ps[init_ps <= epsilon] = epsilon
    init_ps[init_ps >= 1 - epsilon] = 1 - epsilon

    print(f"init: {init_ps}")
    my_generic_distribution = GenericBernoulliDistribution(dimension=dim)
    upsilon_init = my_generic_distribution.get_upsilon(init_ps)
    sufficient_statistic = my_generic_distribution.sufficient_statistic_numpy
    sampling = lambda theta, n: my_generic_distribution.sampling_method_numpy(theta, n, eps=1e-3)
    for _ in range(N_repeat):
        if os.path.exists(f"{OUTPUT_PATH}/{title(_)}"):
            with open(f"{OUTPUT_PATH}/{title(_)}", "rb") as f:
                my_pickle = pickle.load(f)
                res, res_all = my_pickle['res'], my_pickle['all']
        else:
            res, res_all = lsvi(sampling, sufficient_statistic,
                                tgt_log_density,
                                upsilon_init, n_iter, n_samples, lr_schedule=lr_schedule, return_all=False)
            PARAMS = {'n_iter': n_iter, 'n_samples': n_samples, 'lr': lr}

            with open(f"{OUTPUT_PATH}/{title(_)}",
                      "wb") as f:
                pickle.dump({'desc': desc, 'PARAMS': PARAMS, 'res': res, 'all': res_all}, f)


if __name__ == "__main__":
    OUTPUT_PATH = "./output"  # "output_qmc"
    N_repeat = 100  # number of repetitions
    # LSVI parameter
    lr = 0.5
    n_samples = 5 * 10 ** 4
    n_iter = 25
    experiment(n_samples, n_iter, lr, N_repeat, OUTPUT_PATH=OUTPUT_PATH)
