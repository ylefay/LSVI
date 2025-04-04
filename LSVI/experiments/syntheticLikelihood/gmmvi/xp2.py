import os
import logging
# Tensorflow may give warnings when the Cholesky decomposition fails.
# However, these warning can usually be ignored because the NgBasedOptimizer
# will handle them by rejecting the update and decreasing the stepsize for
# the failing component. To keep the console uncluttered, we suppress warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
import pickle
from gmmvi.optimization.gmmvi import GMMVI
from gmmvi.configs import load_yaml
from gmmvi.experiments.target_distributions.logistic_regression import make_breast_cancer
from gmmvi.models.full_cov_gmm import FullCovGMM
from gmmvi.models.gmm_wrapper import GmmWrapper
from gmmvi.experiments.setup_experiment import construct_initial_mixture
from pathlib import Path

import tensorflow as tf
import numpy as np
from gmmvi.experiments.target_distributions.lnpdf import LNPDF
from math import pi

from gmmvi.models.gmm_wrapper import GmmWrapper
#For creating a GMMVI object using GMMVI.build_from_config, we need:
# 1. A dictionary containing the hyperparameters
my_path = os.path.dirname(os.path.realpath(__file__))
config = load_yaml(os.path.join(my_path, "config.yml"))

# 2. A target distribution

class LogisticRegression(LNPDF):
    """This class is used for implementing the logistic regression experiments based on the BreastCancer and
    GermanCredit dataset :cite:p:`UCI`, reimplementing the experiments used by :cite:t:`Arenz2020`.

    Parameters:
        dataset_id: a string
            Should be either "breast_cancer" or "german_credit"
    """

    def __init__(self):
        super(LogisticRegression, self).__init__(use_log_density_and_grad=False)
        self.const_term = tf.constant(tf.cast(0.5 * tf.math.log(2. * np.pi), dtype=tf.float32))
        data = np.loadtxt("./pima")
        X = data[:, 2:]
        X /= np.std(X, 0)[np.newaxis, :]
        X = np.hstack((np.ones((len(X), 1)), X))
        self.data = tf.cast(X, tf.float32)
        self.labels = data[:, 1]
        self.num_dimensions = self.data.shape[1]
        self._prior_std = tf.constant(10., dtype=tf.float32)
        self.prior_mean = tf.constant(0., dtype=tf.float32)
        self.labels = tf.Variable(tf.expand_dims(self.labels.astype(np.float32), 1))

    def get_num_dimensions(self):
        return self.num_dimensions

    @property
    def prior_std(self):
        return self._prior_std

    def log_likelihood(self, x):
        features = -tf.matmul(self.data, tf.transpose(x))
        log_likelihoods = tf.where(self.labels == 1, tf.transpose(tf.math.log_sigmoid(features)),
                                   tf.transpose(tf.math.log_sigmoid(features) - features))
        return log_likelihoods

    def log_density(self, x):
        features = -tf.matmul(self.data, tf.transpose(x))
        log_likelihoods = tf.reduce_sum(tf.where(self.labels == 1, tf.math.log_sigmoid(features),
                                                 tf.math.log_sigmoid(features) - features), axis=0)
        log_prior = tf.reduce_sum(-tf.math.log(self.prior_std) - self.const_term - 0.5 * tf.math.square(
            (x - self.prior_mean) / self.prior_std), axis=1)
        log_posterior = log_likelihoods + log_prior
        return log_posterior

target_distribution = LogisticRegression()

# 3. An (wrapped) initial model
dims = target_distribution.get_num_dimensions()
initial_weights = tf.ones(1, tf.float32)
initial_means = tf.zeros((1, dims), tf.float32)
initial_covs = tf.reshape(100 * tf.eye(dims), [1, dims, dims])
model = FullCovGMM(initial_weights, initial_means, initial_covs)
# Above config contains a section model_initialization, and, therefore,
# we could also create the initial model using:
# model = construct_initial_mixture(dims, **config["model_initialization"])
wrapped_model = GmmWrapper.build_from_config(model=model, config=config)


# Now we can create the GMMVI object and start optimizing
gmmvi = GMMVI.build_from_config(config=config,
                                target_distribution=target_distribution,
                                model=wrapped_model)
max_iter = 10
repeats = 3
import numpy as np

elbo = np.zeros((repeats, max_iter))
for repeat in range(repeats):
    for n in range(max_iter):
        gmmvi.train_iter()

        samples = gmmvi.model.sample(1000)[0]
        elbo[repeat, n] = tf.reduce_mean(target_distribution.log_density(samples)
                              - model.log_density(samples))

pickle.dump(elbo, open("elbo.pkl", "wb"))

