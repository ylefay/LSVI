import logging
import os

from gmmvi.experiments.target_distributions.logistic_regression import make_breast_cancer

# Tensorflow may give warnings when the Cholesky decomposition fails.
# However, these warning can usually be ignored because the NgBasedOptimizer
# will handle them by rejecting the update and decreasing the stepsize for
# the failing component. To keep the console uncluttered, we suppress warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import jax.numpy as jnp
import jax
import pickle
from gmmvi.optimization.gmmvi import GMMVI
from gmmvi.configs import load_yaml
from gmmvi.models.full_cov_gmm import FullCovGMM
from gmmvi.models.gmm_wrapper import GmmWrapper
from experiments.syntheticLikelihood.fowler_toad import get_tgt_density

# For creating a GMMVI object using GMMVI.build_from_config, we need:
# 1. A dictionary containing the hyperparameters
my_path = os.path.dirname(os.path.realpath(__file__))
config = load_yaml(os.path.join(my_path, "config.yml"))
OUTPUT_PATH = "../output/"
# 2. A target distribution
sdata = pickle.load(open(f"{OUTPUT_PATH}/ht_data_63_66_[0 0].pkl", "rb"))

scales2 = jnp.array([1., 1., 1.])
dim = 3
target_density = lambda x: get_tgt_density(sdata, 100, shrinkage=0.5, transform=True, scales2=scales2)(jax.random.PRNGKey(0), x)
import numpy as np
import tensorflow as tf

from gmmvi.experiments.target_distributions.lnpdf import LNPDF


class tg(LNPDF):
    def __init__(self, target_density):
        self.target_density = target_density
        super().__init__(use_log_density_and_grad=False, safe_for_tf_graph=False)

    def log_density(self, x):
        return tf.convert_to_tensor(np.asarray(self.target_density(jnp.asarray(x))))

    def return_dims(self):
        return 3
make_breast_cancer()

# 3. An (wrapped) initial model
dims = 3
initial_weights = tf.ones(1, tf.float32)
initial_means = tf.zeros((1, dims), tf.float32)
initial_covs = tf.reshape(0.1 * tf.eye(dims), [1, dims, dims])
model = FullCovGMM(initial_weights, initial_means, initial_covs)
# Above config contains a section model_initialization, and, therefore,
# we could also create the initial model using:
# model = construct_initial_mixture(dims, **config["model_initialization"])
wrapped_model = GmmWrapper.build_from_config(model=model, config=config)

# Now we can create the GMMVI object and start optimizing
gmmvi = GMMVI.build_from_config(config=config,
                                target_distribution=tg(target_density),
                                model=wrapped_model)
max_iter = 50
for n in range(max_iter):
    gmmvi.train_iter()

    if n % 10 == 0:
        samples = gmmvi.model.sample(1000)[0]
        elbo = tf.reduce_mean(tg.log_density(samples)
                              - model.log_density(samples))
        print(f"{n}/{max_iter}: "
              f"The model now has {gmmvi.model.num_components} components "
              f"and an elbo of {elbo}.")
