import pickle

import numpy as np
import pymc as pm

from experiments.logisticRegression.utils import get_dataset

OUTPUT_PATH = "./output_mfg"


def experiment(n_iter, n_samples=None):
    flipped_predictors, response = get_dataset(dataset="Pima", flip=False)
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
        pm.find_MAP()
        callback = pm.variational.callbacks.CheckParametersConvergence(diff='absolute')
        start_means = {'beta': np.zeros(dim)}
        start_sigma = {'beta': np.ones(dim)}
        approx = pm.fit(n=n_iter, callbacks=[callback], obj_n_mc=n_samples, start=start_means,
                        start_sigma=start_sigma)  # obj_n_mc=n_samples
        with open(f"{OUTPUT_PATH}/res_mfg_advi_{n_iter}.pkl", "wb") as f:
            pickle.dump(
                {'desc': "PIMA dataset, mean field Gaussian ADVI", 'mean': approx.mean.eval(), 'cov': approx.cov.eval(),
                 'loss': approx.hist, 'means': approx.means, 'covs': np.diagonal(approx.covs, axis1=1, axis2=2)}, f)


if __name__ == "__main__":
    N_iters = [2 * 10e3]
    n_samples = None
    for n_iter in N_iters:
        print(n_iter)
        experiment(int(n_iter), n_samples)
