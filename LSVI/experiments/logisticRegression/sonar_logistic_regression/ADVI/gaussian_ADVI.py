import pickle

import numpy as np
import pymc as pm

from experiments.logisticRegression.utils import get_dataset

OUTPUT_PATH = "./output"


def experiment(n_repeat, n_iter, n_samples=None, OUTPUT_PATH="./output"):
    flipped_predictors, response = get_dataset(dataset="Sonar", flip=False)
    dim = flipped_predictors.shape[1]
    response += 1
    response /= 2

    list_mean = np.zeros((n_repeat, dim))
    list_cov = np.zeros((n_repeat, dim, dim))
    list_hist = np.zeros((n_repeat, n_iter))
    list_means = np.zeros((n_repeat, n_iter, dim))
    list_covs = np.zeros((n_repeat, n_iter, dim, dim))

    def _experiment():
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
            return approx.mean.eval(), approx.cov.eval(), approx.hist, approx.means, approx.covs

    # Looping over the desired number of repetitions, sequentially, and storing the results.
    for _ in range(n_repeat):
        mean, cov, hist, means, covs = _experiment()
        list_mean[_] = mean
        list_cov[_] = cov
        list_hist[_] = hist
        list_means[_] = means
        list_covs[_] = covs
    # writing the PKL.
    with open(f"{OUTPUT_PATH}/res_advi_{n_iter}.pkl", "wb") as f:
        pickle.dump(
            {'desc': "Sonar dataset, full rank Gaussian ADVI", 'mean': list_mean, 'cov': list_cov,
             'loss': list_hist, 'means': list_means, 'covs': list_covs}, f)


if __name__ == "__main__":
    N_iters = [1e2]
    n_samples = None
    n_repetitions = 100
    for n_iter in N_iters:
        print(n_iter)
        experiment(n_repetitions, int(n_iter), n_samples)
