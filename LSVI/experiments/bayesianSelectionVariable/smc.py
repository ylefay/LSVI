import particles
import sklearn.linear_model as lin
from particles import binary_smc as bin
from particles import distributions as dists
from particles import smc_samplers as ssps


def run_smc(preds, response, N=10 ** 5, P=1_000, nruns=3):
    M = N // P
    data = (preds, response)
    npreds = preds.shape[1]
    # compare with full regression
    reg = lin.LinearRegression(fit_intercept=False)
    reg.fit(preds, response)

    prior = dists.IID(bin.Bernoulli(0.5), npreds)
    model = bin.BayesianVS(data=data, prior=prior)

    move = ssps.MCMCSequenceWF(mcmc=bin.BinaryMetropolis(), len_chain=P)
    fk = ssps.AdaptiveTempering(model, len_chain=P, move=move)
    results = particles.multiSMC(fk=fk, N=M, verbose=True, nruns=nruns, nprocs=0)
    return results
