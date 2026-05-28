from particles.binary_smc import BayesianVS


def get_tgt_log_density(reg, obs):
    """
    Define the log target density of the posterior distribution
    derived in Schäfer, Chopin 2013.
    """
    data = (reg, obs)
    vs = BayesianVS(data)

    # def loglik(gamma): # do not loop over the first dim. of gamma, gamma is a 1D array
    # gamma = gamma[np.newaxis, :]
    # return vs.loglik(gamma)[0]

    # return loglik
    return vs.loglik
