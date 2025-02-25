"""
Second example from Schäfer and Chopin (2013), concrete compressive strength. 
"""

import pickle

import numpy as np
import pandas as pd

from experiments.bayesianSelectionVariable.smc import run_smc


def experiment(N, P, nruns, OUTPUT_PATH="./output_SMC"):
    data = pd.read_csv("./concrete_from_particles.csv", header=None)
    response = np.array(data.iloc[:, 0].to_numpy())
    preds = np.array(data.iloc[:, 1:].to_numpy())
    M = N // P
    results = run_smc(preds, response, N=N, P=P, nruns=nruns)
    pickle.dump(results, open(f"{OUTPUT_PATH}/smc_{N}_{P}_{M}_{nruns}.pkl", "wb"))


if __name__ == "__main__":
    OUTPUT = "output_SMC"
    N = 10 ** 5
    P = 1_000
    nruns = 3
    experiment(N, P, nruns, OUTPUT_PATH=OUTPUT)
