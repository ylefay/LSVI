import numpy as np
import pandas as pd


def load(path="./"):
    radio_pd = pd.read_csv(f"{path}/radio.csv")
    radio_pd = radio_pd.loc[:, ('Toad', 'x', 'Day')]
    Y = np.zeros((66, 63))
    for i in range(1, 67):
        for j in range(1, 64):
            try:
                Y[i - 1, j - 1] = radio_pd.loc[(radio_pd['Toad'] == i) & (radio_pd['Day'] == j), 'x'].values[0]
            except:
                Y[i - 1, j - 1] = np.nan
    np.savetxt(f"{path}/radio_matrix.csv", Y, delimiter=",")
    return Y


if __name__ == "__main__":
    load()
