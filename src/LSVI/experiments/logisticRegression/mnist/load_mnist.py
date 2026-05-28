import numpy as np
import pandas as pd


def mnist_dataset(return_test=False, flip=True, path_prefix="."):
    def preprocess(data):
        idx = (data[:, 0] == 0) + (data[:, 0] == 8)
        data = data[idx]
        idx = data[:, 0] == 8
        data[idx, 0] = 1
        data[~idx, 0] = -1
        labels = data[:, 0]
        data = data[:, 1:]
        data = data / 255
        # data = np.delete(data, np.where(np.product(data == 0, axis=0) == 1), axis=-1)
        if not flip:
            return data, labels.astype(dtype=float)
        else:
            data *= labels[:, np.newaxis]
            return data

    mnist_train = np.array(pd.read_csv(f"{path_prefix}/mnist/mnist_train.csv", header=None))
    mnist_train = preprocess(mnist_train)
    if return_test:
        mnist_test = np.array(pd.read_csv(f"{path_prefix}/mnist/mnist_test.csv", header=None))
        mnist_test = preprocess(mnist_test)
        return mnist_train, mnist_test
    else:
        return mnist_train


def mnist_dataset_full(return_test=False, path_prefix="."):
    def preprocess(data):
        labels = data[:, 0]
        data = data[:, 1:]
        data = data / 255
        n_max = np.max(labels) + 1
        onehot = np.zeros((labels.shape[0], n_max))
        onehot[np.arange(labels.shape[0]), labels] = 1
        return data, onehot.astype(dtype=float)

    mnist_train = np.array(pd.read_csv(f"{path_prefix}/mnist/mnist_train.csv", header=None))
    mnist_train = preprocess(mnist_train)
    if return_test:
        mnist_test = np.array(pd.read_csv(f"{path_prefix}/mnist/mnist_test.csv", header=None))
        mnist_test = preprocess(mnist_test)
        return mnist_train, mnist_test
    else:
        return mnist_train
