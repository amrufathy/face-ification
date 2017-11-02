import numpy as np

from cachpy import cachpy


def get_eigen_values_vectors(covariance_matrix, alpha):
    path = 'pickles/pca/eigen_values_vectors_' + str(alpha) + '.pickle'

    @cachpy(path)
    def get_eigen_values_vectors_for_alpha(covariance_matrix):
        return np.linalg.eigh(covariance_matrix)

    return get_eigen_values_vectors_for_alpha(covariance_matrix)


def pca(data_matrix, alpha):
    mean_vector = np.mean(data_matrix, axis=0).T
    print('calculated mean vector', mean_vector.shape)

    centered_matrix = data_matrix - mean_vector.T
    print('calculated centered matrix', centered_matrix.shape)
    del mean_vector

    covariance_matrix = np.dot(centered_matrix.T, centered_matrix) * (1 / data_matrix.shape[0])
    print('calculated covariance matrix', covariance_matrix.shape)
    del data_matrix, centered_matrix

    eigen_values, eigen_vectors = get_eigen_values_vectors(covariance_matrix, alpha)
    print('calculated eigen values and vectors')
    del covariance_matrix

    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    print('sorted eigen values and vectors')

    r = 1
    while (sum(eigen_values[:r]) / sum(eigen_values)) < alpha:
        r += 1

    print('calculated r')

    return np.asmatrix(eigen_vectors[:, :r])
