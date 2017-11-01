import numpy as np

from cachpy import cachpy


def get_eigen_values_vectors(covariance_matrix, alpha):
    path = 'pca/pickles/eigen_values_vectors_' + str(alpha) + '.pickle'

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
    del centered_matrix

    eigen_values, eigen_vectors = get_eigen_values_vectors(covariance_matrix, alpha)
    print('calculated eigen values and vectors')
    del covariance_matrix

    # Get eigen-(value, vector) pairs, sorted in desc order of values
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
    eigen_pairs.sort()
    eigen_pairs.reverse()
    print('calculated pairs')

    total_eigen_values_sum = sum(eigen_values)
    eig_vals_cum_sums = np.cumsum(sorted(eigen_values, reverse=True))
    del eigen_values, eigen_vectors
    print('calculated sums')

    new_bases = []

    for idx, val in enumerate(eig_vals_cum_sums):
        if (val / total_eigen_values_sum) < alpha:
            # noinspection PySimplifyBooleanCheck
            if new_bases == []:
                new_bases = eigen_pairs[idx][1]
            new_bases = np.hstack((new_bases, eigen_pairs[idx][1]))
        else:
            break

    del eigen_pairs

    new_bases = np.asmatrix(new_bases)
    print('calculated new bases')
    # print(new_bases)

    return new_bases
