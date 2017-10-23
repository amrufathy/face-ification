import pickle

import numpy as np


def get_eigen_values_vectors(covariance_matrix, alpha):
    from os.path import isfile

    values_path = 'pickles/eigen_values_' + str(alpha) + '.pickle'
    vectors_path = 'pickles/eigen_vectors_' + str(alpha) + '.pickle'

    if isfile(values_path) and isfile(vectors_path):
        with open(values_path, 'rb') as f:
            eigen_values = pickle.load(f)
        with open(vectors_path, 'rb') as f:
            eigen_vectors = pickle.load(f)
    else:
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        with open(values_path, 'wb') as f:
            pickle.dump(obj=eigen_values, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(vectors_path, 'wb') as f:
            pickle.dump(obj=eigen_vectors, file=f, protocol=pickle.HIGHEST_PROTOCOL)

    return eigen_values, eigen_vectors


def pca(data_matrix, alpha):
    mean_vector = np.mean(data_matrix, axis=0).T
    print('calculated mean vector', mean_vector.shape)

    centered_matrix = data_matrix - mean_vector.T
    print('calculated centered matrix', centered_matrix.shape)
    # print(centered_matrix)
    del mean_vector

    covariance_matrix = np.dot(centered_matrix.T, centered_matrix) * (1 / data_matrix.shape[0])
    print('calculated covariance matrix', covariance_matrix.shape)
    # print(covariance_matrix)
    del centered_matrix

    eigen_values, eigen_vectors = get_eigen_values_vectors(covariance_matrix, alpha)
    print('calculated eigen values and vectors')
    # print(eigen_values)
    # print(eigen_vectors)
    del covariance_matrix

    # Get eigen-(value, vector) pairs, sorted in desc order of values
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
    eigen_pairs.sort()
    eigen_pairs.reverse()
    print('calculated pairs')
    # print(eigen_pairs[0:5])

    total_eigen_values_sum = sum(eigen_values)
    eig_vals_cum_sums = np.cumsum(sorted(eigen_values, reverse=True))
    del eigen_values, eigen_vectors
    print('calculated sums')
    # print(total_eigen_values_sum)
    # print(eig_vals_cum_sums[0:10])

    new_bases = []

    for idx, val in enumerate(eig_vals_cum_sums):
        if (val / total_eigen_values_sum) < alpha:
            if new_bases == []:
                new_bases = eigen_pairs[idx][1]
            new_bases = np.hstack((new_bases, eigen_pairs[idx][1]))
        else:
            break

    del eigen_pairs
    # Here we have new bases
    new_bases = np.asmatrix(new_bases)
    print('calculated new bases')
    # print(new_bases)
    # reduced_dimensionality_data = np.dot(data_matrix, new_bases)
    # reduced_dimensionality_data = np.hstack((reduced_dimensionality_data, labels))
    # print('calculated reduced dimensionality data\n')

    return new_bases
