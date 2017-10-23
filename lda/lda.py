import pickle

import numpy as np


# noinspection PyPep8Naming
def get_Sb_matrix(mean_vectors, overall_mean, classes_matrices):
    from os.path import isfile

    file_path = 'lda/pickles/sb_matrix.pickle'
    if isfile(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        S_b = 0
        for idx, m_v in enumerate(mean_vectors):
            diff = m_v - overall_mean
            outer = diff.dot(diff.T)
            S_b += (classes_matrices[idx].shape[0] * outer)
        S_b = np.asmatrix(S_b)

        with open(file_path, 'wb') as f:
            pickle.dump(obj=S_b, file=f, protocol=pickle.HIGHEST_PROTOCOL)

        return S_b


# noinspection PyPep8Naming
def get_S_matrix(centered_class_matrices):
    from os.path import isfile

    file_path = 'lda/pickles/s_matrix.pickle'
    if isfile(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        S = 0
        for ccm in centered_class_matrices:
            S += ccm.T.dot(ccm)

        with open(file_path, 'wb') as f:
            pickle.dump(obj=S, file=f, protocol=pickle.HIGHEST_PROTOCOL)

        return S


# noinspection PyPep8Naming
def get_S_inv_Sb_matrix(S, S_b):
    from os.path import isfile

    file_path = 'lda/pickles/s_inv_sb_matrix.pickle'
    if isfile(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        S_inv_Sb = np.linalg.inv(S).dot(S_b)

        with open(file_path, 'wb') as f:
            pickle.dump(obj=S_inv_Sb, file=f, protocol=pickle.HIGHEST_PROTOCOL)

        return S_inv_Sb


def get_eigen_values_vectors(covariance_matrix):
    from os.path import isfile

    values_path = 'lda/pickles/eigen_values.pickle'
    vectors_path = 'lda/pickles/eigen_vectors.pickle'

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


# noinspection PyPep8Naming
def lda(data_matrix, classes_matrices):
    mean_vectors = [np.mean(class_matrix[:, :-1], axis=0).T for class_matrix in classes_matrices]
    overall_mean = np.mean(data_matrix[:, :-1], axis=0).T
    print('calculated means')

    S_b = get_Sb_matrix(mean_vectors, overall_mean, classes_matrices)
    print('calculated Sb matrix')

    centered_class_matrices = [(class_matrix[:, :-1] - mean_vectors[i].T) for i, class_matrix in
                               enumerate(classes_matrices)]
    print('calculated centered matrices')
    del mean_vectors, overall_mean

    S = get_S_matrix(centered_class_matrices)
    print('calculated S matrix')
    del centered_class_matrices

    S_inv_Sb = get_S_inv_Sb_matrix(S, S_b)
    print('calculated S_inv_Sb')
    del S, S_b

    eigen_values, eigen_vectors = get_eigen_values_vectors(S_inv_Sb)
    del S_inv_Sb
    print('calculated eigen values and eigen vectors')

    # Get eigen-(value, vector) pairs, sorted in desc order of values
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
    eigen_pairs.sort()
    eigen_pairs.reverse()
    eigen_pairs = eigen_pairs[0:38]
    print('calculated pairs')
    del eigen_values, eigen_vectors

    new_bases = []
    for value, vector in eigen_pairs:
        # noinspection PySimplifyBooleanCheck
        if new_bases == []:
            new_bases = vector

        new_bases = np.hstack((new_bases, vector))

    new_bases = np.asmatrix(new_bases)

    return new_bases
