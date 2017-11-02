import numpy as np

from cachpy import cachpy

base_path = 'pickles/lda/'


@cachpy(base_path + 'sb_matrix.pickle')
def calculate_sb_matrix(mean_vectors, overall_mean, classes_matrices):
    # noinspection PyPep8Naming
    S_b = 0
    for idx, m_v in enumerate(mean_vectors):
        diff = m_v - overall_mean
        outer = diff.dot(diff.T)
        S_b += (classes_matrices[idx].shape[0] * outer)

    return np.asmatrix(S_b)


@cachpy(base_path + 's_matrix.pickle')
def calculate_s_matrix(centered_class_matrices):
    # noinspection PyPep8Naming
    S = 0
    for ccm in centered_class_matrices:
        S += ccm.T.dot(ccm)

    return np.asmatrix(S)


# noinspection PyPep8Naming
@cachpy(base_path + 's_inv_sb_matrix.pickle')
def calculate_s_inv_sb_matrix(S, S_b):
    return np.linalg.inv(S).dot(S_b)


@cachpy(base_path + 'eigen_values_vectors.pickle')
def calculate_eigen_values_vectors(in_matrix):
    return np.linalg.eigh(in_matrix)


# noinspection PyPep8Naming
def lda(data_matrix, classes_matrices):
    mean_vectors = [np.mean(class_matrix[:, :-1], axis=0).T for class_matrix in classes_matrices]
    overall_mean = np.mean(data_matrix, axis=0).T
    print('calculated means')

    S_b = calculate_sb_matrix(mean_vectors, overall_mean, classes_matrices)
    print('calculated Sb matrix')

    centered_class_matrices = [(class_matrix[:, :-1] - mean_vectors[i].T) for i, class_matrix in
                               enumerate(classes_matrices)]
    print('calculated centered matrices')
    del mean_vectors, overall_mean

    S = calculate_s_matrix(centered_class_matrices)
    print('calculated S matrix')
    del centered_class_matrices

    S_inv_Sb = calculate_s_inv_sb_matrix(S, S_b)
    print('calculated S_inv_Sb')
    del S, S_b

    eigen_values, eigen_vectors = calculate_eigen_values_vectors(S_inv_Sb)
    del S_inv_Sb
    print('calculated eigen values and eigen vectors')

    idx = eigen_values.argsort()[::-1]
    eigen_vectors = eigen_vectors[:, idx]
    print('sorted eigen values and vectors')

    return np.asmatrix(eigen_vectors[:, :38])
