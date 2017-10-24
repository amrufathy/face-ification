import numpy as np

from cache import cache

base_path = 'pickles/lda/'


@cache(base_path + 'sb_matrix.pickle')
def calculate_sb_matrix(mean_vectors, overall_mean, classes_matrices):
    # noinspection PyPep8Naming
    S_b = 0
    for idx, m_v in enumerate(mean_vectors):
        diff = m_v - overall_mean
        outer = diff.dot(diff.T)
        S_b += (classes_matrices[idx].shape[0] * outer)

    return np.asmatrix(S_b)


@cache(base_path + 's_matrix.pickle')
def calculate_s_matrix(centered_class_matrices):
    # noinspection PyPep8Naming
    S = 0
    for ccm in centered_class_matrices:
        S += ccm.T.dot(ccm)

    return np.asmatrix(S)


# noinspection PyPep8Naming
@cache(base_path + 's_inv_sb_matrix.pickle')
def calculate_s_inv_sb_matrix(S, S_b):
    return np.linalg.inv(S).dot(S_b)


@cache(base_path + 'eigen_values_vectors.pickle')
def calculate_eigen_values_vectors(in_matrix):
    return np.linalg.eigh(in_matrix)


# noinspection PyPep8Naming
def lda(data_matrix, classes_matrices):
    mean_vectors = [np.mean(class_matrix[:, :-1], axis=0).T for class_matrix in classes_matrices]
    overall_mean = np.mean(data_matrix[:, :-1], axis=0).T
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
