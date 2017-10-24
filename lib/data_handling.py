import os

import numpy as np
from scipy.ndimage import imread
from sklearn.model_selection import train_test_split

from lib.cache import cache


def generate_random_data_matrix(path='orl_faces', test_ratio=0.3):
    data_matrix, train_matrix, test_matrix = [], [], []

    for subject in os.listdir(path):
        subject_matrix = []
        for image_src in os.listdir(os.path.join(path, subject)):
            image_src = os.path.join(path, os.path.join(subject, image_src))
            image = imread(image_src, mode='L').flatten()

            # noinspection PyTypeChecker
            image = np.append(image, int(subject[1:]))

            subject_matrix.append(image)
            data_matrix.append(image)

        train, test = train_test_split(subject_matrix, test_size=test_ratio)
        train_matrix.extend(train)
        test_matrix.extend(test)

    data_matrix, train_matrix, test_matrix = np.asmatrix(data_matrix), np.asmatrix(train_matrix), np.asmatrix(
        test_matrix)

    return data_matrix, train_matrix, test_matrix


def generate_balanced_data_matrix(path='orl_faces'):
    data_matrix, train_matrix, test_matrix = [], [], []

    for subject in os.listdir(path):
        for idx, image_src in enumerate(os.listdir(os.path.join(path, subject))):
            image_src = os.path.join(path, os.path.join(subject, image_src))
            image = imread(image_src, mode='L').flatten()

            # print(idx, image_src)
            # noinspection PyTypeChecker
            image = np.append(image, int(subject[1:]))
            data_matrix.append(image)

            if idx % 2 == 0:
                train_matrix.append(image)
            else:
                test_matrix.append(image)

    data_matrix, train_matrix, test_matrix = np.asmatrix(data_matrix), np.asmatrix(train_matrix), np.asmatrix(
        test_matrix)

    return data_matrix, train_matrix, test_matrix


def generate_subjects_matrices(path='orl_faces'):
    classes_matrices = []

    for subject in os.listdir(path):
        subject_matrix = []
        for idx, image_src in enumerate(os.listdir(os.path.join(path, subject))):
            image_src = os.path.join(path, os.path.join(subject, image_src))
            image = imread(image_src, mode='L')

            # noinspection PyTypeChecker
            image = np.append(image, int(subject[1:]))

            subject_matrix.append(image)

        classes_matrices.append(np.asmatrix(subject_matrix))

    return classes_matrices


@cache('pickles/pca/projection_matrices.pickle')
def get_projection_matrices(data_matrix, alphas):
    from lib.pca import pca
    projection_matrices = []
    for alpha in alphas:
        projection_matrices.append(pca(data_matrix, alpha))

    return projection_matrices
