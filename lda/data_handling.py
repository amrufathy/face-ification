import os

import numpy as np
from scipy.ndimage import imread


def generate_balanced_data_matrix(path='orl_faces'):
    data_matrix, classes_matrices = [], []

    for subject in os.listdir(path):
        subject_matrix = []
        for idx, image_src in enumerate(os.listdir(os.path.join(path, subject))):
            image_src = os.path.join(path, os.path.join(subject, image_src))
            image = imread(image_src, mode='L')

            # noinspection PyTypeChecker
            image = np.append(image, int(subject[1:]))

            data_matrix.append(image)
            subject_matrix.append(image)

        classes_matrices.append(np.asmatrix(subject_matrix))

    data_matrix = np.asmatrix(data_matrix)

    return data_matrix, classes_matrices
