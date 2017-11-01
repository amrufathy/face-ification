def get_accuracy(labels, predictions):
    import numpy as np
    labels, predictions = np.ravel(labels), np.ravel(predictions)

    # noinspection PyTypeChecker
    difference = list(predictions == labels)
    accuracy = difference.count(True) / len(difference)
    return accuracy * 100
