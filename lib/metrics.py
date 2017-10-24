def get_accuracy(labels, predictions):
    labels = list(labels.flatten().tolist()[0])

    # noinspection PyTypeChecker
    difference = list(predictions == labels)
    accuracy = difference.count(True) / len(difference)
    return accuracy * 100
