import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from lib.data_handling import *
from lib.metrics import get_accuracy

os.environ['MKL_DYNAMIC'] = 'false'

test_split = 0.3

all_data, train_data, test_data = generate_random_data_matrix(test_ratio=test_split)
data_matrix, labels = all_data[:, :-1], all_data[:, -1]
train_matrix, train_labels = train_data[:, :-1], train_data[:, -1]
test_matrix, test_labels = test_data[:, :-1], test_data[:, -1]

alphas = [0.8, 0.85, 0.9, 0.95]
accuracies = []

projection_matrices = get_projection_matrices(data_matrix, alphas)

for projection_matrix in projection_matrices:
    new_train_matrix = train_matrix.dot(projection_matrix)
    new_test_matrix = test_matrix.dot(projection_matrix)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(new_train_matrix, np.ravel(train_labels))
    predictions = knn.predict(new_test_matrix)

    accuracy = get_accuracy(test_labels, predictions)
    print(accuracy)
    accuracies.append(accuracy)

plt.plot(alphas, accuracies)
plt.scatter(alphas, accuracies)
plt.title(('Train ({}%) - Test ({}%)'.format(((1 - test_split) * 100), (test_split * 100))))
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.show()
