from sklearn.neighbors import KNeighborsClassifier

from lda.data_handling import generate_balanced_data_matrix
from lda.lda import lda
from metrics import get_accuracy
from pca.data_handling import generate_balanced_data_matrix as get_data

data_matrix, classes_matrices = generate_balanced_data_matrix()

projection_matrix = lda(data_matrix, classes_matrices)

all_data, train_data, test_data = get_data()
data_matrix, labels = all_data[:, :-1], all_data[:, -1]
train_matrix, train_labels = train_data[:, :-1], train_data[:, -1]
test_matrix, test_labels = test_data[:, :-1], test_data[:, -1]

new_train_matrix = train_matrix.dot(projection_matrix)
new_test_matrix = test_matrix.dot(projection_matrix)

knn = KNeighborsClassifier(n_neighbors=1, n_jobs=1)
knn.fit(new_train_matrix, train_labels)
predictions = knn.predict(new_test_matrix)

accuracy = get_accuracy(test_labels, predictions)
print(accuracy)
