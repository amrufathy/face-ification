from sklearn.neighbors import KNeighborsClassifier

from lib.data_handling import *
from lib.metrics import get_accuracy
from lib.pca import pca

os.environ['MKL_DYNAMIC'] = 'false'

test_split = 0.3

all_data, train_data, test_data = generate_face_non_face_data_matrix(test_ratio=test_split)
data_matrix, labels = all_data[:, :-1], all_data[:, -1]
train_matrix, train_labels = train_data[:, :-1], train_data[:, -1]
test_matrix, test_labels = test_data[:, :-1], test_data[:, -1]

projection_matrix = pca(data_matrix, 0.9)

new_train_matrix = train_matrix.dot(projection_matrix)
new_test_matrix = test_matrix.dot(projection_matrix)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(new_train_matrix, np.ravel(train_labels))
predictions = knn.predict(new_test_matrix)

accuracy = get_accuracy(test_labels, predictions)
print(accuracy)
