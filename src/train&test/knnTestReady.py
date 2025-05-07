import pickle
import time
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from src.support import utils

os.environ['OMP_NUM_THREADS'] = '1'

utils.seed_everything(1) #seed

days = ["Tuesday", "Wednesday", "Thursday", "Friday"]

test_loaders = pickle.load(open('C:/Users/black/PycharmProjects/SlowDOSEarlyDetection/src/support/files/test_loaders.pkl', 'rb'))
x_train = pickle.load(open('C:/Users/black/PycharmProjects/SlowDOSEarlyDetection/src/support/files/x_train.pkl', 'rb'))
y_train = pickle.load(open('C:/Users/black/PycharmProjects/SlowDOSEarlyDetection/src/support/files/y_train.pkl', 'rb'))
weights = pickle.load(open('C:/Users/black/PycharmProjects/SlowDOSEarlyDetection/src/support/files/weights_tensor.pkl', 'rb'))

#-----KNN model-----#
knn_model = KNeighborsClassifier(n_neighbors=3)
#-----KNN model-----#

print("Starting KNN model training...")
knn_start = time.time()
#-----KNN model training-----#
knn_model.fit(x_train, y_train.ravel())
#-----KNN model training-----#
knn_end = time.time()
print("KNN done!")
print(f"Training time: {knn_end - knn_start:.2f} seconds")

for i in range(len(test_loaders)):
    x_test, y_test = utils.convertDataLoaderToNumpy(test_loaders[i])  #DataLoader to numpy
    print(f"Starting {days[i]} KNN testing...")
    knn_pred = knn_model.predict(x_test)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_precision = precision_score(y_test, knn_pred, average='weighted')
    knn_recall = recall_score(y_test, knn_pred, average='weighted')
    knn_f1 = f1_score(y_test, knn_pred, average='weighted')
    print("KNN test results:")
    print(f"accuracy: {knn_accuracy}, precision: {knn_precision}, recall: {knn_recall}, f1: {knn_f1}")