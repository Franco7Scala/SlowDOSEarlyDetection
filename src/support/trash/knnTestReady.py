import pickle
import time
import os

default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from src.support import utils
from src.support.utils import get_base_dir

os.environ['OMP_NUM_THREADS'] = '1'

utils.seed_everything(1) #seed

x_test = pickle.load(open(f'{get_base_dir()}/pickles/x_test_slowdos.pkl', 'rb'))
y_test = pickle.load(open(f'{get_base_dir()}/pickles/y_test_slowdos.pkl', 'rb'))
x_train = pickle.load(open(f'{get_base_dir()}/pickles/x_train_slowdos.pkl', 'rb'))
y_train = pickle.load(open(f'{get_base_dir()}/pickles/y_train_slowdos.pkl', 'rb'))

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

print(f"Starting KNN testing...")
knn_pred = knn_model.predict(x_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred, average='weighted')
knn_recall = recall_score(y_test, knn_pred, average='weighted')
knn_f1 = f1_score(y_test, knn_pred, average='weighted')
knn_auc = roc_auc_score(y_test, knn_pred, average='weighted')
print("KNN test results:")
print(f"accuracy: {knn_accuracy}, precision: {knn_precision}, recall: {knn_recall}, f1: {knn_f1}, auc: {knn_auc}")
print(classification_report(y_test, knn_pred, target_names=["Benign", "SlowDoS"]))