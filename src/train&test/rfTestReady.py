import pickle
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from src.support import utils

utils.seed_everything(1) #seed

days = ["Tuesday", "Wednesday", "Thursday", "Friday"]

test_loaders = pickle.load(open('C:/Users/black/PycharmProjects/SlowDOSEarlyDetection/src/support/files/test_loaders.pkl', 'rb'))
x_train = pickle.load(open('C:/Users/black/PycharmProjects/SlowDOSEarlyDetection/src/support/files/x_train.pkl', 'rb'))
y_train = pickle.load(open('C:/Users/black/PycharmProjects/SlowDOSEarlyDetection/src/support/files/y_train.pkl', 'rb'))
weights = pickle.load(open('C:/Users/black/PycharmProjects/SlowDOSEarlyDetection/src/support/files/weights_tensor.pkl', 'rb'))

#-----RandomForest model-----#
rf_model = RandomForestClassifier(n_estimators=80)
#-----RandomForest model-----#

print("Starting RandomForest model training...")
rf_start = time.time()
#-----RandomForest model training-----#
rf_model.fit(x_train, y_train.ravel())
#-----RandomForest model training-----#
rf_end = time.time()
print("RandomForest done!")
print(f"Training time: {rf_end - rf_start:.2f} seconds")

for i in range(len(test_loaders)):
    x_test, y_test = utils.convertDataLoaderToNumpy(test_loaders[i])  #DataLoader to numpy
    print(f"Starting {days[i]} RandomForest testing...")
    rf_pred = rf_model.predict(x_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred, average="weighted")
    rf_recall = recall_score(y_test, rf_pred, average="weighted")
    rf_f1 = f1_score(y_test, rf_pred, average="weighted")
    print("RandomForest test results:")
    print(f"accuracy: {rf_accuracy}, precision: {rf_precision}, recall: {rf_recall}, f1: {rf_f1}")