import pickle
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier

from src.support import utils

utils.seed_everything(1) #seed

x_test = pickle.load(open('/src/support/files/pickels/x_test_slowdos.pkl', 'rb'))
y_test = pickle.load(open('/src/support/files/pickels/y_test_slowdos.pkl', 'rb'))
x_train = pickle.load(open('/src/support/files/pickels/x_train_slowdos.pkl', 'rb'))
y_train = pickle.load(open('/src/support/files/pickels/y_train_slowdos.pkl', 'rb'))


#-----DecisionTree model-----#
dt_model = DecisionTreeClassifier(max_leaf_nodes=3)
#-----DecisionTree model-----#

print("Starting DecisionTree model training...")
dt_start = time.time()
#-----Decisiontree model training-----#
dt_model.fit(x_train, y_train)
#-----DecisionTree model training-----#
dt_end = time.time()
print("DecisionTree done!")
print(f"Training time: {dt_end - dt_start:.2f} seconds")

print(f"Starting DecisionTree testing...")
dt_pred = dt_model.predict(x_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred, average="weighted")
dt_recall = recall_score(y_test, dt_pred, average="weighted")
dt_f1 = f1_score(y_test, dt_pred, average="weighted")
print("DecisionTree test results:")
print(f"accuracy: {dt_accuracy}, precision: {dt_precision}, recall: {dt_recall}, f1: {dt_f1}")
print(classification_report(y_test, dt_pred, target_names=["Beinign", "SlowDoS"]))