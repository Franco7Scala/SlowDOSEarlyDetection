import pickle
import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from src.support import utils
from src.support.utils import get_base_dir

utils.seed_everything(1) #seed

x_test = pickle.load(open(f'{get_base_dir()}/pickels/x_test_slowdos.pkl', 'rb'))
y_test = pickle.load(open(f'{get_base_dir()}/pickels/y_test_slowdos.pkl', 'rb'))
x_train = pickle.load(open(f'{get_base_dir()}/pickels/x_train_slowdos.pkl', 'rb'))
y_train = pickle.load(open(f'{get_base_dir()}/pickels/y_train_slowdos.pkl', 'rb'))

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

print(f"Starting RandomForest testing...")
rf_pred = rf_model.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, average="weighted")
rf_recall = recall_score(y_test, rf_pred, average="weighted")
rf_f1 = f1_score(y_test, rf_pred, average="weighted")
rf_auc = roc_auc_score(y_test, rf_pred, average="weighted")
print("RandomForest test results:")
print(f"accuracy: {rf_accuracy}, precision: {rf_precision}, recall: {rf_recall}, f1: {rf_f1}, auc: {rf_auc}")
print(classification_report(y_test, rf_pred, target_names=["Benign", "SlowDoS"]))