import pickle
import time

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    roc_auc_score

from src.support import utils
from src.support.utils import get_base_dir

utils.seed_everything(1) #seed

x_test = pickle.load(open(f'{get_base_dir()}/pickles/x_test_slowdos.pkl', 'rb'))
y_test = pickle.load(open(f'{get_base_dir()}/pickles/y_test_slowdos.pkl', 'rb'))
x_train = pickle.load(open(f'{get_base_dir()}/pickles/x_train_slowdos.pkl', 'rb'))
y_train = pickle.load(open(f'{get_base_dir()}/pickles/y_train_slowdos.pkl', 'rb'))

#-----XGBoost model-----#
xgb_model = xgb.XGBClassifier(n_estimators=80)
#-----XGBoost model-----#

print("Starting XGBoost model training...")
xgb_start = time.time()
#-----XGBoost model training-----#
xgb_model.fit(x_train, y_train)
#-----XGBoost model training-----#
xgb_end = time.time()
print("XGBoost done!")
print(f"Training time: {xgb_end - xgb_start:.2f} seconds")

print(f"Starting XGBoost testing...")
xgb_pred = xgb_model.predict(x_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred, average="weighted")
xgb_recall = recall_score(y_test, xgb_pred, average="weighted")
xgb_f1 = f1_score(y_test, xgb_pred, average="weighted")
xgb_auc = roc_auc_score(y_test, xgb_pred, average='weighted')
print("XGBoost test results:")
print(f"accuracy: {xgb_accuracy}, precision: {xgb_precision}, recall: {xgb_recall}, f1: {xgb_f1}, auc: {xgb_auc}")
print(classification_report(y_test, xgb_pred, target_names=["Benign", "SlowDoS"]))
