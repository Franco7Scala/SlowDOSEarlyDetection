import pickle
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.support import utils

utils.seed_everything(1) #seed

days = ["Tuesday", "Wednesday", "Thursday", "Friday"]

test_loaders = pickle.load(open('C:/Coding/PyCharm Projects/src/support/files/test_loaders.pkl', 'rb'))
x_train = pickle.load(open('C:/Coding/PyCharm Projects/src/support/files/x_train.pkl', 'rb'))
y_train = pickle.load(open('C:/Coding/PyCharm Projects/src/support/files/y_train.pkl', 'rb'))

#-----XGBoost model-----#
xgb_model = xgb.XGBClassifier()
#-----XGBoost model-----#

print("Starting XGBoost model training...")
xgb_start = time.time()
#-----XGBoost model training-----#
xgb_model.fit(x_train, y_train)
#-----XGBoost model training-----#
xgb_end = time.time()
print("XGBoost done!")
print(f"Training time: {xgb_end - xgb_start:.2f} seconds")

for i in range(len(test_loaders)):
    x_test, y_test = utils.convertDataLoaderToNumpy(test_loaders[i])  #DataLoader to numpy
    print(f"Starting {days[i]} XGBoost testing...")
    xgb_pred = xgb_model.predict(x_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_precision = precision_score(y_test, xgb_pred)
    xgb_recall = recall_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred)
    print("XGBoost test results:")
    print(f"accuracy: {xgb_accuracy}, precision: {xgb_precision}, recall: {xgb_recall}, f1: {xgb_f1}")
