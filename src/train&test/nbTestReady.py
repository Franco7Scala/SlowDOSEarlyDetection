import pickle
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    roc_auc_score
from sklearn.naive_bayes import GaussianNB

from src.support import utils
from src.support.utils import get_base_dir

utils.seed_everything(1) #seed

x_test = pickle.load(open(f'{get_base_dir()}/pickels/x_test_slowdos.pkl', 'rb'))
y_test = pickle.load(open(f'{get_base_dir()}/pickels/y_test_slowdos.pkl', 'rb'))
x_train = pickle.load(open(f'{get_base_dir()}/pickels/x_train_slowdos.pkl', 'rb'))
y_train = pickle.load(open(f'{get_base_dir()}/pickels/y_train_slowdos.pkl', 'rb'))

#-----NaiveBayes model-----#
nb_model = GaussianNB()
#-----NaiveBayes model-----#

print("Starting NaiveBayes model training...")
nb_start = time.time()
#-----NaiveBayes model training-----#
nb_model.fit(x_train, y_train.ravel())
#-----NaiveBayes model training-----#
nb_end = time.time()
print("NaiveBayes done!")
print(f"Training time: {nb_end - nb_start:.2f} seconds")

print(f"Starting NaiveBayes testing...")
nb_pred = nb_model.predict(x_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_precision = precision_score(y_test, nb_pred, average="weighted")
nb_recall = recall_score(y_test, nb_pred, average="weighted")
nb_f1 = f1_score(y_test, nb_pred, average="weighted")
nb_auc = roc_auc_score(y_test, nb_pred, average="weighted")
print("NaiveBayes test results:")
print(f"accuracy: {nb_accuracy}, precision: {nb_precision}, recall: {nb_recall}, f1: {nb_f1}, auc: {nb_auc}")
print(classification_report(y_test, nb_pred, target_names=["Benign", "SlowDoS"]))