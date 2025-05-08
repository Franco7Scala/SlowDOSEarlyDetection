import pickle
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB

from src.support import utils

utils.seed_everything(1) #seed

days = ["Tuesday", "Wednesday", "Thursday", "Friday"]

test_loaders = pickle.load(open('C:/Coding/PyCharm Projects/src/support/files/test_loaders.pkl', 'rb'))
x_train = pickle.load(open('C:/Coding/PyCharm Projects/src/support/files/x_train.pkl', 'rb'))
y_train = pickle.load(open('C:/Coding/PyCharm Projects/src/support/files/y_train.pkl', 'rb'))
weights = pickle.load(open('C:/Coding/PyCharm Projects/src/support/files/weights_tensor.pkl', 'rb'))

#-----NaiveBayes model-----#
nb_model = GaussianNB()
#-----NaiveBayes model-----#

print("Starting NaiveBayes model training...")
nb_start = time.time()
#-----NaiveBayes model training-----#
nb_model.fit(x_train, y_train)
#-----NaiveBayes model training-----#
nb_end = time.time()
print("NaiveBayes done!")
print(f"Training time: {nb_end - nb_start:.2f} seconds")

for i in range(len(test_loaders)):
    x_test, y_test = utils.convertDataLoaderToNumpy(test_loaders[i])  #DataLoader to numpy
    print(f"Starting {days[i]} NaiveBayes testing...")
    nb_pred = nb_model.predict(x_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)
    nb_precision = precision_score(y_test, nb_pred, average="weighted")
    nb_recall = recall_score(y_test, nb_pred, average="weighted")
    nb_f1 = f1_score(y_test, nb_pred, average="weighted")
    print("NaiveBayes test results:")
    print(f"accuracy: {nb_accuracy}, precision: {nb_precision}, recall: {nb_recall}, f1: {nb_f1}")