import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score
from utils import *


train_data, train_labels = load_dataset("./data/train.pkl")
test_data, test_labels = load_dataset("./data/test.pkl")

for kernel in ["rbf", "linear"]:
    clf = svm.SVC(kernel=kernel)
    clf.fit(train_data, train_labels)

    train_predictions = clf.predict(train_data)
    test_predictions = clf.predict(test_data)

    print(f"train accuracy with {kernel}: ", accuracy_score(train_predictions, train_labels))
    print(f"test accuracy with {kernel}: ", accuracy_score(test_predictions, test_labels))