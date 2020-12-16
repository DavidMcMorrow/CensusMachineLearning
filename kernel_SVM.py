import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import time

def kernel_SVM_model_cross_validation(X_train, Y_train):
    mean_accuracy = []
    accuracy_std_error = []
    mean_precision = []
    precision_std_error = []
    c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for c_value in c_range:
        model = svm.SVC(kernel="precomputed", C=c_value, max_iter=5000)
        gram_train = np.dot(X_train, X_train.T)
        model.fit(gram_train, Y_train)
        scores = cross_val_score(model, gram_train, Y_train, cv=5, scoring="accuracy")
        mean_accuracy.append(np.array(scores).mean())
        accuracy_std_error.append(np.array(scores).std())
        scores = cross_val_score(model, gram_train, Y_train, cv=5, scoring="precision_macro")
        mean_precision.append(np.array(scores).mean())
        precision_std_error.append(np.array(scores).std())

    for i in range(0, len(c_range)):
        print(c_range[i], ":   Accuracy - ", mean_accuracy[i], "  Error - ", accuracy_std_error[i], " Precision - ", mean_precision[i], " Error - ", precision_std_error[i])

    fig = plt.figure()
    plt.title("C Value vs Mean Accuracy With Standard Deviation kernel SVM")
    plt.errorbar(c_range, mean_accuracy, yerr=accuracy_std_error, color="b", label="Accuracy")
    plt.errorbar(c_range, mean_precision, yerr=precision_std_error, color="orange", label="Precision")
    plt.xlabel("C Values")
    plt.ylabel("Mean Accuracy")
    plt.xscale("log")
    plt.legend()
    plt.show()

def train_chosen_kernel_SVM_model(X_train, Y_train, C_value, X_test, Y_test):
    start = time.time()
    model = svm.SVC(kernel="precomputed", C=C_value, max_iter=2000)
    gram_train = np.dot(X_train, X_train.T)
    model.fit(gram_train, Y_train)
    end = time.time()
    print("time to complete logistic regression training for C value ", C_value, " - ", round(end-start))
    
    start = time.time()
    gram_test = np.dot(X_test, X_train.T)
    pred = model.predict(gram_test)
    end = time.time()
    print("time to make Kernel SVM predictions for C value ", C_value, " - ", round(end-start))
    printingConfusionMatrix(Y_test, pred)

def printingConfusionMatrix(yTest, yPred):
	print(classification_report(yTest, yPred))
	print(confusion_matrix(yTest, yPred))
	print("accuracy", accuracy_score(yTest, yPred))