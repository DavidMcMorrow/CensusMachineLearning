import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import time

def logistic_regression_model_cross_validation(X_train, Y_train):

    #l2 penalty
    mean_accuracy = []
    accuracy_std_error = []
    mean_precision = []
    precision_std_error = []
    c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    for c_value in c_range:
        model = LogisticRegression(penalty="l2", C=c_value, max_iter=2000)
        model.fit(X_train, Y_train)
        scores = cross_val_score(model, X_train, Y_train, cv=5, scoring="accuracy")
        mean_accuracy.append(np.array(scores).mean())
        accuracy_std_error.append(np.array(scores).std())
        scores = cross_val_score(model, X_train, Y_train, cv=5, scoring="precision_macro")
        mean_precision.append(np.array(scores).mean())
        precision_std_error.append(np.array(scores).std())

    for i in range(0, len(c_range)):
        print(c_range[i], ":   Accuracy - ", mean_accuracy[i], "  Error - ", accuracy_std_error[i], " Precision - ", mean_precision[i], " Error - ", precision_std_error[i])

    fig = plt.figure()
    plt.title("C Value vs Mean Accuracy With Standard Deviation using L2 Regularisation")
    plt.errorbar(c_range, mean_accuracy, yerr=accuracy_std_error, color="b", label="Accuracy")
    plt.errorbar(c_range, mean_precision, yerr=precision_std_error, color="orange", label="Precision")
    plt.xlabel("C Values")
    plt.ylabel("Mean Accuracy")
    plt.xscale("log")
    plt.legend()
    plt.show()

def train_chosen_logistic_regression_model(X_train, Y_train, C_value):
    start = time.time()
    model = LogisticRegression(penalty="l2", C=C_value, max_iter=2000)
    model.fit(X_train, Y_train)
    end = time.time()
    print("time to complete logistic regression training for C value ", C_value, " - ", round(end-start))
    return model


#notes
#C = 0.001 gives similar accuracy to the baseline 56%
#as C increases accuracy and precision increase quiclky to a C value of 1 before increasing more slowly
#time taken to train each model increases as the C value increases
#C = 1 seems to be the best choice for the model (everything after that is likely overfitting without much of an increase in performance)