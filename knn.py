import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import time

def knn_model_cross_validation(X_train, Y_train):
    mean_accuracy = []
    accuracy_std_error = []
    mean_precision = []
    precision_std_error = []
    distance_parameters = [0, 1, 5, 10, 25]

    for d_param in distance_parameters:
        def gaussian_kernel(distances):
            weights = np.exp(-d_param*(distances**2))
            return weights/np.sum(weights)

        model =  KNeighborsClassifier(n_neighbors=1000,weights=gaussian_kernel)
        model.fit(X_train, Y_train)
        scores = cross_val_score(model, X_train, Y_train, cv=5, scoring="accuracy")
        mean_accuracy.append(np.array(scores).mean())
        accuracy_std_error.append(np.array(scores).std())
        scores = cross_val_score(model, X_train, Y_train, cv=5, scoring="precision_macro")
        mean_precision.append(np.array(scores).mean())
        precision_std_error.append(np.array(scores).std())

    for i in range(0, len(distance_parameters)):
        print(distance_parameters[i], ":   Accuracy - ", mean_accuracy[i], "  Error - ", accuracy_std_error[i], " Precision - ", mean_precision[i], " Error - ", precision_std_error[i])

    fig = plt.figure()
    plt.title("KNN: Cross Validation, KFold = 5")
    plt.errorbar(distance_parameters, mean_accuracy, yerr=accuracy_std_error, color="b", label="Accuracy")
    plt.errorbar(distance_parameters, mean_precision, yerr=precision_std_error, color="orange", label="Precision")
    plt.xlabel("γ Values")
    plt.ylabel("Accuracy & Precision (%)")
    plt.legend(loc = 'lower right')
    plt.show()

def train_chosen_knn_model(X_train, Y_train, gamma, X_test, Y_test):

    def gaussian_kernel(distances):
        weights = np.exp(-10*(distances**2))
        return weights/np.sum(weights)

    start = time.time()
    model = KNeighborsClassifier(n_neighbors=1000,weights=gaussian_kernel)
    model.fit(X_train, Y_train)
    end = time.time()
    print("time to complete Knn training for γ value ", gamma, " - ", round(end-start))
    
    start = time.time()
    pred = model.predict(X_test)
    end = time.time()
    print("time to make knn predictions for γ value ", gamma, " - ", round(end-start))
    printingConfusionMatrix(Y_test, pred)

def printingConfusionMatrix(yTest, yPred):
	print(classification_report(yTest, yPred))
	print(confusion_matrix(yTest, yPred))
	print("accuracy", accuracy_score(yTest, yPred))