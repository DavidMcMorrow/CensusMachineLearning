from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, roc_curve, classification_report
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
import time

def plottingCrossValidation(xData, accuracyData, accuracyErrorbars, precisionData, precisionErrorbars,xAxis, yAxis, title):
    plt.figure()
    plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
    plt.errorbar(xData, precisionData, yerr=precisionErrorbars, linewidth=3, label="Precision", c='orange')
    plt.errorbar(xData, accuracyData, yerr=accuracyErrorbars, linewidth=3, label="Accuracy", c='b')
    plt.xscale("log")
    plt.xlabel(xAxis); plt.ylabel(yAxis); plt.title(title)
    plt.legend(); plt.show()

def linearSVMClassifierCrossValidation(training_x, training_y):
    print("trainingX", training_x)
    accuracyMeanError = []; accuracyStdError = []
    precisionMeanError = []; precisionStdError = []
    k_fold = 5
    CValues = [
            0.0001, 
            0.01, 
            1, 
            10,
            100, 
            1000
        ]
    for c in CValues:
        print("c =", c)
        clf = LinearSVC(C=c, max_iter=5000, dual=True)
        clf.fit(training_x, training_y, )
        accuracyScores = cross_val_score(clf, training_x, training_y, cv=k_fold, scoring='accuracy') * 100
        precisionScores = cross_val_score(clf, training_x, training_y, scoring='precision_micro') * 100
        accuracyMeanError.append(np.array(accuracyScores).mean())
        accuracyStdError.append(np.array(accuracyScores).std())
        precisionMeanError.append(np.array(precisionScores).mean())
        precisionStdError.append(np.array(precisionScores).std())
            
    plotTitle = "LinearSVC: Cross Validation, KFold = " + str(k_fold)
    plottingCrossValidation(CValues, accuracyMeanError, accuracyStdError, precisionMeanError, precisionStdError, 'C Values', 'Accuracy & Precision (%)', plotTitle)

def optimsedLinearSVMClassifier(training_x, training_y, testing_x, testing_y):
    start = time.time()
    optimisedC = 1
    clf = LinearSVC(C = optimisedC, max_iter=2000)
    clf.fit(training_x, training_y)
    end = time.time()
    print("time to complete LinearSVC training for C value ", optimisedC, " - ", round(end-start))

    start = time.time()
    yPred = clf.predict(testing_x)
    end = time.time()
    print("time to make LinearSVC predictions for C value ", optimisedC, " - ", round(end-start))
    print("--------title =", "LinearSVM", "C =", optimisedC, "------------")
    printingConfusionMatrix(testing_y, yPred)
    print("-----------------------")



    # start = time.time()
    # model = KNeighborsClassifier(n_neighbors=1000,weights=gaussian_kernel)
    # model.fit(X_train, Y_train)
    # end = time.time()
    # print("time to complete Knn training for γ value ", gamma, " - ", round(end-start))
    
    # start = time.time()
    # pred = model.predict(X_test)
    # end = time.time()
    # print("time to make knn predictions for γ value ", gamma, " - ", round(end-start))
    # printingConfusionMatrix(Y_test, pred)

def printingConfusionMatrix(yTest, yPred):
	print(classification_report(yTest, yPred))
	print(confusion_matrix(yTest, yPred))
	print("accuracy", accuracy_score(yTest, yPred))