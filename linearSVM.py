from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score

def printInfo(title, model, yPred, testing_y, c):
    print("--------title =", title, "C =", c, "------------")
    modelConfusionMatrix = confusion_matrix(testing_y, yPred)
    print("-----Confusion Matrix---- \n")
    print(modelConfusionMatrix)
    print("----------------------------- \n")
    print("Accuracy(macro) = ", accuracy_score(yPred, testing_y))
    print("Precision(macro) = ", precision_score(yPred, testing_y, average='macro'))
    print("Recall(macro) = ", recall_score(yPred, testing_y, average='macro'))
    print("F1 Score(macro) = ",f1_score(yPred, testing_y, average='macro'))
    print("-----------------------")

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
        print("c = ", c)
        clf = LinearSVC(C=c, max_iter=2000)
        clf.fit(training_x, training_y, )
        accuracyScores = cross_val_score(clf, training_x, training_y, cv=k_fold, scoring='accuracy') * 100
        precisionScores = cross_val_score(clf, training_x, training_y, scoring='precision_macro') * 100
        accuracyMeanError.append(np.array(accuracyScores).mean())
        accuracyStdError.append(np.array(accuracyScores).std())
        precisionMeanError.append(np.array(precisionScores).mean())
        precisionStdError.append(np.array(precisionScores).std())
            
    plotTitle = "LinearSVC: Cross Validation, KFold = " + str(k_fold)
    plottingCrossValidation(CValues, accuracyMeanError, accuracyStdError, precisionMeanError, precisionStdError, 'C Values', 'Accuracy & Precision (%)', plotTitle)



def optimsedLinearSVMClassifier(training_x, training_y, testing_x, testing_y):
    optimisedC = 10
    clf = LinearSVC(C = optimisedC, max_iter=2000)
    clf.fit(training_x, training_y)
    yPred = clf.predict(testing_x)
    printInfo("LinearSVM", clf, yPred, testing_y, optimisedC)
        
          