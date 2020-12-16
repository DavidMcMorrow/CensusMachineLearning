from scipy import stats
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, classification_report, confusion_matrix
import random
import numpy as np
import time

def randomBaselineClassifier(testingOutput):
    start = time.time()
    print("len(testingOutput) = ", len(testingOutput))
    yPred = np.random.randint(1, 4, size=[len(testingOutput)])
    print("----------------Random Baseline-----------------")
    printingConfusionMatrix(testingOutput, yPred)
    end = time.time()
    print("time to complete random baseline classifier is: ", round(end-start))
    print("----------------------------------------")
    
def modeBaselineClassifier(trainingOutput, testingOutput):
    start = time.time()
    mode = stats.mode(trainingOutput)
    print("----------------Mode Baseline-----------------")
    yPred = [mode.mode[0]] * len(testingOutput)
    printingConfusionMatrix(testingOutput, yPred)
    end = time.time()
    print("time to complete most common baseline classifier is:", round(end-start))
    print("----------------------------------------")

def printingConfusionMatrix(yTest, yPred):
	print(classification_report(yTest, yPred))
	print(confusion_matrix(yTest, yPred))
	print("accuracy", accuracy_score(yTest, yPred))