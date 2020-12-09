from scipy import stats
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, classification_report, confusion_matrix
import random
import numpy as np

def randomBaselineClassifier(output):
    yPred = np.random.randint(1, 4, size=[7465])
    print("----------------Random Baseline-----------------")
    printingConfusionMatrix(output, yPred)
    print("----------------------------------------")


def modeBaselineClassifier(output):
    mode = stats.mode(output)
    print("----------------Mode Baseline-----------------")
    yPred = [mode.mode[0]] * len(output)
    printingConfusionMatrix(output, yPred)
    print("----------------------------------------")


def printingConfusionMatrix(yTest, yPred):
	print(classification_report(yTest, yPred))
	print(confusion_matrix(yTest, yPred))
	print("accuracy", accuracy_score(yTest, yPred))

	
