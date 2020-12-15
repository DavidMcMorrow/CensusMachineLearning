import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import math as m
from linearSVM import *
from baseline import *
from kernel_SVM import *
from logistic_regression_model import *
from knn import *
from sklearn.metrics import accuracy_score, average_precision_score

def produce_input_feature(list_of_inputs, uniqueValues):
    
    #tokenise each of the input fields
    vectorizer = CountVectorizer()
    vectorizer.fit(uniqueValues)
    
    
    people = []
    for i in range(0, len(list_of_inputs[0])):
        person = []
        for input in list_of_inputs:
            person.append(str(input[i]))
            person.append(" ")
        people.append(("".join(person)))

    feature_inputs = []
    for person in people:
        feature_inputs.append(vectorizer.transform([person]).toarray())

    X = []
    for input in feature_inputs:
        X.append(input[0])

    X = np.array(X)
    
    print("shape of X", X.shape)

    return X

def printUniqueClassess(classesData, classesNames):
    uniqueValues = []
    for i in range(0, len(classesData)):
        print("Number of unique", classesNames[i], "values =", len(list(set(classesData[i]))))
        uniqueValues.extend(set(classesData[i]))
    return uniqueValues

def generalisingSmallClasses(className, newValue, threshold):
    uniqueValues = list(set(className))
    # print("unique", uniqueValues)
    for i in uniqueValues:
        if((className == i).sum() < threshold):
            className[className == i] = newValue
    return className
    
df = pd.read_csv("test_1851.csv", comment='#')
surname = df.iloc[:,0]
forename = df.iloc[:,1]
family_ID = df.iloc[:,2]
parish = df.iloc[:,3]
age = df.iloc[:,4]
sex = df.iloc[:,5]
relation_to_head_of_household = df.iloc[:,6]
marital_status = df.iloc[:,7]
occupation = df.iloc[:,8]
education = df.iloc[:,9] #this is what we are trying to classify

occupation = generalisingSmallClasses(occupation, "other", 50)

for i in range(0, len(surname)):
    surname[i] = "sur" + str(surname[i]).lower() + "sur"
    forename[i] = "fore" + str(forename[i]).lower() + "fore"
    family_ID[i] = "fam" + str(family_ID[i]) + "fam"
    parish[i] = "par" + str(parish[i]).lower() + "par"
    age[i] = "a" +  str(age[i]) + "a"
    sex[i] = "sex" + str(sex[i]).lower() + "sex"
    relation_to_head_of_household[i] = "rel" + str(relation_to_head_of_household[i]).lower() + "rel"
    marital_status[i] = "mar" + str(marital_status[i]).lower() + "mar"
    occupation[i] = "occ" + str(occupation[i]).lower() + "occ"
    education[i] = str(education[i]).lower()


array_of_all_data = [surname, forename, family_ID, parish, age, sex, relation_to_head_of_household, marital_status, occupation]
# array_of_all_data = [family_ID, parish, age, sex, relation_to_head_of_household, marital_status, occupation]
#surname = generalisingSmallClasses(surname, "other", 4)

uniqueValues = printUniqueClassess(array_of_all_data, ["surname", "forename", "family_ID", "parish", "age", "sex", "relation_to_head_of_household", "marital_status", "occupation"])
# uniqueValues = printUniqueClassess(array_of_all_data, ["family_ID", "parish", "age", "sex", "relation_to_head_of_household", "marital_status", "occupation"])
print("unique values", len(uniqueValues))


#make education contain only 4 possible numbers - 1 for read and write, 2 for read only, 3 for neither, and 4 for undefined
number_of_each_Class = [0,0,0,0]
labels = [0] * len(education)
for i in range(0, len(education)):
    if (education[i] == "read and write"):
        labels[i] = 1
        number_of_each_Class[0] +=1
    elif (education[i] == "read only"):
        labels[i] = 2
        number_of_each_Class[1] +=1
    elif (education[i] == "neither"):
        labels[i] = 3
        number_of_each_Class[2] +=1
    else:
        labels[i] = 0
        number_of_each_Class[3] +=1

print("count", number_of_each_Class)
labels = np.array(labels)
print("labels", len(labels))

#generate feature vector (pass in a array for each of the raw data columns (should all be the same length))
X = produce_input_feature([surname, forename, family_ID, parish, age, sex, relation_to_head_of_household, marital_status, occupation], uniqueValues)
# X = produce_input_feature([family_ID, parish, age, sex, relation_to_head_of_household, marital_status, occupation], uniqueValues)


# test model
from sklearn.model_selection import train_test_split

indices = np.arange(len(labels))
training_data, test_data = train_test_split(indices, test_size=0.2)

print("---------------------------------------baseline models ----------------------------------------")
randomBaselineClassifier(labels)
# modeBaselineClassifier(labels)

print("-------------------------------------- Linear SVM -----------------------------------------------------")
#linearSVMClassifierCrossValidation(X[training_data], labels[training_data])
optimsedLinearSVMClassifier(X[training_data], labels[training_data], X[test_data], labels[test_data])

print("-------------------------------------- Logistic Regression --------------------------------------------")
#logistic_regression_model_cross_validation(X[training_data], labels[training_data])
train_chosen_logistic_regression_model(X[training_data], labels[training_data], 1, X[test_data], labels[test_data])

print("-------------------------------------- Kernal SVM -----------------------------------------------------")
#kernel_SVM_model_cross_validation(X[training_data], labels[training_data])
train_chosen_kernel_SVM_model(X[training_data], labels[training_data], 1, X[test_data], labels[test_data])

print("-------------------------------------- KNN ------------------------------------------------------------")
#knn_model_cross_validation(X[training_data], labels[training_data])
train_chosen_knn_model(X[training_data], labels[training_data], 1, X[test_data], labels[test_data])


# ------------------------------------------- NOTES ---------------------------------------------------------------
#items which seemed to have biggest effect -> sex, relation to head of household, surname,marital status (kind of), parish kind of
#similar accuracy when trained with sex, relation to head of household, surname, marital status ,parish as all
#just surname does similar
#altering max iterations and C value with all got an accuracy of 68% (66% when ran again)
#was only ever training on approx. 7000 points because of the way I was splitting the data, achived accuracy of 70+ but takes a hell of a lot longer
#73% on just [surname, forename, family_ID, relation_to_head_of_household, sex]
#74% with all
# training on all data now with each person recieving a single sparce vector containing all of their data
