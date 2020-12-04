import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import math as m

def produce_input_feature(list_of_inputs):

    #create a single list containing all words used as inputs
    all_fields = []
    for input in list_of_inputs:
        all_fields.extend(input.astype('U'))

    #tokenise each of the input fields
    vectorizer = CountVectorizer()
    vectorizer.fit(all_fields)

    people = []
    for i in range(0, len(list_of_inputs[0])):
        person = []
        for input in list_of_inputs:
            person.append(str(input[i]))
            person.append(" ")
        people.append(("".join(person)))

    # print("people", len(people), "person", len(person))
    # print(people[0], "\n",people[5], "\n",people[10],)

    feature_inputs = []
    for person in people:
        feature_inputs.append(vectorizer.transform([person]).toarray())

    X = []
    for input in feature_inputs:
        X.append(input[0])

    X = np.array(X)
    print("shape of X", X.shape)

    return X

def printUniqueClassess(classes):
    for i in classes:
        print("unique", len(list(set(i))))

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


#printUniqueClassess([surname, forename])
printUniqueClassess([surname, forename, family_ID, parish, age, sex, relation_to_head_of_household, marital_status, occupation])

#print("unique", len(list(set(surname))))

occupation = generalisingSmallClasses(occupation, "other", 50)
#surname = generalisingSmallClasses(surname, "other", 4)

print("len(list(set(surname)))", len(list(set(surname))))
print("len(list(set(forename)))", len(list(set(forename))))
print("len(list(set(family_ID)))", len(list(set(family_ID))))
print("len(list(set(age)))", len(list(set(age))))
print("len(list(set(sex)))", len(list(set(sex))))
print("len(list(set(relation_to_head_of_household)))", len(list(set(relation_to_head_of_household))))
print("len(list(set(marital_status)))", len(list(set(marital_status))))
print("len(list(set(occupation)))", len(list(set(occupation))))
print("len(list(set(education)))", len(list(set(education))))


#make education contain only 3 possible numbers - 1 for read and write, 2 for read only, 3 for neither, and 4 for undefined
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

X = produce_input_feature([surname, forename, family_ID, parish, age, sex, relation_to_head_of_household, marital_status, occupation])
# test model
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, roc_curve
# 
# indices = np.arange(len(labels))
# training_data, test_data = train_test_split(indices, test_size=0.2)
# 
# model = LogisticRegression(max_iter=10000, C=100, penalty="l2")
# model.fit(X[training_data], labels[training_data])
# label_predictions = model.predict(X[test_data])
# correct_predictions = 0
# increment = 0
# for test_datapoints in test_data:
    # if(label_predictions[increment] == labels[test_datapoints]):
        # correct_predictions += 1
    # increment += 1
# 
# accuracy = correct_predictions/ np.size(label_predictions)
# print("\n\nLogistic regression model metrics")
# print("accuracy", accuracy)
# LR_confusion_matrix = confusion_matrix(labels[test_data], label_predictions)
# print("confusion Matrix")
# print(LR_confusion_matrix)
<<<<<<< HEAD
# 
=======

>>>>>>> 7845d47f652a2f012a6b49b59d9ae279422d7c11

# ----------------------------------------------------------------------------------------------------------
#items which seemed to have biggest effect -> sex, relation to head of household, surname,marital status (kind of), parish kind of
#similar accuracy when trained with sex, relation to head of household, surname, marital status ,parish as all
#just surname does similar
#altering max iterations and C value with all got an accuracy of 68% (66% when ran again)
#was only ever training on approx. 7000 points because of the way I was splitting the data, achived accuracy of 70+ but takes a hell of a lot longer
#73% on just [surname, forename, family_ID, relation_to_head_of_household, sex]
# 74% with all
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#created a new issue, this is defenitly not the way to go, think I have to create a single sparce vector for each person (each on will have 9 1's/ #of features used)


