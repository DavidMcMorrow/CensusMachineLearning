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

    feature_inputs = []
    i = 0
    for input in list_of_inputs:
        feature_inputs.append(vectorizer.transform(surname.astype('U')).toarray())
        print(vectorizer.transform(surname.astype('U')).toarray().shape)
        i += 1

    X = []
    for input in feature_inputs:
        X.extend(input)

    X = np.array(X)
    print("shape of X", X.shape)

    return X

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

#make the number of labels match the size of the flattened input
temp = []
for i in range(0, 9):
    temp.extend(labels)
labels = np.array(temp)
print("labels", len(labels))

#generate feature vector (pass in a array for each of the raw data columns (should all be the same length))
X = produce_input_feature([surname, forename, family_ID, parish, age, sex, relation_to_head_of_household, marital_status, occupation])

# test model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve

indices = np.arange(len(surname))
training_data, test_data = train_test_split(indices, test_size=0.2)

model = LogisticRegression(max_iter=10000, C=1000, penalty="l2")
model.fit(X[training_data], labels[training_data])
label_predictions = model.predict(X[test_data])
correct_predictions = 0
increment = 0
for test_datapoints in test_data:
    if(label_predictions[increment] == labels[test_datapoints]):
        correct_predictions += 1
    increment += 1

accuracy = correct_predictions/ np.size(label_predictions)
print("\n\nLogistic regression model metrics")
print("accuracy", accuracy)
LR_confusion_matrix = confusion_matrix(labels[test_data], label_predictions)
print("confusion Matrix")
print(LR_confusion_matrix)


# ----------------------------------------------------------------------------------------------------------
#items which seemed to have biggest effect -> sex, relation to head of household, surname,marital status (kind of), parish kind of
#similar accuracy when trained with sex, relation to head of household, surname, marital status ,parish as all
#just surname does similar
#altering max iterations and C value with all got an accuracy of 68% (66% when ran again)