import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

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
for i in range(0, 8):
    temp.extend(labels)
labels = np.array(temp)


#---------------------------------------------------------------- begining of function --------------------------------------------------------------------
#create a single list containing all words used as inputs
all_fields = []
all_fields.extend(surname.astype('U'))
all_fields.extend(forename.astype('U'))
all_fields.extend(family_ID.astype('U'))
all_fields.extend(parish.astype('U'))
all_fields.extend(age.astype('U'))
all_fields.extend(sex.astype('U'))
all_fields.extend(relation_to_head_of_household.astype('U'))
all_fields.extend(marital_status.astype('U'))
all_fields.extend(occupation.astype('U'))

#tokenise each of the input fields
from sklearn.feature_extraction.text import CountVectorizer
import math as m
vectorizer = CountVectorizer()
vectorizer.fit(all_fields)

surname_input = vectorizer.transform(surname.astype('U')).toarray()
forename_input = vectorizer.transform(forename.astype('U')).toarray()
parish_input = vectorizer.transform(parish.astype('U')).toarray()
age_input = vectorizer.transform(age.astype('U')).toarray()
sex_input = vectorizer.transform(sex.astype('U')).toarray()
relation_to_head_of_household_input = vectorizer.transform(relation_to_head_of_household.astype('U')).toarray()
marital_status_input = vectorizer.transform(marital_status.astype('U')).toarray()
occupation_input = vectorizer.transform(occupation.astype('U')).toarray()
family_ID_input = vectorizer.transform(family_ID.astype('U')).toarray() #!!!!!!!!!!!!!!!!!!!!!!!not working properly for family ID (skipping certain values) !!!!!!!!!!!!

print("\nfamily id", family_ID_input.shape)
print("surname", surname_input.shape)
print("forename", forename_input.shape)
print("parish", parish_input.shape)
print("age", age_input.shape)
print("sex", sex_input.shape)
print("relation to head of household", relation_to_head_of_household_input.shape)
print("marital status", marital_status_input.shape)
print("occupation", occupation_input.shape)
print("education", education.shape)
print("\n\nlables", len(labels))
# they are all of type class 'scipy.sparse.csr.csr_matrix'

# test for if it worked
# for i in range(0, len(surname)):
#     if(family_ID[i] == 1):
#         print("input ", i, "   ", family_ID_input[i])
        
# # create input vector
X = []
# for i in range(0, len(surname)):
#     X.append([surname_input[i], forename_input[i], parish_input[i], age_input[i], sex_input[i], relation_to_head_of_household_input[i], marital_status_input[i], occupation_input[i]])
X.extend(surname_input)
X.extend(forename_input)
X.extend(parish_input)
X.extend(age_input)
X.extend(sex_input)
X.extend(relation_to_head_of_household_input)
X.extend(marital_status_input)
X.extend(occupation_input)

X = np.array(X)
print("shape of X", X.shape)
#---------------------------------------------------------------- end of function --------------------------------------------------------------------
# array X is 7465 x 9

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