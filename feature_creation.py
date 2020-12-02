import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv("test_1851.csv", comment='#')
surname = df.iloc[:,0]
forename = df.iloc[:,1]
family_ID = df.iloc[:,2]
parish = df.iloc[:,3]
county = df.iloc[:,4]
age = df.iloc[:,5]
sex = df.iloc[:,6]
relation_to_head_of_household = df.iloc[:,7]
marital_status = df.iloc[:,8]
occupation = df.iloc[:,9]
education = df.iloc[:,10] #this is what we are trying to classify

#make education contain only 3 possible numbers - 1 for read and write, 2 for read only, 3 for neither, and 4 for undefined
number_of_each_Class = [0,0,0,0]
for i in range(0, len(education)):
    if (education[i] == "read and write"):
        education[i] = 1
        number_of_each_Class[0] +=1
    elif (education[i] == "read only"):
        education[i] = 0
        number_of_each_Class[1] +=1
    elif (education[i] == "neither"):
        education[i] = 3
        number_of_each_Class[2] +=1
    else:
        education[i] = 4
        number_of_each_Class[3] +=1

print("count", number_of_each_Class)

education = education.astype('int')

#tokenise each of the input fields
from sklearn.feature_extraction.text import CountVectorizer
import math as m
vectorizer = CountVectorizer()

surname_input = vectorizer.fit_transform(surname.astype('U'))
forename_input = vectorizer.fit_transform(forename.astype('U'))
parish_input = vectorizer.fit_transform(parish.astype('U'))
county_input = vectorizer.fit_transform(county.astype('U'))
age_input = vectorizer.fit_transform(age.astype('U'))
sex_input = vectorizer.fit_transform(sex.astype('U'))
relation_to_head_of_household_input = vectorizer.fit_transform(relation_to_head_of_household.astype('U'))
marital_status = vectorizer.fit_transform(marital_status.astype('U'))
occupation_input = vectorizer.fit_transform(occupation.astype('U'))
family_ID_input = vectorizer.fit_transform(family_ID.astype('U')) #!!!!!!!!!!!!!!!!!!!!!!!not working properly for family ID (skipping certain values) !!!!!!!!!!!!

# print("family id values\n",  family_ID_input)

print("family id", family_ID_input.shape)
print("surname", surname_input.shape)
print("forename", forename_input.shape)
print("parish", parish_input.shape)
print("county", county_input.shape)
print("age", age_input.shape)
print("sex", sex_input.shape)
print("relation to head of household", relation_to_head_of_household_input.shape)
print("marital status", marital_status.shape)
print("occupation", occupation_input.shape)

# test for if it worked
# for i in range(0, len(surname)):
#     if(family_ID[i] == 1):
#         print("input ", i, "   ", family_ID_input[i])

#create input vector 
# X = []
# for i in range(0, len(surname)):
#     X.append([surname_input[i],forename_input[i],parish_input[i],county_input[i],age_input[i],sex_input[i],relation_to_head_of_household_input[i],marital_status[[i]],occupation_input[i]])
# X = np.array(X, dtype=object) #!!!!!!!!!!!!! need to convert to array before training a model

#array X is 7465 x 9

#test model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve

indices = np.arange(len(surname))
training_data, test_data = train_test_split(indices, test_size=0.2)

model = LogisticRegression()
model.fit(surname_input[training_data], education[training_data])
label_predictions = model.predict(surname_input[test_data])
correct_predictions = 0
increment = 0
for test_datapoints in test_data:
    if(label_predictions[increment] == education[test_datapoints]):
        correct_predictions += 1
    increment += 1

accuracy = correct_predictions/ np.size(label_predictions)
print("\n\nLogistic regression model metrics")
print("accuracy", accuracy)
LR_confusion_matrix = confusion_matrix(education[test_data], label_predictions)
print("confusion Matrix")
print(LR_confusion_matrix)


# ----------------------------------------------------------------------------------------------------------
#general notes, only 40 ppl not from antrim (makes field kinda usless)