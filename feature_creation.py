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
        education[i] = 2
        number_of_each_Class[1] +=1
    elif (education[i] == "neither"):
        education[i] = 3
        number_of_each_Class[2] +=1
    else:
        education[i] = 4
        number_of_each_Class[3] +=1

print("count", number_of_each_Class)

#tokenise each of the input fields
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

surname_input = vectorizer.fit_transform(surname.astype('U'))
forename_input = vectorizer.fit_transform(forename.astype('U'))
family_ID_input = family_ID
parish_input = vectorizer.fit_transform(parish.astype('U'))
county_input = vectorizer.fit_transform(county.astype('U'))
age_input = age
sex_input = vectorizer.fit_transform(sex.astype('U'))
relation_to_head_of_household_input = vectorizer.fit_transform(relation_to_head_of_household.astype('U'))
marital_status = vectorizer.fit_transform(marital_status.astype('U'))
occupation_input = vectorizer.fit_transform(occupation.astype('U'))


#test for if it worked
# for i in range(0, len(surname)):
#     if(forename[i] == "William"):
#         print("input ", i, "   ", forename_input[i])


#general notes, only 40 ppl not from antrim (makes field kinda usless)