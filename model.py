#!/usr/bin/env python
# coding: utf-8

# # Step 1 :  Import Library and Dataset

# In[1]:


import pandas as pd
import numpy as np



# Read the data in
employee = pd.read_csv(r"C:\Users\USER\Desktop\Python Code\Decision Tree by Irfan\churn.csv")

# ### Removing Irrelavent Variable
employee = employee.drop(['customerID'],axis=1)
employee.columns


#Replacing spaces with null values in total charges column
employee['TotalCharges'] =employee["TotalCharges"].replace(" ",np.nan).astype(float) 
# string cannot be convert float direclty 

employee.TotalCharges.fillna(employee.TotalCharges.mean(),inplace=True) # one column at a time bb


# #Employee Numeric columns
employee_num = employee[employee.select_dtypes(include=[np.number]).columns.tolist()]

employee_dummies = employee[employee.select_dtypes(include=['object']).columns.tolist()]

from sklearn.preprocessing import LabelEncoder
employee_dummies=employee_dummies.apply(LabelEncoder().fit_transform)

employee_combined = pd.concat([employee_num, employee_dummies],axis=1)


# # Step 3: Data Partition

#Dividing data into train and test dataset
from sklearn.model_selection import train_test_split
train_x = employee_combined.drop(['Churn'],axis=1)
train_y = employee_combined['Churn']

# Train test split

X_train, X_test, y_train, y_test =train_test_split(train_x,train_y,test_size=0.3,random_state=231)


#Import Tree Classifier model
from sklearn import tree

dt = tree.DecisionTreeClassifier(criterion='gini',
                                 min_samples_leaf=40,
                                 min_samples_split=100,
                                 max_depth=4)
#Train the model using the training sets
dt.fit(X_train,y_train)


import pickle
# Saving model
pickle.dump(dt, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

