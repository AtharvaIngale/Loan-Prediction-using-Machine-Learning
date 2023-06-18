# -*- coding: utf-8 -*-

from subprocess import call
import tkinter as tk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz



#url = "https://raw.githubusercontent.com/callxpert/datasets/master/Loan-applicant-details.csv"
url = "C:/Users/Sagar/Downloads/100%Loan_prediction/100%Loan_prediction/Loan-applicant-details.csv"
names = ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']
dataset = pd.read_csv(url, names=names)


print(dataset.head(20))

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    dataset[i] = le.fit_transform(dataset[i])
    
    
    
array = dataset.values



X = array[:,6:11]
X = X.astype('int') 
Y = array[:,12]
Y = Y.astype('int') 

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)


from sklearn.svm import SVC
svcclassifier = SVC(kernel='linear')
svcclassifier.fit(x_train, y_train)
y_pred = svcclassifier.predict(x_test)
print(y_pred)

    
print("=" * 40)
print("==========")
print("Classification Report : ",(classification_report(y_test, y_pred)))
print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
ACC = (accuracy_score(y_test, y_pred) * 100)
repo = (classification_report(y_test, y_pred))
    
#label4 = tk.Label(root,text =str(repo),width=45,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
#label4.place(x=205,y=200)
    
#label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as HEART_DISEASE_MODEL.joblib",width=45,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
#label5.place(x=205,y=420)
#from joblib import dump
#dump (svcclassifier,"HEART_DISEASE_MODEL.joblib")
#print("Model saved as HEART_DISEASE_MODEL.joblib")


# model = DecisionTreeClassifier()
# model.fit(x_train,y_train)
# predictions = model.predict(x_test)
# print(accuracy_score(y_test, predictions))