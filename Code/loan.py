from subprocess import call
import tkinter as tk
import tkinter as tk
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

root = tk.Tk()
root.title("loan Prediction")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
# ++++++++++++++++++++++++++++++++++++++++++++

image2 = Image.open('heart.jpg')

image2 = image2.resize((w, h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)


background_label = tk.Label(root, image=background_image)
background_label.image = background_image



background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)
lbl = tk.Label(root, text="Loan Prediction System", font=('times', 35,' bold '), height=1, width=32,bg="violet Red",fg="Black")
lbl.place(x=300, y=10)
# _+++++++++++++++++++++++++++++++++++++++++++++++++++++++
data = pd.read_csv("train.csv")



data = data.dropna()

le = LabelEncoder()
data['Loan_Status'] = le.fit_transform(data['Loan_Status'])

data['Gender'] = le.fit_transform(data['Gender'])
data['Married'] = le.fit_transform(data['Married'])
data['Education'] = le.fit_transform(data['Education'])
data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
data['Property_Area'] = le.fit_transform(data['Property_Area'])

data.head()

"""Feature Selection => Manual"""
x = data.drop(['Loan_Status', 'Loan_ID'], axis=1)


def Data_Preprocessing():
    data = pd.read_csv("train.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    data['Loan_Status'] = le.fit_transform(data['Loan_Status'])
    print(data['Credit_History'])
    data['Gender'] = le.fit_transform(data['Gender'])
    print("Gender Encoding")
    data['Married'] = le.fit_transform(data['Married'])
    data['Education'] = le.fit_transform(data['Education'])
    data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
    data['Property_Area'] = le.fit_transform(data['Property_Area'])

    



    data['Gender'] = le.fit_transform(data['Gender'])
    data['Married'] = le.fit_transform(data['Married'])
    data['Education'] = le.fit_transform(data['Education'])
    data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
    data['Property_Area'] = le.fit_transform(data['Property_Area'])




    """Feature Selection => Manual"""
    x = data.drop(['Loan_Status', 'Loan_ID'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Loan_Status']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    

    load = tk.Label(root, font=("Tempus Sans ITC", 15, "bold"), width=50, height=2, background="green",
                    foreground="white", text="Data Loaded=>Splitted into 80% for Training & 20% for Testing")
    load.place(x=200, y=80)


def Model_Training():
    data = pd.read_csv("Ctrain.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    data['Loan_Status'] = le.fit_transform(data['Loan_Status'])
    print(data['Credit_History'])
    data['Gender'] = le.fit_transform(data['Gender'])
    print("Gender Encoding")
    data['Married'] = le.fit_transform(data['Married'])
    data['Education'] = le.fit_transform(data['Education'])
    data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
    data['Property_Area'] = le.fit_transform(data['Property_Area'])
    
       
          
    data['Gender'] = le.fit_transform(data['Gender'])
    data['Married'] = le.fit_transform(data['Married'])
    data['Education'] = le.fit_transform(data['Education'])
    data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
    data['Property_Area'] = le.fit_transform(data['Property_Area'])

    """Feature Selection => Manual"""
    x = data.drop(['Loan_Status', 'Loan_ID'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Loan_Status']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=2)

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
    
    label4 = tk.Label(root,text =str(repo),width=45,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=305,y=200)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as Loan_Prediction_MODEL.joblib",width=45,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=305,y=420)
    from joblib import dump
    dump (svcclassifier,"Loan_Prediction_MODEL.joblib")
    print("Model saved as Loan_Prediction_MODEL.joblib")

def Model_Training1():
    data = pd.read_csv("train.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    data['Loan_Status'] = le.fit_transform(data['Loan_Status'])
    print(data['Credit_History'])
    data['Gender'] = le.fit_transform(data['Gender'])
    print("Gender Encoding")
    data['Married'] = le.fit_transform(data['Married'])
    data['Education'] = le.fit_transform(data['Education'])
    data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
    data['Property_Area'] = le.fit_transform(data['Property_Area'])
    
       
          
    data['Gender'] = le.fit_transform(data['Gender'])
    data['Married'] = le.fit_transform(data['Married'])
    data['Education'] = le.fit_transform(data['Education'])
    data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
    data['Property_Area'] = le.fit_transform(data['Property_Area'])

    """Feature Selection => Manual"""
    x = data.drop(['Loan_Status', 'Loan_ID'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Loan_Status']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=2)

    from sklearn.ensemble import RandomForestClassifier  
    svcclassifier =RandomForestClassifier(n_estimators= 10, criterion="entropy") 
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
    
    label4 = tk.Label(root,text =str(repo),width=45,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=305,y=200)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as RF_MODEL.joblib",width=45,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=305,y=420)
    from joblib import dump
    dump (svcclassifier,"RF_MODEL.joblib")
    print("Model saved as RF_MODEL.joblib")


def Model_Training2():
    data = pd.read_csv("train.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    data['Loan_Status'] = le.fit_transform(data['Loan_Status'])
    print(data['Credit_History'])
    data['Gender'] = le.fit_transform(data['Gender'])
    print("Gender Encoding")
    data['Married'] = le.fit_transform(data['Married'])
    data['Education'] = le.fit_transform(data['Education'])
    data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
    data['Property_Area'] = le.fit_transform(data['Property_Area'])
    
       
          
    data['Gender'] = le.fit_transform(data['Gender'])
    data['Married'] = le.fit_transform(data['Married'])
    data['Education'] = le.fit_transform(data['Education'])
    data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
    data['Property_Area'] = le.fit_transform(data['Property_Area'])

    """Feature Selection => Manual"""
    x = data.drop(['Loan_Status', 'Loan_ID'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Loan_Status']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=2)

    from sklearn.tree import DecisionTreeClassifier 
    svcclassifier = DecisionTreeClassifier(criterion='entropy', random_state=0)  
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
    
    label4 = tk.Label(root,text =str(repo),width=45,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=305,y=200)
    
    label5 = tk.Label(root,text ="Accuracy : "+str(ACC)+"%\nModel saved as DT_MODEL.joblib",width=45,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=305,y=420)
    from joblib import dump
    dump (svcclassifier,"DT_MODEL.joblib")
    print("Model saved as DT_MODEL.joblib")
def call_file():
    #import Check_loan
    #Check_loan.Train()
    root1 = tk.LabelFrame(root, text=" -Check Loan -- ", width=660, height=630, bd=5, font=('times', 14, ' bold '),bg="SeaGreen1")
    root1.grid(row=0, column=0, sticky='nw')
    root1.place(x=500, y=80)
    Gender = tk.StringVar()
    Married = tk.StringVar()
    Dependents = tk.IntVar()
    Education = tk.StringVar()
    Self_Employed = tk.StringVar()
    ApplicantIncome = tk.IntVar()
    CoapplicantIncome = tk.IntVar()
    LoanAmount = tk.IntVar()
    Loan_Amount_Term = tk.IntVar()
    Credit_History = tk.IntVar()
    Property_Area = tk.StringVar()
    
    def Detect():
        e1=Gender.get()
        print(e1)
        e2=Married.get()
        print(e2)
        #b1=Lb1.get(Lb1.curselection())
        #e3.set(b1) 
        #value = Lb1.get(Lb1.curselection())
        #e3.set(value)  
        e3=Dependents.get()
        print(e3)
        #print(type(e3))
        e4=Education.get()
        print(e4)
        e5=Self_Employed.get()
        print(e5)
        e6=ApplicantIncome.get()
        print(e6)
        e7=CoapplicantIncome.get()
        print(e7)
        e8=LoanAmount.get()
        print(e8)
        e9=Loan_Amount_Term.get()
        print(e9)
        e10=Credit_History.get()
        print(e10)
        e11=Property_Area.get()
        print(e11)
   
        #########################################################################################
        
        from joblib import dump , load
        a1=load('Loan_Prediction_MODEL.joblib')#File chnge  RF
        #a1=load('C:/Users/rutik/Desktop/100%Loan_prediction/100%Loan_prediction/RF_MODEL.joblib')#
        #a1=load('C:/Users/rutik/Desktop/100%Loan_prediction/100%Loan_prediction/DT_MODEL.joblib')#
        v= a1.predict([[e1, e2, e3, e4, e5, e6, e7, e8, e9,e10, e11]])
        print(v)
        if v[0]==1:
            print("Y")
            yes = tk.Label(root,text="Person is eligible for Loan\n",background="green",foreground="white",width=30,height=2,font=('times', 20, ' bold '))
            yes.place(x=10,y=600)
            
        else:
            print("N")
            no = tk.Label(root, text="Person is not eligible for Loan", background="red", foreground="white",width=30,height=2,font=('times', 20, ' bold '))
            no.place(x=10, y=600)
           
                                                                                            

    l1=tk.Label(root1,text="Gender",background="purple",font=('times', 20, ' bold '),width=15)
    l1.place(x=5,y=1)
    R1 = Radiobutton(root1, text="Male",font=('times', 20, ' bold '), variable=Gender, value=1).place(x=250,y=1)
    R2 = Radiobutton(root1, text="Female",font=('times', 20, ' bold '), variable=Gender, value=2).place(x=350,y=1)
    #Gender=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Gender)
    #Gender.place(x=200,y=1)

    l2=tk.Label(root1,text="Married",background="purple",font=('times', 20, ' bold '),width=15)
    l2.place(x=5,y=50)
    R3 = Radiobutton(root1, text="Yes",font=('times', 20, ' bold '), variable=Married, value=1).place(x=250,y=50)
    R4 = Radiobutton(root1, text="No",font=('times', 20, ' bold '), variable=Married, value=2).place(x=350,y=50)
    #Married=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Married)
    #Married.place(x=200,y=50)

    l3=tk.Label(root1,text="Dependents",background="purple",font=('times', 20, ' bold '),width=15)
    l3.place(x=5,y=100)
    Dependents=tk.Entry(root1,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Dependents)
    Dependents.place(x=250,y=100)
    
    
    #Lb1 = Listbox(root,width=20,height=3)
    #Lb1.place(x=200,y=100)
    #Lb1.insert(1, "1")
    #Lb1.insert(2, "2")
    #Lb1.insert(3, "3")
    #chest_pain=Lb1.curselection()
    #Lb1.pack()
    #R1 = Radiobutton(root, text="Typical", variable=chest_pain, value=1).place(x=200,y=100)
    #R2 = Radiobutton(root, text="asymptomatic", variable=chest_pain, value=2).place(x=200,y=120)
    #R3 = Radiobutton(root, text="nontypical", variable=chest_pain, value=3).place(x=200,y=140)

    l4=tk.Label(root1,text="Education",background="purple",font=('times', 20, ' bold '),width=15)
    l4.place(x=5,y=150)
    R1 = Radiobutton(root1, text="Graduate",font=('times', 20, ' bold '), variable=Education, value=1).place(x=250,y=150)
    R2 = Radiobutton(root1, text="Under Graduate",font=('times', 20, ' bold '), variable=Education, value=2).place(x=400,y=150)
    
    #Education=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Education)
    #Education.place(x=200,y=160)

    l5=tk.Label(root1,text="Self Employed",background="purple",font=('times', 20, ' bold '),width=15)
    l5.place(x=5,y=200)
    R1 = Radiobutton(root1, text="Yes",font=('times', 20, ' bold '), variable=Self_Employed, value=1).place(x=250,y=200)
    R2 = Radiobutton(root1, text="No",font=('times', 20, ' bold '), variable=Self_Employed, value=2).place(x=350,y=200)
    #Self_Employed=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Self_Employed)
    #Self_Employed.place(x=200,y=200)

    l6=tk.Label(root1,text="ApplicantIncome",background="purple",font=('times', 20, ' bold '),width=15)
    l6.place(x=5,y=250)
    ApplicantIncome=tk.Entry(root1,bd=2,width=5,font=("TkDefaultFont", 20),textvar=ApplicantIncome)
    ApplicantIncome.place(x=250,y=250)

    l7=tk.Label(root1,text="CoapplicantIncome",background="purple",font=('times', 20, ' bold '),width=15)
    l7.place(x=5,y=300)
    CoapplicantIncome=tk.Entry(root1,bd=2,width=5,font=("TkDefaultFont", 20),textvar=CoapplicantIncome)
    CoapplicantIncome.place(x=250,y=300)

    l8=tk.Label(root1,text="Loan Amount",background="purple",font=('times', 20, ' bold '),width=15)
    l8.place(x=5,y=350)
    LoanAmount=tk.Entry(root1,bd=2,width=5,font=("TkDefaultFont", 20),textvar=LoanAmount)
    LoanAmount.place(x=250,y=350)

    l9=tk.Label(root1,text="Loan Amount Term",background="purple",font=('times', 20, ' bold '),width=15)
    l9.place(x=5,y=400)
    Loan_Amount_Term=tk.Entry(root1,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Loan_Amount_Term)
    Loan_Amount_Term.place(x=250,y=400)

    l10=tk.Label(root1,text="Credit History",background="purple",font=('times', 20, ' bold '),width=15)
    l10.place(x=5,y=450)
    Credit_History=tk.Entry(root1,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Credit_History)
    Credit_History.place(x=250,y=450)

    l11=tk.Label(root1,text="Property Area",background="purple",font=('times', 20, ' bold '),width=15)
    l11.place(x=5,y=500)
    R1 = Radiobutton(root1, text="Urban",font=('times', 20, ' bold '), variable=Property_Area, value=1).place(x=250,y=500)
    R2 = Radiobutton(root1, text="Semiurban",font=('times', 20, ' bold '), variable=Property_Area, value=2).place(x=370,y=500)
    R3 = Radiobutton(root1, text="Rural",font=('times', 20, ' bold '), variable=Property_Area, value=3).place(x=550,y=500)
    #Property_Area=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Property_Area)
    #Property_Area.place(x=200,y=500)

    button1 = tk.Button(root1,text="Submit",command=Detect,font=('times', 15, ' bold '),width=10,bg='red')
    button1.place(x=250,y=550)


def Data_Display():
    columns = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
               'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
               'Credit_History','Property_Area']

    data1 = pd.read_csv('loan_prediction.csv')

    data1.shape

    data1.shape

    data1.head()

    data1

    data1

    Loan_ID = data1.ix[:, 1]
    Gender = data1.ix[:, 2]
    Married = data1.ix[:, 3]
    Dependents = data1.ix[:, 4]
    Education = data1.ix[:, 5]
    Self_Employed = data1.ix[:, 6]
    ApplicantIncome = data1.ix[:, 7]
    CoapplicantIncome = data1.ix[:, 8]
    LoanAmount = data1.ix[9]
    Loan_Amount_Term = data1.ix[10]
    Credit_History = data1.ix[11]
    Property_Area = data1.ix[12]

    display = tk.LabelFrame(root, width=100, height=400, )
    display.place(x=200, y=100)

    tree = ttk.Treeview(display, columns=(
    'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
              'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
               'Credit_History','Property_Area'))

    style = ttk.Style()
    style.configure('Treeview', rowheight=50)
    style.configure("Treeview.Heading", font=("Tempus Sans ITC", 15, "bold italic"))
    style.configure(".", font=('Helvetica', 15), background="blue")
    style.configure("Treeview", foreground='white', background="black")

    tree["columns"] = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12")
    tree.column("1", width=50)
    tree.column("2", width=50)
    tree.column("3", width=50)
    tree.column("4", width=50)
    tree.column("5", width=50)
    tree.column("6", width=50)
    tree.column("7", width=50)
    tree.column("8", width=50)
    tree.column("9", width=50)
    tree.column("10", width=50)
    tree.column("11", width=50)
    tree.column("12", width=50)

    tree.heading("1", text="Loan_ID")
    tree.heading("2", text="Gender")
    tree.heading("3", text="Married")
    tree.heading("4", text="Dependents")
    tree.heading("5", text="Education")
    tree.heading("6", text="Self_Employed")
    tree.heading("7", text="ApplicantIncome")
    tree.heading("8", text="CoapplicantIncome")
    tree.heading("9", text="LoanAmount")
    tree.heading("10", text="Loan_Amount_Term")
    tree.heading("11", text="Credit_History")
    tree.heading("12", text="Property_Area")

    treeview = tree

    tree.grid(row=0, column=0, sticky=tk.NSEW)

    print("Data Displayed")
    for i in range(0, 304):
        tree.insert("", 'end', values=(
        Loan_ID[i], Gender[i], Married[i], Dependents[i], Education[i], Self_Employed[i], ApplicantIncome[i], CoapplicantIncome[i], LoanAmount[i], Loan_Amount_Term[i],
        Credit_History[i], Property_Area[i]))
        i = i + 1


check = tk.Frame(root, w=100)
check.place(x=700, y=100)


def window():
    root.destroy()

button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Data_Preprocessing", command=Data_Preprocessing, width=15, height=2)
button2.place(x=5, y=90)

button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model Train SVM", command=Model_Training, width=15, height=2)
button3.place(x=5, y=170)

button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model Train RF", command=Model_Training1, width=15, height=2)
button4.place(x=5, y=250)

button6 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model Train DT", command=Model_Training2, width=15, height=2)
button6.place(x=5, y=330)

button5 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Loan Prediction", command=call_file, width=15, height=2)
button5.place(x=5, y=430)
exit = tk.Button(root, text="Exit", command=window, width=15, height=2, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=5, y=530)

root.mainloop()

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''