import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image , ImageTk 
from tkinter.filedialog import askopenfilename
#import cv2
import numpy as np
import time
import sqlite3
#import tfModel_test as tf_test
global fn
fn=""
##############################################+=============================================================
root = tk.Tk()
root.configure(background="seashell2")
#root.geometry("1300x700")


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("MAIN PAGE")


#430
#++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 =Image.open('h3.jpg')
image2 =image2.resize((750,890), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=850, y=0) #, relwidth=1, relheight=1)
#


#frame_display = tk.LabelFrame(root, text=" --Display-- ", width=900, height=250, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display.grid(row=0, column=0, sticky='nw')
#frame_display.place(x=300, y=100)

#frame_display1 = tk.LabelFrame(root, text=" --Result-- ", width=900, height=200, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display1.grid(row=0, column=0, sticky='nw')
#frame_display1.place(x=300, y=430)

#frame_display2 = tk.LabelFrame(root, text=" --Calaries-- ", width=900, height=50, bd=5, font=('times', 14, ' bold '),bg="lightblue4")
#frame_display2.grid(row=0, column=0, sticky='nw')
#frame_display2.place(x=300, y=380)

frame_alpr = tk.LabelFrame(root,width=850, height=1850, bd=5, font=('times', 14, 'bold'),bg="#152238")
frame_alpr.grid(row=0, column=0)
frame_alpr.place(x=0, y=0)

image2 =Image.open('logo.png')
image2 =image2.resize((140,120), Image.ANTIALIAS)

background_image=ImageTk.PhotoImage(image2)
background_label = tk.Label(frame_alpr, image=background_image)

background_label.image = background_image

background_label.place(x=350, y=25)

lbl = tk.Label(frame_alpr, text="Loan Prediction", font=('Times New Roman', 30,' bold '),bg="white",fg="black")
lbl.place(x=250, y=180)

lbl = tk.Label(frame_alpr, text=" System", font=('Times New Roman', 30,' bold '),bg="white",fg="black")
lbl.place(x=310, y=250)

logo_label=tk.Label()
logo_label.place(x=90,y=55)


logo_label1=tk.Label(text='To develop a system that capable \n Identify the type of Loan Prediction...',compound='bottom',font=("Times New Roman", 20, 'bold', 'italic'),width=35, bg="white", fg="black")
#logo_label=tk.Label(height=500, width=400)
logo_label1.place(x=130,y=590)


def login():

    from subprocess import call
    call(["python", "GUI_main.py"])  
   



#################################################################################################################


# def register():

#     from subprocess import call
#     call(["python", "registration.py"])  
   
def window():
    root.destroy()

button1 = tk.Button(frame_alpr, text=" START",command=login,width=15, height=1, font=('times', 15, ' bold '),bg="skyblue",fg="black")
button1.place(x=300, y=450)

# button2 = tk.Button(frame_alpr, text="LOGIN",command=login,width=15, height=1, font=('times', 15, ' bold '),bg="#3BB9FF",fg="black")


# button2.place(x=200, y=450)

# button3 = tk.Button(frame_alpr, text="Train Model", command=train_model, width=12, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
# button3.place(x=10, y=160)
#




root.mainloop()