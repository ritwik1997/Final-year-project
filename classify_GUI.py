# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:07:47 2019

@author: rrajpuro
"""

from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import threading
import os 
import v3_classification as RIC
from time import time

bgColor = '#DCDCDC'#'#EBEAEB'
options = ["SVM", "KNN", "RFC"]
optionMan = ['Isomap100','LLE600','SpectralEmbedding100','tSNE3']

class GUI:
    def __init__(self,win):

        self.win = win
        self.status_message = "Classifier Ready"
        self.error_flag = False
        self.warning_flag = False
        self.warning_msg = ""
        self.error_type = ""
        self.error_msg = ""
        self.execution_success = False
        self.display_count = 0

        self.win.title("Robust Image Classification")
        self.win.configure(bg = bgColor)
        self.win.geometry("800x600")
        self.win.resizable(width = True, height = True)

        frame1 = Frame(win, bg=bgColor)
        frame1.pack()

        Label(frame1, text="RIC", font=("arial bold", 30), bg=bgColor).grid(row=0, column=0, columnspan=1)

        Label(self.win, text = 'Choose the Manifold Learning', bg = bgColor ).pack()
        self.manifold = StringVar(self.win)
        self.dropDown = OptionMenu(self.win, self.manifold, *optionMan).pack()
        
        Label(self.win, text = 'Choose the classifier', bg = bgColor ).pack()
        self.classifier = StringVar(self.win)
        self.dropDown = OptionMenu(self.win, self.classifier, *options).pack()

        frame2 = Frame(win, pady=20, bg=bgColor)  # Row of buttons
        frame2.pack()
        b1 = Button(frame2, text="Run", command=self.load_run)
        b1.pack(side=LEFT, padx=10)
        
        # global config_template_label
        frame3 = Frame(win, bg=bgColor)
        Label(frame3, text="Choose Image to be classified", bg=bgColor).grid(row=1, column=0, sticky=E)
        button1 = Button(frame3, text="Browse", command=self.open_test_img)
        button1.grid(row=1, column=3, sticky=W)
        global template_label
        template_label = Label(frame3, bg=bgColor, height=2, justify=LEFT, wraplength=390)
        template_label.grid(row=2, column=0, columnspan=2, pady=5)
        button2 = Button(frame3, text="Classify", command=self.classify_run)
        button2.grid(row=3, column=2, sticky=W)
        frame3.pack()

        frame4 = Frame(win, pady=20, bg=bgColor)  # select of names
        frame4.pack(fill=BOTH)
        global status_label
        status_label = Label(frame4, bd=1, relief=SUNKEN, anchor=W, height=4,wraplength=350, justify=LEFT)
        status_label.pack(fill=BOTH, expand=YES)

    def callback(self):
        #print(self.status_message)
        #display_count = 0
        status_label.configure(text=self.status_message)
        if self.error_flag:
            messagebox.showerror(self.error_type, "Process Terminated!!\n" + self.error_msg)
            self.error_flag = False
            self.error_msg = ""
            self.error_type = ""
            self.status_message = "Classifier Ready"
            self.warning_msg = ""
            self.warning_flag = False
        if self.warning_flag:
            messagebox.showwarning("WARNING",self.warning_msg)
            self.warning_msg = ""
            self.warning_flag = False
        if self.execution_success:
            self.display_count+=1
            if self.display_count == 6:
                self.display_count = 0
                self.execution_success = False
                self.status_message = "Classifier Ready"
        self.win.after(2000, self.callback)
        
    def load_func(self):
        try:
            start = time()
            global clf,X_test,testImages
            manType = self.manifold.get()
            X_train, X_test, y_train, y_test, testImages = RIC.load(manType)
            if self.classifier.get()=='SVM':
                clf = RIC.svmFit(X_train,y_train)            
            elif self.classifier.get()=='KNN':
                clf = RIC.knnFit(X_train,y_train)           
            elif self.classifier.get()=='RFC':
                clf = RIC.rfcFit(X_train,y_train)
            end = time()
            self.execution_success =True
            self.status_message = "Images are ready to be classified\nTime taken to load is "+ str(end-start) +" seconds"
        except:
            self.error_flag = True
            self.error_msg = "OOPS!!!\n I have encountered an unknown issue. Contact the developers to fix it!"
            self.error_type = "UNKNOWN ERROR"
            #err_hand.default_error()
    
    def classify_func(self):
        try:
            start = time()
            category = RIC.classify(os.path.basename(img_path),clf,testImages,X_test)
            end = time()
            self.execution_success =True
            self.status_message = "The Image was Succesfully classified in "+str(end-start)+" seconds\nIt is of type "+str(category)
#            self.status_message = RIC.svmFit()

        except:
            self.error_flag = True
            self.error_msg = "OOPS!!!\n I have encountered an unknown issue. Contact the developers to fix it!"
            self.error_type = "UNKNOWN ERROR"
            #err_hand.default_error()

    def load_run(self):
        self.status_message = "Executing algorithms.\nPlease Wait"
        t1 = threading.Thread(target = self.load_func)
        #t1.daemon = True
        t1.start()
        return()

    def classify_run(self):
        self.status_message = "Executing algorithms.\nPlease Wait"
        t1 = threading.Thread(target = self.classify_func)
        #t1.daemon = True
        t1.start()
        return()

    def open_test_img(self):
        global img_path
        img_path = filedialog.askopenfilename(title="Select Test Image")
        template_label.configure(text = img_path)
        return(img_path)

if __name__ == '__main__':
    root = Tk()
    RIC_GUI = GUI(root)
    RIC_GUI.callback()
    root.mainloop()