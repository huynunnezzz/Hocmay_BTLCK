from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# Loading the dataset
data = pd.read_csv('healthcare-dataset-stroke-data1.csv')
dt_Train, dt_Test = train_test_split(data, test_size=0.3 , shuffle = True)


form = Tk()
form.title("Dự đoán khả năng đột quỵ của người:")
form.geometry("1000x500")

lable_title = Label(form, text="Nhập thông tin:", font=("Arial Bold", 10), fg="red")
lable_title.grid(row=1, column=1)

lable_Gender = Label(form, text="Gender:")
lable_Gender.grid(row=2, column=1)
textbox_Gender = Entry(form)
textbox_Gender.grid(row=2, column=2)

lable_Age = Label(form, text="Age:")
lable_Age.grid(row=3, column=1)
textbox_Age = Entry(form)
textbox_Age.grid(row=3, column=2)

lable_Hypertension = Label(form, text="Hypertension:")
lable_Hypertension.grid(row=4, column=1)
textbox_Hypertension = Entry(form)
textbox_Hypertension.grid(row=4, column=2)

lable_Heartdisease = Label(form, text="Heartdisease:")
lable_Heartdisease.grid(row=5, column=1)
textbox_Heartdisease = Entry(form)
textbox_Heartdisease.grid(row=5, column=2)

lable_Evermarried = Label(form, text="Evermarried:")
lable_Evermarried.grid(row=6, column=1)
textbox_Evermarried = Entry(form)
textbox_Evermarried.grid(row=6, column=2)

lable_Worktype = Label(form, text="Worktype:")
lable_Worktype.grid(row=2, column=4, )
textbox_Worktype = Entry(form)
textbox_Worktype.grid(row=2, column=5)

lable_Residencetype = Label(form, text="Residencetype:")
lable_Residencetype.grid(row=3, column=4, pady=10)
textbox_Residencetype = Entry(form)
textbox_Residencetype.grid(row=3, column=5)

lable_Avgglucoselevel = Label(form, text="Avgglucoselevel:")
lable_Avgglucoselevel.grid(row=4, column=4, pady=10)
textbox_Avgglucoselevel = Entry(form)
textbox_Avgglucoselevel.grid(row=4, column=5)

lable_Bmi = Label(form, text="Bmi:")
lable_Bmi.grid(row=5, column=4, pady=10)
textbox_Bmi = Entry(form)
textbox_Bmi.grid(row=5, column=5)

lable_Smokingstatus = Label(form, text="Smokingstatus:")
lable_Smokingstatus.grid(row=6, column=4, pady=10)
textbox_Smokingstatus = Entry(form)
textbox_Smokingstatus.grid(row=6, column=5)


def error(y,y_pred):
    sum=0
    for i in range(0,len(y)):
        sum = sum + abs(y[i] - y_pred[i])
    return sum/len(y)  # tra ve trung binh

def check( X_train, W):
    return np.sign(np.dot(W.T,X_train))

def Stop( W,X_train,y_train,max_count):
    count = 0
    while count < max_count:
        for i in range(len(X_train)):
            y_pre = check(X_train[i].T,W)
            if( y_pre==-1):
                y_pre=0
            if ( y_pre != y_train[i]):
                W = W + check(X_train[i].T,W)*X_train[i]
                count += 1
    return W


k = 5
kf = KFold(n_splits=k)
#Kfold_perceptron code tay
def Kfold_perceptron_tay():
    W = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1])
    min = 9999999
    for train_index, validation_index in kf.split(dt_Train):
        x_train, X_validation = dt_Train.iloc[train_index, :-1].values, dt_Train.iloc[validation_index, :-1].values
        y_train, y_validation = dt_Train.iloc[train_index, -1], dt_Train.iloc[validation_index, -1]
        x_train = np.insert(x_train, 0, 1, axis=1)
        x_validation = np.insert(X_validation, 0, 1, axis=1)
        y_train = np.array(y_train)
        y_validation = np.array(y_validation)

        W = Stop(W,x_train,y_train,2000)
        y_train_pre = check(x_train.T,W)
        y_validation_pre = check(x_validation.T,W)

        sum_error = error(y_train, y_train_pre) + error(y_validation, y_validation_pre)
        if (sum_error < min):
            min = sum_error
            W_find = W
    return W_find

X_test_perceptron_codetay = np.array(dt_Test.iloc[:,:-1])
X_test_perceptron_codetay = np.insert(X_test_perceptron_codetay, 0, 1, axis=1)
y_test_perceptron_codetay = np.array(dt_Test.iloc[:,-1])
W_find = Kfold_perceptron_tay()
y_pre_perceptron_codetay = []
for i in range(0,len(X_test_perceptron_codetay)):
        y_pre_perceptron_codetay.append(check(X_test_perceptron_codetay[i].T,W_find))
        if ( y_pre_perceptron_codetay==-1):
             y_pre_perceptron_codetay=0

print("Độ đo Perceptron code tay: ")
print('Ty le du doan dung Kfold+Perceptron: ', accuracy_score(y_test_perceptron_codetay, y_pre_perceptron_codetay))
print('Precision Kfold+Perceptron: ', precision_score(y_test_perceptron_codetay, y_pre_perceptron_codetay))
print('Recall Kfold+Perceptron: ', recall_score(y_pre_perceptron_codetay, y_test_perceptron_codetay))
print('f1_score Kfold+Perceptron: ', f1_score(y_test_perceptron_codetay, y_pre_perceptron_codetay))



X_test = np.array(dt_Test.iloc[:,:-1])
y_test = np.array(dt_Test.iloc[:,-1])

def Kfold_perceptron(min):
    for train_index, validation_index in kf.split(dt_Train):
        x_train, X_validation = dt_Train.iloc[train_index, :-1].values, dt_Train.iloc[validation_index, :-1].values
        y_train, y_validation = dt_Train.iloc[train_index, -1], dt_Train.iloc[validation_index, -1]
        per = Perceptron(penalty='l2',max_iter=600)
        per.fit(x_train, y_train)
        y_train_pred = per.predict(x_train)
        y_validation_pred = per.predict(X_validation)
        y_train = np.array(y_train)
        y_validation = np.array(y_validation)
        sum_error = error(y_train, y_train_pred) + error(y_validation, y_validation_pred)
        if (sum_error < min):
            min = sum_error
            per_find = per
        return per_find

y_pre_per = Kfold_perceptron(99999999).predict(X_test)
lbl_perceptron = Label(form)
lbl_perceptron.grid(column=2, row=9)
lbl_perceptron.configure(text="Các độ đo của Perceptron: " + '\n'
                              + "Tỷ lệ dự đoán đúng: " + str(accuracy_score(y_test, y_pre_per)) + '\n'
                              + "Tỷ lệ dự đoán sai: " + str(1 - accuracy_score(y_test, y_pre_per)) + '\n'
                              + "Precision: " + str(precision_score(y_test, y_pre_per)) + '\n'
                              + "Recall: " + str(recall_score(y_test, y_pre_per)) + '\n'
                              + "f1_score: " + str(f1_score(y_test, y_pre_per)) + '\n',
                         font=("Arial Bold", 10), fg="red")


def Dudoan_perceptron():
    Gender = textbox_Gender.get()
    Age = textbox_Age.get()
    Hypertension = textbox_Hypertension.get()
    Heartdisease = textbox_Heartdisease.get()
    Evermarried = textbox_Evermarried.get()
    Worktype = textbox_Worktype.get()
    Residencetype = textbox_Residencetype.get()
    Avgglucoselevel = textbox_Avgglucoselevel.get()
    Bmi = textbox_Bmi.get()
    Smokingstatus = textbox_Smokingstatus.get()
    if ((Gender == '') or (Age == '') or (Hypertension == '') or (Heartdisease == '') or (Evermarried == '') or (Worktype == '') or (Residencetype == '')  or (Avgglucoselevel == '') or (Bmi == '') or (Smokingstatus == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([1,float(Gender),float(Age),float(Hypertension),float(Heartdisease),float(Evermarried),float(Worktype),float(Residencetype),float(Avgglucoselevel),float(Bmi),float(Smokingstatus)])
        y_dudoan_perceptron = check(X_dudoan.T,W_find)
        lbl_perceptron_dudoan.configure(text=y_dudoan_perceptron)

button_perceptron = Button(form, text='Kết quả dự đoán Perceptron:', command=Dudoan_perceptron)
button_perceptron.grid(row=11, column=2)
lbl_perceptron_dudoan = Label(form, text="Dữ liệu trên thuộc lớp: ")
lbl_perceptron_dudoan.grid(row=12,column=2)
lbl_perceptron_dudoan = Label(form, text="..." +'\n')
lbl_perceptron_dudoan.grid( row=12,column=3)

#kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’}
#KFold+SVM
def Kfold_SVM(min):
    for train_index, validation_index in kf.split(dt_Train):
        X_train, X_validation = dt_Train.iloc[train_index, :-1].values, dt_Train.iloc[validation_index, :-1].values
        y_train, y_validation = dt_Train.iloc[train_index, -1], dt_Train.iloc[validation_index, -1]
        svm = SVC(kernel='rbf',max_iter=-1)
        svm.fit(X_train, y_train)
        y_train_pred = svm.predict(X_train)
        y_validation_pred = svm.predict(X_validation)
        y_train = np.array(y_train)
        y_validation = np.array(y_validation)
        sum_error = error(y_train, y_train_pred) + error(y_validation, y_validation_pred)
        if (sum_error < min):
            min = sum_error
            svm_find = svm
        return svm_find

y_pre_svm = Kfold_SVM(99999999).predict(X_test)
lbl_svm = Label(form)
lbl_svm.grid(column=5, row=9)
lbl_svm.configure(text="Các độ đo của SVM: " + '\n'
                    + "Tỷ lệ dự đoán đúng: " + str(accuracy_score(y_test,y_pre_svm)) + '\n'
                    + "Tỷ lệ dự đoán sai: " + str(1 - accuracy_score(y_test,y_pre_svm)) + '\n'
                    + "Precision: " + str(precision_score(y_test,y_pre_svm)) + '\n'
                    + "Recall: " + str(recall_score(y_pre_svm,y_test)) + '\n'
                    + "f1_score: " + str(f1_score(y_test,y_pre_svm)) + '\n',
               font=("Arial Bold", 10), fg="red")

def Dudoan_SVM():
    Gender = textbox_Gender.get()
    Age = textbox_Age.get()
    Hypertension = textbox_Hypertension.get()
    Heartdisease = textbox_Heartdisease.get()
    Evermarried = textbox_Evermarried.get()
    Worktype = textbox_Worktype.get()
    Residencetype = textbox_Residencetype.get()
    Avgglucoselevel = textbox_Avgglucoselevel.get()
    Bmi = textbox_Bmi.get()
    Smokingstatus = textbox_Smokingstatus.get()
    if ((Gender == '') or (Age == '') or (Hypertension == '') or (Heartdisease == '') or (Evermarried == '') or (Worktype == '') or (Residencetype == '')  or (Avgglucoselevel == '') or (Bmi == '') or (Smokingstatus == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([Gender,Age,Hypertension,Heartdisease,Evermarried,Worktype,Residencetype,Avgglucoselevel,Bmi,Smokingstatus]).reshape(1, -1)
        y_dudoan_svm = Kfold_SVM(99999999).predict(X_dudoan)
        lbl_svm_dudoan.configure(text=y_dudoan_svm[0])

button_svm = Button(form, text='Kết quả dự đoán SVM:', command=Dudoan_SVM)
button_svm.grid(row=11, column=5)
lbl_svm_dudoan = Label(form, text="Dữ liệu trên thuộc lớp: ")
lbl_svm_dudoan.grid( row=12,column=5)
lbl_svm_dudoan = Label(form, text="...")
lbl_svm_dudoan.grid( row=12,column=6)


form.mainloop()