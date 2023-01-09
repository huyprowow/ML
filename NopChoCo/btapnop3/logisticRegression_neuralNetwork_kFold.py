import os
from tkinter import *
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier


def kFoldCrossValidation(XTrain, YTrain, modelType):
    best_model = modelType
    maxAcc = -float('inf')
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    for train_index, val_index in kf.split(XTrain):
        # print("TRAIN:", XTrain.iloc[train_index],
        #       "VAL:", XTrain.iloc[val_index])
        X_kf_train, X_kf_val = XTrain.iloc[train_index], XTrain.iloc[val_index]
        Y_kf_train, Y_kf_val = YTrain.iloc[train_index], YTrain.iloc[val_index]
        model = modelType
        model.fit(X_kf_train, Y_kf_train)
        Y_pred_train = model.predict(X_kf_train)
        Y_pred_val = model.predict(X_kf_val)
        sum_current_accuracy = accuracy_score(
            Y_kf_train, Y_pred_train) + accuracy_score(Y_kf_val, Y_pred_val)
        # print("sum_current_accuracy: ", sum_current_accuracy)
        if (sum_current_accuracy < maxAcc):
            maxAcc = sum_current_accuracy
            best_model = model
        # print("score: ", model.score(X_kf_train, Y_kf_train))
    return best_model


def predictLabel(model, lbl, Temperature, Humidity, Light, CO2, HumidityRatio):
    if ((Temperature == '') or (Humidity == '') or (Light == '') or (CO2 == '') or (HumidityRatio == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        XPred = np.array([Temperature, Humidity, Light,
                         CO2, HumidityRatio], dtype=float).reshape(1, -1)
        YPred = model.predict(XPred)
        lbl.configure(text=((YPred[0] == 1) and "{0} Có khả năng có người sử dụng phòng".format(
            YPred) or "{0} Không có cơ hội cho thuê phòng".format(YPred)))


def scoreModel(model, lblScore, XTest, YTest, ModelName):
    YPred = model.predict(XTest)
    lblScore.configure(text="Tỉ lệ dự đoán đúng của {0} ".format(ModelName)+'\n'
                       + "Precision: " +
                       str(precision_score(YTest, YPred,
                           average='macro')*100)+"%"+'\n'
                       + "Recall: " +
                       str(recall_score(YTest, YPred, average='macro')*100)+"%"+'\n'
                       + "F1-score: " +
                       str(f1_score(YTest, YPred, average='macro')*100)+"%"+'\n'
                       )


def reportModel(model, lblReport, XTest, YTest, ModelName):
    YPred = model.predict(XTest)
    lblReport.configure(text="Báo cáo {0} ".format(ModelName)+'\n'
                        + classification_report(YTest, YPred,
                                                target_names=['Không', 'Có'])
                        )


def correctRatio(model, XTest, YTest):
    YPred = model.predict(XTest)
    correct = 0
    for i in range(len(YPred)):
        if (YPred[i] == YTest.iloc[i]):
            correct = correct+1
    return (correct/len(YPred))*100


def App(LRBest, NNBest, XTest, YTest):
    form = Tk()
    form.title("Phân loại khả năng cho thuê của phòng:")
    form.geometry("1000x500")
    label_Title = Label(form, text="Nhập thông tin phòng:",
                        font=("Arial Bold", 10), fg="red")
    label_Title.grid(row=1, column=1, padx=40, pady=10)

    label_Temperature = Label(form, text="Nhiệt độ:")
    label_Temperature.grid(row=2, column=1, padx=40, pady=10)
    textbox_Temperature = Entry(form)
    textbox_Temperature.grid(row=2, column=2)

    label_Humidity = Label(form, text="Độ ẩm tương đối:")
    label_Humidity.grid(row=3, column=1, pady=10)
    textbox_Humidity = Entry(form)
    textbox_Humidity.grid(row=3, column=2)

    label_Light = Label(form, text="Ánh sáng")
    label_Light.grid(row=4, column=1, pady=10)
    textbox_Light = Entry(form)
    textbox_Light.grid(row=4, column=2)

    label_CO2 = Label(form, text="CO2:")
    label_CO2.grid(row=5, column=1, pady=10)
    textbox_CO2 = Entry(form)
    textbox_CO2.grid(row=5, column=2)

    label_HumidityRatio = Label(form, text="Tỷ lệ độ ẩm:")
    label_HumidityRatio.grid(row=6, column=1, pady=10)
    textbox_HumidityRatio = Entry(form)
    textbox_HumidityRatio.grid(row=6, column=2)

    btnPredLR = Button(
        form,
        text='Kết quả dự đoán theo Linear Regression',
        command=lambda: predictLabel(
            LRBest,
            lblResultLR,
            textbox_Temperature.get(),
            textbox_Humidity.get(),
            textbox_Light.get(),
            textbox_CO2.get(),
            textbox_HumidityRatio.get(),
        )
    )
    btnPredLR.grid(row=2, column=3)
    lblResultLR = Label(form, text="...")
    lblResultLR.grid(row=2, column=4, padx=20)

    btnPredNN = Button(
        form,
        text='Kết quả dự đoán theo Neural Network',
        command=lambda: predictLabel(
            NNBest,
            lblResultNN,
            textbox_Temperature.get(),
            textbox_Humidity.get(),
            textbox_Light.get(),
            textbox_CO2.get(),
            textbox_HumidityRatio.get(),
        )
    )
    btnPredNN.grid(row=4, column=3)
    lblResultNN = Label(form, text="...")
    lblResultNN.grid(row=4, column=4, padx=20)

    lblCorrectRatioLR = Label(form, text="Khả năng dự đoán đúng: {0}%".format(
        correctRatio(LRBest, XTest, YTest)))
    lblCorrectRatioLR.grid(row=3, column=3)
    lblCorrectRatioNN = Label(form, text="Khả năng dự đoán đúng: {0}%".format(
        correctRatio(NNBest, XTest, YTest)))
    lblCorrectRatioNN.grid(row=5, column=3)

    labelTitleScore = Label(form, text="Tổng quan thông tin Model:",
                            font=("Arial Bold", 10), fg="red")
    labelTitleScore.grid(row=7, column=1, padx=10, pady=10)
    lblScoreLR = Label(form)
    lblScoreLR.grid(row=8, column=1)
    scoreModel(LRBest, lblScoreLR, XTest, YTest, "Linear Regression")
    lblScoreNN = Label(form)
    lblScoreNN.grid(row=8, column=3)
    scoreModel(NNBest, lblScoreNN, XTest, YTest, "Neural Network")

    lblReportLR = Label(form)
    lblReportLR.grid(row=8, column=2, pady=20)
    reportModel(LRBest, lblReportLR, XTest, YTest, "Linear Regression")
    lblReportNN = Label(form)
    lblReportNN.grid(row=8, column=4, pady=20)
    reportModel(NNBest, lblReportNN, XTest, YTest, "Neural Network")
    form.mainloop()


def main():
    df = pd.read_csv(r'{0}'.format(os.getcwd()+'\\roomOccupancy.csv'))
    X, Y = pd.DataFrame(df.iloc[:, :5].values), df['Occupancy']
    # print(X, Y)
    XTrain, XTest, YTrain, YTest = train_test_split(
        X, Y, test_size=0.3, random_state=0, shuffle=True)
    LRBest = kFoldCrossValidation(XTrain, YTrain, LogisticRegression())
    NNBest = kFoldCrossValidation(XTrain, YTrain, MLPClassifier(
        hidden_layer_sizes=(100, ), solver='lbfgs', alpha=1e-5, random_state=1))
    # YPred_LR = LRBest.predict(XTest)
    # print("LR: ", LRBest.score(XTest, YTest))
    # YPred_NN = NNBest.predict(XTest)
    # print("NN:", NNBest.score(XTest, YTest))
    App(LRBest, NNBest, XTest, YTest)


if __name__ == '__main__':
    main()
