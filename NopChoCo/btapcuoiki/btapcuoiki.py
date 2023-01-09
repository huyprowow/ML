import os
from tkinter import *
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
from LRCustom import LogisticRegressionCustom
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


def predictLabel(model, lbl,
                 LongHair,
                 ForeWidth,
                 ForeHeight,
                 NoseWide,
                 NoseLong,
                 lipThin,
                 DistanceFromNoseToLipOption):
    if ((ForeWidth == '') or (ForeHeight == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        XPred = np.array([
            (LongHair == "Đúng") and 1 or 0,
            ForeWidth,
            ForeHeight,
            (NoseWide == "Đúng") and 1 or 0,
            (NoseLong == "Đúng") and 1 or 0,
            (lipThin == "Đúng") and 1 or 0,
            (DistanceFromNoseToLipOption == "Đúng") and 1 or 0], dtype=float).reshape(1, -1)
        YPred = model.predict(XPred)
        lbl.configure(text=((YPred[0] == 1) and "{0} Nam".format(
            YPred) or "{0} Nữ".format(YPred)))


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
                                                target_names=['Nữ', 'Nam'])
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
    form.title("Phân loại giới tính:")
    form.geometry("1100x600")
    label_Title = Label(form, text="Nhập đặc điểm người cần dự đoán:",
                        font=("Arial Bold", 10), fg="red")
    label_Title.grid(row=1, column=1, padx=40, pady=10)

    longHairOption = ["Đúng", "Sai"]
    value_LongHair = StringVar(form)
    value_LongHair.set(longHairOption[0])
    label_LongHair = Label(form, text="Tóc dài ?")
    label_LongHair.grid(row=2, column=1, padx=40, pady=10)
    option_LongHair = OptionMenu(form, value_LongHair, *longHairOption)
    option_LongHair.grid(row=2, column=2)

    label_ForeWidth = Label(
        form, text="Chiều rộng trán (cm):")
    label_ForeWidth.grid(row=3, column=1, pady=10)
    textbox_ForeWidth = Entry(form)
    textbox_ForeWidth.grid(row=3, column=2)

    label_ForeHeight = Label(
        form, text="Chiều cao trán (cm):")
    label_ForeHeight.grid(row=4, column=1, pady=10)
    textbox_ForeHeight = Entry(form)
    textbox_ForeHeight.grid(row=4, column=2)

    noseWideOption = ["Đúng", "Sai"]
    value_NoseWide = StringVar(form)
    value_NoseWide.set(noseWideOption[0])
    label_NoseWide = Label(form, text="Mũi rộng ?")
    label_NoseWide.grid(row=5, column=1, padx=40, pady=10)
    option_NoseWide = OptionMenu(form, value_NoseWide, *noseWideOption)
    option_NoseWide.grid(row=5, column=2)

    noseLongOption = ["Đúng", "Sai"]
    value_NoseLong = StringVar(form)
    value_NoseLong.set(noseLongOption[0])
    label_NoseLong = Label(form, text="Mũi dài ?")
    label_NoseLong.grid(row=6, column=1, padx=40, pady=10)
    option_NoseLong = OptionMenu(form, value_NoseLong, *noseLongOption)
    option_NoseLong.grid(row=6, column=2)

    lipThinOption = ["Đúng", "Sai"]
    value_lipThin = StringVar(form)
    value_lipThin.set(lipThinOption[0])
    label_lipThin = Label(form, text="Môi mỏng ?")
    label_lipThin.grid(row=7, column=1, padx=40, pady=10)
    option_lipThin = OptionMenu(form, value_lipThin, *lipThinOption)
    option_lipThin.grid(row=7, column=2)

    distanceFromNoseToLipOption = ["Đúng", "Sai"]
    value_DistanceFromNoseToLipOption = StringVar(form)
    value_DistanceFromNoseToLipOption.set(distanceFromNoseToLipOption[0])
    label_DistanceFromNoseToLipOption = Label(
        form, text="Khoảng cách từ mũi đến môi xa ?")
    label_DistanceFromNoseToLipOption.grid(row=8, column=1, padx=40, pady=10)
    option_DistanceFromNoseToLipOption = OptionMenu(
        form, value_DistanceFromNoseToLipOption, *distanceFromNoseToLipOption)
    option_DistanceFromNoseToLipOption.grid(row=8, column=2)

    btnPredLR = Button(
        form,
        text='Kết quả dự đoán theo Linear Regression',
        command=lambda: predictLabel(
            LRBest,
            lblResultLR,
            value_LongHair.get(),
            textbox_ForeWidth.get(),
            textbox_ForeHeight.get(),
            value_NoseWide.get(),
            value_NoseLong.get(),
            value_lipThin.get(),
            value_DistanceFromNoseToLipOption.get()
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
            value_LongHair.get(),
            textbox_ForeWidth.get(),
            textbox_ForeHeight.get(),
            value_NoseWide.get(),
            value_NoseLong.get(),
            value_lipThin.get(),
            value_DistanceFromNoseToLipOption.get()
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
    labelTitleScore.grid(row=9, column=1, padx=10, pady=10)
    lblScoreLR = Label(form)
    lblScoreLR.grid(row=10, column=1)
    scoreModel(LRBest, lblScoreLR, XTest, YTest, "Linear Regression")
    lblScoreNN = Label(form)
    lblScoreNN.grid(row=10, column=3)
    scoreModel(NNBest, lblScoreNN, XTest, YTest, "Neural Network")

    lblReportLR = Label(form)
    lblReportLR.grid(row=10, column=2, pady=20)
    reportModel(LRBest, lblReportLR, XTest, YTest, "Linear Regression")
    lblReportNN = Label(form)
    lblReportNN.grid(row=10, column=4, pady=20)
    reportModel(NNBest, lblReportNN, XTest, YTest, "Neural Network")
    form.mainloop()


def main():
    df = pd.read_csv(r'{0}'.format(os.getcwd()+'\\gender_classification.csv'))
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    X, Y = pd.DataFrame(df.iloc[:, :7].values), df['gender']
    # print(X, Y)
    XTrain, XTest, YTrain, YTest = train_test_split(
        X, Y, test_size=0.3, random_state=0, shuffle=True)
    LRBest = kFoldCrossValidation(XTrain, YTrain, LogisticRegressionCustom()) #learning_rate=0.001, n_iters=1000
    NNBest = kFoldCrossValidation(XTrain, YTrain, MLPClassifier(
        hidden_layer_sizes=(50, 30), solver='lbfgs', alpha=1e-5, random_state=1))
    # YPred_LR = LRBest.predict(XTest)
    # print("LR: ", LRBest.score(XTest, YTest))
    # YPred_NN = NNBest.predict(XTest)
    # print("NN:", NNBest.score(XTest, YTest))
    App(LRBest, NNBest, XTest, YTest)


if __name__ == '__main__':
    main()
