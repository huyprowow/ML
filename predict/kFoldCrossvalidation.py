import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# ham danh gia


def NSE(y_true, y_pred):
    tu = 0
    mau = 0
    for i in range(len(y_true)):
        tu += (y_true[i]-y_pred[i])**2
        mau += (y_true[i]-np.mean(y_true))**2
    return 1-tu/mau


def R2(y_true, y_pred):
    tu = 0
    mau1 = 0
    mau2 = 0
    for i in range(len(y_true)):
        tu += (y_true[i]-np.mean(y_true))*(y_pred[i]-np.mean(y_pred))
        mau1 += (y_true[i]-np.mean(y_true))**2
        mau2 += (y_pred[i]-np.mean(y_pred))**2
    return (tu/math.sqrt(mau1*mau2))**2


def MAE(y_true, y_pred):
    tu = 0
    for i in range(len(y_true)):
        tu += abs(y_true[i]-y_pred[i])
    return tu/len(y_true)


def RMSE(y_true, y_pred):
    tu = 0
    for i in range(len(y_true)):
        tu += (y_true[i]-y_pred[i])**2
    return math.sqrt(tu/len(y_true))


# # doc du lieu
# data = pd.read_csv("py1/USA_Housing.csv")
# # X,Y=data.iloc[:,:-1],data.iloc[:,-1]
# X, Y = data.iloc[:, :5], data.iloc[:, 5]  # 4 cot trc cho x, cot cuoi cho y

# n = 3
# kf = KFold(n_splits=n)

# min_err = 999999999999
# best_train_index = 1
# current = 1
# for train_index, test_index in kf.split(X, Y):
#     print("TRAIN:", train_index, "\nTEST:", test_index)
#     print("TRAIN:", X.iloc[train_index], "\nTEST:", X.iloc[test_index])
#     print("TRAIN:", Y.iloc[train_index], "\nTEST:", Y.iloc[test_index])
#     currentReg = LinearRegression().fit(
#         X.iloc[train_index], Y.iloc[train_index])  # fit tung phan train
#     y_pred = currentReg.predict(X.iloc[test_index])  # predict tung phan test
#     err_score = mean_absolute_error(Y.iloc[test_index], y_pred)
#     print("\n err_scrore= ", err_score)
#     if (err_score < min_err):
#         print("current min err= ", err_score)
#         min_err = err_score
#         reg = currentReg
#         best_train_index = current
#     current += 1
# print("\nbest_train_index= ", best_train_index)
# print("\n", reg, "\n min_err= ", min_err)

# doc du lieu
data = pd.read_csv("py1/USA_Housing.csv")
dataTrain, dataTest = train_test_split(data, test_size=0.2)
# X,Y=data.iloc[:,:-1],data.iloc[:,-1]
# 4 cot trc cho x, cot cuoi cho y
XTest, YTest = dataTest.iloc[:, :5], dataTest.iloc[:, 5]

n = 3
kf = KFold(n_splits=n)

min_err = 999999999999
current = 1
for train_index, val_index in kf.split(dataTrain):
    XTrain, XVal = dataTrain.iloc[train_index,
                                  :5], dataTrain.iloc[val_index, :5]
    YTrain, YVal = dataTrain.iloc[train_index, 5], dataTrain.iloc[val_index, 5]
    print("TRAIN:", train_index, "\nVAL:", val_index)
    print("TRAIN:", XTrain, "\nVAL:", XVal)
    print("TRAIN:", YTrain, "\nVAL:", YVal)
    currentReg = LinearRegression().fit(XTrain, YTrain)
    y_predTrain = currentReg.predict(XTrain)
    y_predVal = currentReg.predict(XVal)

    sum_err_score = mean_absolute_error(
        YTrain, y_predTrain)+mean_absolute_error(YVal, y_predVal)
    print("\n sum_err_scrore= ", sum_err_score)
    if (sum_err_score < min_err):
        print("current min err= ", sum_err_score)
        min_err = sum_err_score
        reg = currentReg
print("\n", reg, "\n min_err= ", min_err)
yPredict = reg.predict(XTest)
y = np.array(YTest)
print("du doan: ", "thuc te: ", "=> lech: ")
for i in range(len(y)):
    print("{0:.2f} {1:.2f} {2:.2f}".format(
        yPredict[i], y[i], abs(yPredict[i]-y[i])))
