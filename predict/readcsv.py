import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import os
print(os.getcwd())
data = pd.read_csv("py1/USA_Housing.csv")
dt_Train, dt_Test = train_test_split(
    data, test_size=0.3, shuffle=False)  # cat 70 train 30 test tu tren xuong k tron lan

X_train = dt_Train.iloc[:, :5]  # lay cot 0->4
Y_train = dt_Train.iloc[:, 5]  # lay cot 5
X_test = dt_Test.iloc[:, :5]
Y_test = dt_Test.iloc[:, 5]

# goi linear regression
reg = LinearRegression().fit(X_train, Y_train)
y_pred = reg.predict(X_test)  # du doan y tu x test
y = np.array(Y_test)  # y thuc te
# in diem cang gan 1.0 cang chuan
print("diem: ", r2_score(y, y_pred))
# chech lech
for i in range(len(y_pred)):
    print("thuc te  ", "du doan  ", "chenh lech")
    print("{0:.2f} {1} => {2}".format(y_pred[i], y[i], abs(y_pred[i]-y[i])))
#4 ham do do
# chia k folder cross validation du doan


# import pandas as pd
# import math
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_absolute_error

# ham danh gia
# def NSE(y_true, y_pred):
#     tu = 0
#     mau = 0
#     for i in range(len(y_true)):
#         tu += (y_true[i]-y_pred[i])**2
#         mau += (y_true[i]-np.mean(y_true))**2
#     return 1-tu/mau


# def R2(y_true, y_pred):
#     tu = 0
#     mau1 = 0
#     mau2 = 0
#     for i in range(len(y_true)):
#         tu += (y_true[i]-np.mean(y_true))*(y_pred[i]-np.mean(y_pred))
#         mau1 += (y_true[i]-np.mean(y_true))**2
#         mau2 += (y_pred[i]-np.mean(y_pred))**2
#     return (tu/math.sqrt(mau1*mau2))**2


# def MAE(y_true, y_pred):
#     tu = 0
#     for i in range(len(y_true)):
#         tu += abs(y_true[i]-y_pred[i])
#     return tu/len(y_true)


# def RMSE(y_true, y_pred):
#     tu = 0
#     for i in range(len(y_true)):
#         tu += (y_true[i]-y_pred[i])**2
#     return math.sqrt(tu/len(y_true))

# doc du lieu
# data = pd.read_csv("py1/USA_Housing.csv")
# dataTrain, dataTest = train_test_split(data, test_size=0.2) #t??ch dl 80 train 20 test
# # X,Y=data.iloc[:,:-1],data.iloc[:,-1]
# # 4 cot trc cho x, cot cuoi cho y
# XTest, YTest = dataTest.iloc[:, :5], dataTest.iloc[:, 5] #t??ch dl test

# n = 3
# kf = KFold(n_splits=n)# k folder voi tach n phan

# min_err = 999999999999 # khoi t???o loi nho nhat
# # current=1# t???p train hien t???i ( kfold t??ch n t???p )( c?? hay k t??y b??i) trong for ++ n?? l??n
# for train_index, val_index in kf.split(dataTrain): #t??ch dl train th??nh n t???p, ch??? s??? t???p train v?? gi?? tr??? ????? validaion ( 1 c??i ????? train 1 c??i x??c th???c l???i)
#     # t??ch dl train th??nh ma tr???n x ????? train x validation d??? ??o??n ra 2 gi?? tr??? y predict, c???a Xtrain, xval
#     # sau ???? so l???i voi ytrain y val ????? t??nh t??? l??? l???i 
#     # 1 ??px train ch??? l???y 1 th???ng ????? validation c??n l???i ????? train
#     XTrain, XVal = dataTrain.iloc[train_index,
#                                   :5], dataTrain.iloc[val_index, :5]
#     YTrain, YVal = dataTrain.iloc[train_index, 5], dataTrain.iloc[val_index, 5]
#     # print("TRAIN:", train_index, "\nVAL:", val_index)
#     # print("TRAIN:", XTrain, "\nVAL:", XVal)
#     # print("TRAIN:", YTrain, "\nVAL:", YVal)
#     currentReg = LinearRegression().fit(XTrain, YTrain) # fit t???o h???i quy
#     y_predTrain = currentReg.predict(XTrain) # d??? ??o??n y t??? x train
#     y_predVal = currentReg.predict(XVal) # d??? ??o??n y t??? x val
#     # t??nh t???ng l???i c???a 2 c??i train v???i val b???ng d??ng y th???c t??? v???i d??? ??o??n
#     sum_err_score = mean_absolute_error(
#         YTrain, y_predTrain)+mean_absolute_error(YVal, y_predVal) # mean_absolute_error() l?? h??m mse s???n trong th?? vi???n sklean
#     print("\n sum_err_scrore= ", sum_err_score)
#     if (sum_err_score < min_err): # n???u t???ng l???i nh??? h??n min_err => t???t h??n => l??u l???i ???????ng h???i quy v???i g??n l???i min_err
#         print("current min err= ", sum_err_score)
#         min_err = sum_err_score
#         reg = currentReg
# print("\n", reg, "\n min_err= ", min_err) # k???t th??c th?? ??c l???i nh??? nh???t v?? ???????ng h???i quy t???t nh???t
# # ????a ra d??? ??o??n v???i dl test b???ng ???????ng t???t nh???t t??m ??c
# yPredict = reg.predict(XTest)
# y = np.array(YTest)
# print("du doan: ", "thuc te: ", "=> lech: ")
# for i in range(len(y)):
#     print("{0:.2f} {1:.2f} {2:.2f}".format(yPredict[i], y[i], abs(yPredict[i]-y[i]))) #.2 ch??? ????? lm tr??n

