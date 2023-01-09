"""
 unsuitable data set compressed data is treated by clustering or classification rather than linearity 
 because the data has only 2 labels while regression predicts on 1 value domain (continuous values) :v
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


def NSE(y_test, y_pred):
    return (1 - (np.sum((y_pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)))


def kFoldCrossValidation(XTrain, YTrain):
    best_model = LinearRegression()
    min_error = float('inf')
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    for train_index, val_index in kf.split(XTrain):
        print("TRAIN:", XTrain.iloc[train_index],
              "VAL:", XTrain.iloc[val_index])
        X_kf_train, X_kf_val = XTrain.iloc[train_index], XTrain.iloc[val_index]
        Y_kf_train, Y_kf_val = YTrain.iloc[train_index], YTrain.iloc[val_index]
        model = LinearRegression()
        model.fit(X_kf_train, Y_kf_train)
        Y_pred_train = model.predict(X_kf_train)
        Y_pred_val = model.predict(X_kf_val)
        sum_error = mean_squared_error(
            Y_kf_train, Y_pred_train) + mean_squared_error(Y_kf_val, Y_pred_val)
        # print("sum_error: ", sum_error)
        if (sum_error < min_error):
            min_error = sum_error
            best_model = model
        # print("score: ", model.score(X_kf_train, Y_kf_train))
    return best_model


def main():
    df = pd.read_csv(r'{0}'.format(os.getcwd()+'\MatchTimelinesFirst15.csv'))
    X, Y = df.iloc[:, 3:], df['blue_win']
    XTrain, XTest, YTrain, YTest = train_test_split(
        X, Y, test_size=0.3, random_state=0, shuffle=True)
    best_model = kFoldCrossValidation(XTrain, YTrain)
    YPred = best_model.predict(XTest)
    print("{0} {1} {2}".format("Du doan", "Thuc te", "Chenh lech"))
    for i in range(len(YPred)):
        print("{0} {1} {2}".format(
            YPred[i], YTest.iloc[i], abs(YPred[i]-YTest.iloc[i])))
    print("R^2: ", best_model.score(XTest, YTest))
    print("NSE: ", NSE(YTest, YPred))
    print("RMSE: ", mean_squared_error(YTest, YPred, squared=False))
    print("MSE: ", mean_squared_error(YTest, YPred, squared=True))


if __name__ == '__main__':
    main()
