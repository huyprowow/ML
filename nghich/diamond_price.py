import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split


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
    df = pd.read_csv(r'{0}'.format(os.getcwd()+'\Diamonds Prices2022.csv'))
    df['cut'] = df['cut'].map(
        {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4})
    df['color'] = df['color'].map(
        {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6})
    df['clarity'] = df['clarity'].map(
        {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7})
    X, Y = df.iloc[:, 1:5], df['price']
    XTrain, XTest, YTrain, YTest = train_test_split(
        X, Y, test_size=0.3, random_state=0, shuffle=True)
    best_model = kFoldCrossValidation(XTrain, YTrain)
    YPred = best_model.predict(XTest)
    print("{0} {1} {2}".format("Du doan", "Thuc te", "Chenh lech"))
    for i in range(len(YPred)):
        print("{0} {1} {2}".format(
            YPred[i], YTest.iloc[i], abs(YPred[i]-YTest.iloc[i])))
    print("score: ", best_model.score(XTest, YTest))


if __name__ == '__main__':
    main()
