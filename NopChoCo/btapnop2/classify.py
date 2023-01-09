
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics, tree
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split


def perceptron(XTrain, Ytrain, XTest, Ytest):
    p = Perceptron()
    p.fit(XTrain, Ytrain)
    YPredict = p.predict(XTest)
    # for i in range(len(YPredict)):
    #     print('du doan: ', YPredict[i], 'thuc te: ', Ytest.iloc[i])
    print(metrics.classification_report(
        Ytest, YPredict, target_names=['Không mua', 'Mua']))
    input("Nhấn Enter để tiếp tục...")


def ID3(XTrain, Ytrain, XTest, Ytest):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(XTrain, Ytrain)
    YPredict = clf.predict(XTest)
    # for i in range(len(YPredict)):
    #     print('du doan: ', YPredict[i], 'thuc te: ', Ytest.iloc[i])
    print(metrics.classification_report(
        Ytest, YPredict, target_names=['Không mua', 'Mua']))
    # show cay
    fig, axes = plt.subplots()
    plt.title('ID3')
    tree.plot_tree(clf, filled=True)
    # plt.show()
    # xuat anh
    fig, axes = plt.subplots(figsize=(120, 30), dpi=200)
    tree.plot_tree(clf, filled=True, fontsize=3.5)
    print("Đang xuất ảnh... chờ 1 chút...")
    fig.savefig('ID3.png')
    plt.close('all')


def CART(XTrain, Ytrain, XTest, Ytest):
    clf = tree.DecisionTreeClassifier()
    clf.fit(XTrain, Ytrain)
    YPredict = clf.predict(XTest)
    # for i in range(len(YPredict)):
    #     print('du doan: ', YPredict[i], 'thuc te: ', Ytest.iloc[i])
    print(metrics.classification_report(
        Ytest, YPredict, target_names=['Không mua', 'Mua']))
    # show cay
    fig, axes = plt.subplots()
    plt.title('CART')
    tree.plot_tree(clf, filled=True)
    # plt.show()
    # xuat anh
    print("Đang xuât ảnh... chờ 1 chút...")
    fig, axes = plt.subplots(figsize=(120, 30), dpi=200)
    tree.plot_tree(clf, filled=True, fontsize=3.5)
    fig.savefig('CART.png')
    plt.close('all')


def main():
    data = pd.read_csv(r'{0}'.format(
        os.getcwd()+'\TravelInsurancePrediction.csv'))
    # print(data)
    data['Employment Type'] = data['Employment Type'].map(
        {'Private Sector/Self Employed': 1, 'Government Sector': 0})
    data['GraduateOrNot'] = data['GraduateOrNot'].map({'Yes': 1, 'No': 0})
    data['FrequentFlyer'] = data['FrequentFlyer'].map({'Yes': 1, 'No': 0})
    data['EverTravelledAbroad'] = data['EverTravelledAbroad'].map(
        {'Yes': 1, 'No': 0})
    # print(data)
    XTrain, XTest, Ytrain, Ytest = train_test_split(
        data.iloc[:, 1:9], data.iloc[:, 9], test_size=0.3)
    perceptron(XTrain, Ytrain, XTest, Ytest)
    ID3(XTrain, Ytrain, XTest, Ytest)
    CART(XTrain, Ytrain, XTest, Ytest)


if __name__ == '__main__':
    main()
