import os

import pandas as pd
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split


def SVM(XTrain, YTrain, XTest, YTest):
    linear = svm.SVC(kernel='linear', C=1.0)
    linear.fit(XTrain, YTrain)
    YPredict = linear.predict(XTest)
    for i in range(len(XTest)):
        print('du doan: ', YPredict[i], 'thuc te: ', YTest.iloc[i])
    print(metrics.classification_report(YTest, YPredict))


def main():
    data = pd.read_csv(r'{0}'.format(os.getcwd()+'\cars.csv'))
    data['buying'] = data['buying'].map({'vhigh': 1, 'high': 0})
    data['maint'] = data['maint'].map(
        {'vhigh': 4, 'high': 2, 'med': 1, 'low': 0})
    data['doors'] = data['doors'].map({'2': 2, '3': 3, '4': 4, '5more': 5})
    data['persons'] = data['persons'].map({'2': 2, '4': 4, 'more': 5})
    data['lug_boot'] = data['lug_boot'].map({'big': 2, 'med': 1, 'small': 0})
    data['safety'] = data['safety'].map({'high': 2, 'med': 1, 'low': 0})
    data['acceptability'] = data['acceptability'].map({'acc': 1, 'unacc': 0})
    # print(data)
    dataTrain, dataTest = train_test_split(data, test_size=0.2)
    XTrain, YTrain = dataTrain.iloc[:, 1:7], dataTrain.iloc[:, 7]
    XTest, YTest = dataTest.iloc[:, 1:7], dataTest.iloc[:, 7]
    SVM(XTrain, YTrain, XTest, YTest)


if __name__ == '__main__':
    main()
