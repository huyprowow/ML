# met lone :v slide dang giai thich entrophy thi nham me sang gini xong bung cai anh gini vao cay vc
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics, tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

''' BTVN
# print(os.getcwd()+'\descision_tree_data.csv')
data = pd.read_csv(
    'E:\\Python\\Ml_VSCode_Py\\py1\\classify\\descision_tree_data.csv')
data['age'] = data['age'].map({'<=30': 0, '31…40': 1, '>40': 2})
data['income'] = data['income'].map({'high': 2, 'medium': 1, 'low': 0})
data['student'] = data['student'].map({'yes': 1, 'no': 0})
data['credit_rating'] = data['credit_rating'].map({'excellent': 1, 'fair': 0})
data['buys_computer'] = data['buys_computer'].map({'yes': 1, 'no': 0})
print(data)
dataTrain, dataTest = train_test_split(data, test_size=0.2)
XTrain, Ytrain = dataTrain[['age', 'income', 'student',
                            'credit_rating']], dataTrain['buys_computer']
XTest, Ytest = dataTest[['age', 'income', 'student',
                         'credit_rating']], dataTest['buys_computer']

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(XTrain, Ytrain)
YPredict = clf.predict(XTest)
for i in range(len(YPredict)):
    print('du doan: ', YPredict[i], 'thuc te: ', Ytest.iloc[i])
print('do chinh xac: ', metrics.accuracy_score(Ytest, YPredict))
iris = load_iris()
# vizualize img
# tree.plot_tree(clf)
# import graphviz
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")

# from sklearn.tree import export_graphviz
# from six import StringIO
# from IPython.display import Image
# import pydotplus
# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True,feature_names = ['age', 'income', 'student',
#                          'credit_rating'],class_names=['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('diabetes.png')
# Image(graph.create_png())

# text vizualize
# decision_tree = clf.fit(iris.data, iris.target)
# r = export_text(decision_tree, feature_names=iris['feature_names'])
# print(r)
'''


def ID3(XTrain, Ytrain, XTest, Ytest):
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    clf.fit(XTrain, Ytrain)
    YPredict = clf.predict(XTest)
    for i in range(len(YPredict)):
        print('du doan: ', YPredict[i], 'thuc te: ', Ytest.iloc[i])
    print('do chinh xac accuracy_score: ',
          metrics.accuracy_score(Ytest, YPredict))
    print('do chinh xac precision_score: ',
          metrics.precision_score(Ytest, YPredict))
    print('do chinh xac recall_score: ', metrics.recall_score(Ytest, YPredict))
    tree.plot_tree(clf)
    plt.show()


def Gini(XTrain, Ytrain, XTest, Ytest):
    clf = tree.DecisionTreeClassifier()
    clf.fit(XTrain, Ytrain)
    YPredict = clf.predict(XTest)
    for i in range(len(YPredict)):
        print('du doan: ', YPredict[i], 'thuc te: ', Ytest.iloc[i])
    print('do chinh xac accuracy_score: ',
          metrics.accuracy_score(Ytest, YPredict))
    print('do chinh xac precision_score: ',
          metrics.precision_score(Ytest, YPredict))
    print('do chinh xac recall_score: ', metrics.recall_score(Ytest, YPredict))
    tree.plot_tree(clf)
    plt.show()


def main():
    data = pd.read_csv(
        'E:\\Python\\Ml_VSCode_Py\\py1\\classify\\cars.csv')
    print(data)
    data['buying'] = data['buying'].map({'vhigh': 1, 'high': 0})
    data['maint'] = data['maint'].map(
        {'vhigh': 4, 'high': 2, 'med': 1, 'low': 0})
    data['doors'] = data['doors'].map({'2': 2, '3': 3, '4': 4, '5more': 5})
    data['persons'] = data['persons'].map({'2': 2, '4': 4, 'more': 5})
    data['lug_boot'] = data['lug_boot'].map({'big': 2, 'med': 1, 'small': 0})
    data['safety'] = data['safety'].map({'high': 2, 'med': 1, 'low': 0})
    data['acceptability'] = data['acceptability'].map({'acc': 1, 'unacc': 0})

    print(data)
    dataTrain, dataTest = train_test_split(data, test_size=0.2)
    XTrain, Ytrain = dataTrain[['buying', 'maint', 'doors',
                                'persons', 'lug_boot', 'safety']], dataTrain['acceptability']
    XTest, Ytest = dataTest[['buying', 'maint', 'doors',
                            'persons', 'lug_boot', 'safety']], dataTest['acceptability']

    ID3(XTrain, Ytrain, XTest, Ytest)
    Gini(XTrain, Ytrain, XTest, Ytest)


if __name__ == '__main__':
    main()
