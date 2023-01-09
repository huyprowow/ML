#dang sai luoi sua
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics, svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
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
    X = np.array(
        data[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']].values)
    Y = np.array(data['acceptability'])
    max=-float("inf")
    XTrain, XTest, YTrain, YTest=train_test_split(X,Y,test_size=0.3)
    l=svm.SVC(kernel='linear', C=1.0)
    pca = PCA(n_components=3)
    scaler = StandardScaler()
    XTrain = pca.fit_transform(XTrain)
    XTest = pca.transform(XTest)
    XTrain = scaler.fit_transform(XTrain)
    XTest = scaler.transform(XTest)
    

if __name__=="__main__":
    main()