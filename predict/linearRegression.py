# # f'x=0 (f'x X*X^T*w=X*y)
import sklearn.linear_model as lm
import numpy as np
import numpy.linalg as la

X = np.array([[60, 2, 10],
             [40, 2, 5],
              [100, 3, 7]
              ])
Y = np.array([[10], [12], [20]])


def w():
    # XtX=np.dot(X.transpose(),X) k co w0
    # XtY=np.dot(X.transpose(),Y)
    # return la.solve(XtX,XtY)

    # bai giang sai công thuc la w=(XtX)^-1*XtY

    XT_X = np.dot(X.transpose(), X)
    XT_Y = np.dot(X.transpose(), Y)
    return np.dot(la.pinv(XT_X), XT_Y)


def wsk():
    lg = lm.LinearRegression()
    reg = lg.fit(X, Y)
    print("w0=", reg.intercept_)
    XTest = np.array([[60, 2, 10]])
    yPredict = reg.predict(XTest)  # tu x test=> du doan y
    print("yPredict=", yPredict)
    return reg.coef_  # w


print(w())
print(wsk())
# v1 = np.array([[1, 2], [3, 4]])
# v2 = np.array([[1, 2], [3, 4]])
# print(v1.dot(v2))
# print(v1 *v2)