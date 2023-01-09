import numpy as np
def check(x,y,w):
    classify_err=True
    if -y*sign(w.transpose() @ x)>0:
        classify_err = False
    return classify_err, xi, yi

def perceptron(w0, x, y):
    t = 0
    w = w0
    classify_err, xi, yi = check(x, y, w)
    while classify_err == False:
        w = w+yi*xi
        t += 1
        classify_err, xi, yi = check(x, y, w)

