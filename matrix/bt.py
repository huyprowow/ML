import numpy as np
import math
import numpy.linalg as la
# b2
A = np.array([[1, 4, -1], [2, 0, 1]])
B = np.array([[-1, 0], [1, 3], [-1, 1]])
# a)
x1 = np.add(A, B.transpose())
x2 = np.subtract(A, B.transpose())
print("x1=", x1)
print("x2=", x2)
# b)
x3 = np.dot(A, 2)
x4 = np.dot(A, B)
print("x3=", x3)
print("x4=", x4)
x5 = np.dot(A, la.pinv(A))
print("x5=", x5)

# b4
# f( x ) = x 2 + 5 sin ( x ) => đạo hàm f′(x) = 2 x + 5 cos(x) éo hiểu cc j :v


def grad(x):
    return 2*x + 5*np.cos(x)


def cost(x):
    return x**2 + 5*np.sin(x)


def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)


(x1, it1) = myGD1(.1, -5)
(x2, it2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations' %
      (x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations' %
      (x2[-1], cost(x2[-1]), it2))


def grad1(x):
    return 2*x + 5*np.cos(x)


def myGD(eta, x0, limit,blabla):
    x = x0
    l = 0
    while abs(grad1(x)) > eta and l < limit:
        print(x,"\n")
        l += 1
        x = x - blabla*grad1(x)
    return x


print(myGD(0.00001, -5, 100,0.01))