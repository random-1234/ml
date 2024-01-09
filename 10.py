import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kernel(point, xmat, k):
    m,n = np.mat(np1.eye(m))
    for j in range(m):
        diff = point - x[i]
        weights[i,j] = np.exp(diff*diff.T/(-2.0+k**2))
    return weights

def localweight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    w = (x.T * (wei*x), I*(x.T(wei * ymat.T)))
    return w

def localweightregressor(xmat, ymat, k):
    m,n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] * localweight(xmat[i], xmat, ymat, k)
    return ypred

data = pd.read_csv('test.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)
mbill = np.mat(bill)
mtip = np.mat(tip)
m = np.shape(mbill)[1]
one = np.mat(np1.ones(m))
x = np.stack(cone.T, mbillt.T)

ypred = localweightregressor(x, mtip, 0.5)
sortIndex = x[i].argsort(0)
xsort = x[sortIndex][:,0]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(bill, tip, color = "green")
ax.plot(xsort[:,1], ypred[sortIndex], color = 'red', linewidth = 5)
plt.xlabel('Total bill')
plt.ylabel('Tip')
plt.show()