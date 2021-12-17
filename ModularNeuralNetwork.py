import numpy as np
import pandas as pd


def tanh(Z):
    return (np.e**(Z) - np.e**(-Z))/(np.e**Z + np.e**(-Z))


def sig(Z):
    return 1/(1 + np.e**(-Z))


def LeakyReLu(Z):
    n = []
    for i in range(len(Z)):
        n.append(np.maximum(0.01*Z[i], Z[i]))
    return n


def ReLu(Z):
    return np.log(1 + np.e**Z)


class MNN:
    def __init__(self, X, Y, h):
        self.X = X
        self.Y = Y
        self.dim = np.ones(len(h)+2).astype(int)
        self.dim[0] = len(self.X[0])
        self.dim[1:len(h)+1] = h
        self.dim[-1] = len(Y[0])
        self.w = []
        self.dw = []
        self.db = []
        self.dz = []
        self.b = []
        self.a = []
        self.acc = 0
        for i in range(len(self.dim) - 1):
            self.w.append(np.random.rand(self.dim[i + 1], self.dim[i])*0.01)
            self.dw.append(np.zeros((self.dim[i + 1], self.dim[i])))
            self.b.append(np.random.rand(self.dim[i + 1], 1)*0.0001)
            self.db.append(np.zeros((self.dim[i + 1], 1)))
            self.dz.append(np.zeros((self.dim[i + 1], self.dim[i])))

    def ForwardPropagate(self, ):
        self.a = []
        self.a.append(self.X.T)
        for i in range(len(self.w) - 1):
            self.a.append(sig(self.w[i]@self.a[-1] + self.b[i]))
        self.a.append(sig(self.w[-1]@self.a[-1] + self.b[-1]))

    def BackPropagate(self):
        self.dz[-1] = (self.a[-1] - self.Y.T)
        self.dw[-1] = self.dz[-1]@self.a[-2].T/len(self.X)
        self.db[-1] = np.sum(self.dz[-1], axis=1, keepdims=True)/len(self.X)
        for i in range(-2, -(len(self.dw)+1), -1):
            self.dz[i] = (self.dz[i+1].T@self.w[i+1]).T*self.a[i]*(1 - self.a[i])
            self.dw[i] = self.dz[i]@self.a[i-1].T
            self.db[i] = np.sum(self.dz[i], axis=1, keepdims=True)/len(self.X)
        print(self.db)

    def assignDelta(self, l):
        for i in range(len(self.w)):
            self.w[i] -= l*self.dw[i]
            self.b[i] -= l*self.db[i]

    def trainNet(self, l):
        print('initializing.... ')
        self.acc = 0
        while self.acc <= 0.9:
            self.ForwardPropagate()
            self.BackPropagate()
            self.assignDelta(l)
            self.acc = 0
            predans = np.argmax(self.a[-1].T, axis=1)
            actans = np.argmax(self.Y, axis=1)
            print(predans, actans)
            for i in range(len(predans)):
                if predans[i] == actans[i]:
                    self.acc += 1
            self.acc /= len(self.X)
            print(self.acc)
        return 'Successful', self.w

    def TestNet(self, x):
        na = [np.array(x, ndmin=2).T]
        for j in range(len(self.w) - 1):
            na.append(tanh(self.w[j] @ na[-1] + self.b[j]))
        na.append(sig(self.w[-1] @ na[-1] + self.b[-1]))
        return na[-1]
