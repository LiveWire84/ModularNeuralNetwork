import numpy as np


# Sigmoid func, Takes in Array to apply sigmoid func


def g(Z):
    return 1 / (1 + np.e ** (-Z))


def eta(x):
    ETA = 0.0000000001
    return np.maximum(x, ETA)
# Core of the Neural Network


class NeuralNet:
    #  declaration of Neural Network, x takes in
    def __init__(self, x, y, h):
        self.x = x
        self.y = y
        self.dim = np.ones(len(h) + 2)
        self.dim = self.dim.astype(int)
        self.dim[0] = len(x[0])
        self.dim[1:len(h) + 1] = h
        self.dim[-1] = len(y[0])
        self.q = []
        self.d = []
        self.db = []
        self.b = []
        self.beckoning = []
        self.val = 0
        for i in range(len(self.dim) - 1):
            self.q.append(100*np.random.rand(self.dim[i + 1], self.dim[i]))
            self.d.append(np.zeros((self.dim[i + 1], self.dim[i])))
            self.b.append(100*np.random.rand(self.dim[i + 1], 1))
            self.db.append(np.zeros((self.dim[i + 1], 1)))
            self.beckoning.append(False)  #

    def forwardAndBackPropagate(self, l):
        for i in range(len(self.x)):
            a = [np.array(self.x[i], ndmin=2).T]
            for j in range(len(self.q)):
                a.append(g(self.q[j] @ a[j] + self.b[j]))
            self.val = [a[-1], self.y[i]]
            sigma = [(a[-1] - np.array(self.y[i], ndmin=2).T)]
            for j in range(1, len(self.q)):
                sigma.insert(0, (self.q[-j].T @ sigma[0]) * a[-j - 1] * (1 - a[-j - 1]))
            for j in range(len(sigma)):
                self.db[j] = sigma[j]
                self.d[j] = (sigma[j] @ a[j].T)
            self.calcQ(l)


    def calcQ(self, l):
        for i in range(len(self.q)):
            self.q[i] -= l * (self.d[i]) / len(self.x)
        for i in range(len(self.b)):
            self.b[i] -= l * self.db[i] / len(self.x)

    def TestNet(self, x):
        a = [np.array(x, ndmin=2).T]
        for j in range(len(self.q)):
            a.append(g(self.q[j] @ a[j] + self.b[j]))
        return a[-1]

    def gradientCheck(self, e):
        jqe = 0.00
        jqe2 = 0.00
        for i in range(len(self.x)):
            a = [np.array(self.x[i], ndmin=2).T]
            for j in range(len(self.q)):
                a.append(g((self.q[j] + e) @ a[j] + (self.b[j] + e)))
            af = [np.array(self.x[i], ndmin=2).T]
            jqe -= (np.log(a[-1])*self.y[i] + np.log(eta(1 - a[-1]))*(1 - self.y[i]))
            for j in range(len(self.q)):
                af.append(g((self.q[j] - e) @ af[j] + (self.b[j] - e)))
            jqe2 -= (np.log(af[-1])*self.y[i] + np.log(eta(1 - af[-1]))*(1 - self.y[i]))
        return (jqe - jqe2)/(2 * e)

    def TrainNet(self, l, debug, epochs):
        print('initializing.... ')
        for i in range(epochs):
            while True:
                self.forwardAndBackPropagate(l)
                self.calcQ(l)
                if debug:
                    print(self.val)
                    print(self.q)
        return 'Successful', self.q


