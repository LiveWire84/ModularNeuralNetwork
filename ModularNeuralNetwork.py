import numpy as np
# Sigmoid func, Takes in Array to apply sigmoid func


def g(Z):
    return 1 / (1 + np.e ** (-Z))


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
        for i in range(len(self.dim) - 1):
            self.q.append(np.random.rand(self.dim[i + 1], self.dim[i]))
            self.d.append(np.zeros((self.dim[i + 1], self.dim[i])))
            self.b.append(np.random.rand(self.dim[i + 1], 1))
            self.db.append(np.zeros((self.dim[i + 1], 1)))
            self.beckoning.append(False)                #

    def forwardAndBackPropagate(self):
        for i in range(len(self.x)):
            a = [np.array(self.x[i], ndmin=2).T]
            for j in range(len(self.q)):
                a.append(g(self.q[j] @ a[j] + self.b[j]))
            sigma = [(a[-1] - np.array(self.y[i], ndmin=2).T) * a[-1] * (1 - a[-1])]
            for j in range(1, len(self.q)):
                sigma.insert(0, (self.q[-j].T @ sigma[0]) * a[-j - 1] * (1 - a[-j - 1]))
            for j in range(len(sigma)):
                self.db[j] += sigma[j] / len(self.x)
                self.d[j] += (sigma[j] @ a[j].T) / len(self.x)

    def calcQ(self, l):
        for i in range(len(self.q)):
            self.q[i] -= l * (self.d[i] + self.q[i])
        for i in range(len(self.b)):
            self.b[i] -= l * self.db[i]

    def TestNet(self, x):
        a = [np.array(x, ndmin=2).T]
        for j in range(len(self.q)):
            a.append(g(self.q[j] @ a[j] + self.b[j]))
        return a[-1]

    def TrainNet(self, l, precision, debug):
        while True:
            self.forwardAndBackPropagate()
            self.calcQ(l)
            if debug:
                print(self.d)
                print(self.db)
            for i in range(len(self.db)):
                if (np.around(self.db[i], decimals=precision) == np.zeros(self.db[i].shape)).all():
                    self.beckoning[i] = True
                else:
                    self.beckoning[i] = False
            if all(self.beckoning):
                break
        return 'Successful', self.q

