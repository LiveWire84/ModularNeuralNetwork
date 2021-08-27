import numpy as np
import ModularNeuralNetwork as Mnn
#  Training dataset
#  input dataset
x = np.array([
    [0, 0],
    [1, 0],
    [0, 2],
    [2, 0],
    [4, 0],
    [3, 1],
    [2, 2],
    [4, 4],
    [1, 3],
    [0, 4],
    [1.5, 3],
    [0, 4.5],
    [1, 4],
    [1.5, 4],
    [0, 5]
])
# Output Expected
y = np.array([[0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])

Net = Mnn.NeuralNet(x, y, [1])
print(Net.TrainNet(0.01, 3, True))
while True:
    print(Net.TestNet([float(input('x1:\t')), float(input('x2:\t'))]))