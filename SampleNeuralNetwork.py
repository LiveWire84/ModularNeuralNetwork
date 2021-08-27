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
])                                                                                          #  (Make sure your inputs are an 1-D array even if it has only one value).
# Output Expected
y = np.array([[0], [0], [0], [0], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])    #  (Make sure your output are an 1-D array even if it has only one value).

Net = Mnn.NeuralNet(x, y, [1])
#first parameter takes in input dataset and second parameter takes in expected output.
#The third parameter defines your hidden layer. It takes in a 1-D array which has a length of no. of hidden layers your project require, 
#with inner value being no. of neurons in each hidden layer
#ie [No. of Neurons in layer 1, No. of neurons in layer 2, ......]
print(Net.TrainNet(0.01, 3, True))    #  first parameter is learning rate, second parameter takes in the precision of NN, the final param is debug ie when set true the NN
                                      #  prints the errors in the neurons, returns the final values of all the weights and biases when the NN is trained.
while True:
    print(Net.TestNet([float(input('x1:  ')), float(input('x2:  '))]))  #  To Test the trained network takes in a 1-D array of same len as one of your training sample
