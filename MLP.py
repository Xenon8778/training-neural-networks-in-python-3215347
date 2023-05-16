import numpy as np


class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By defaul it's 1.0."""

    def __init__(self, inputs, bias=1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias)."""
        self.weights = (np.random.rand(inputs+1) * 2) - 1
        self.bias = bias

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        sum = np.dot(np.append(x, self.bias), self.weights)
        return self.sigmoid(sum)

    def set_weights(self, w_init):
        """Set the weights. w_init is a python list with the weights."""
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        """Evaluate the sigmoid function for the floating point input x."""
        return 1/(1+np.exp(-x))


# test code
neuron = Perceptron(inputs=2)
neuron.set_weights([15, 15, -10])  # OR

# print("Gate:")
# print("0 0 = {0:.10f}".format(neuron.run([0, 0])))
# print("0 1 = {0:.10f}".format(neuron.run([0, 1])))
# print("1 0 = {0:.10f}".format(neuron.run([1, 0])))
# print("1 1 = {0:.10f}".format(neuron.run([1, 1])))

neuron1 = Perceptron(inputs=2)
neuron1.set_weights([-10,-10,15])

neuron2 = Perceptron(inputs=2)
neuron2.set_weights([15,15,-10])

neuron3 = Perceptron(inputs=2)
neuron3.set_weights([10,10,-15])

def XORGate(A):
    X1 = neuron1.run(A)
    X2 = neuron2.run(A)
    return neuron3.run([X1,X2])

# layers = number of neurons per layers
# BIAS = bias rate
# eta = learning rate

class MultilayerPerceptron:
    def __init__(self, layers, bias=1):
        self.layers = np.array(layers,dtype=object)
        self.bias = bias
        self.network = []
        self.values = []

        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            if i>0:
                for j in range(self.layers[i]):
                    self.network[i].append(Perceptron(inputs=self.layers[i-1],bias=self.bias))

        self.network = np.array([np.array(x) for x in self.network],dtype=object)
        self.values = np.array([np.array(x) for x in self.values],dtype=object)

    def set_weights(self,w_init):
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i+1][j].set_weights(w_init[i][j])

    def printWeights(self):
        print()
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                print("Layer",i,"Neuron",j,self.network[i][j].weights)
        print()

    def run(self,x):
        x = np.array(x, dtype=object)
        self.values[0] = x
        for i in range(1,len(self.layers)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1]    


MLP = MultilayerPerceptron([2,2,1])
MLP.set_weights([[[-10, -10, 15], [15, 15, -10]], [[10, 10, -15]]])
MLP.printWeights()

print("XOR Gate:")
print("0 0 = {0:.10f}".format(MLP.run([0, 0])[0]))
print("0 1 = {0:.10f}".format(MLP.run([0, 1])[0]))
print("1 0 = {0:.10f}".format(MLP.run([1, 0])[0]))
print("1 1 = {0:.10f}".format(MLP.run([1, 1])[0]))


'''
print("XOR Gate:")
print("0 0 = {0:.10f}".format(XORGate([0, 0])))
print("0 1 = {0:.10f}".format(XORGate([0, 1])))
print("1 0 = {0:.10f}".format(XORGate([1, 0])))
print("1 1 = {0:.10f}".format(XORGate([1, 1])))
'''
