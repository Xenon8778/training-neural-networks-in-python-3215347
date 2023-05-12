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


class MultilayerPerceptron:
    def __init__(self, layers, bias=1):
        self.layers = np.array(layers, dtype=object)
        self.bias = bias
        self.network = []
        self.values = []
        self.d = []

        for i in range(len(self.layers)):
            self.values.append([])
            self.d.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            self.d[i] = [0.0 for j in range(self.layers[i])]
            if i > 0:
                for j in range(self.layers[i]):
                    self.network[i].append(Perceptron(
                        inputs=self.layers[i-1], bias=self.bias))

        self.network = np.array([np.array(x)
                                for x in self.network], dtype=object)
        self.values = np.array([np.array(x)
                               for x in self.values], dtype=object)
        self.d = np.array([np.array(x)
                            for x in self.d], dtype=object)

    def set_weights(self, w_init):
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i+1][j].set_weights(w_init[i][j])

    def printWeights(self):
        print()
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                print("Layer", i+1, "Neuron", j, self.network[i][j].weights)
        print()

    def run(self, x):
        x = np.array(x, dtype=object)
        self.values[0] = x
        for i in range(1, len(self.layers)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1]


X = [3, 4, 3, 2]
MLP = MultilayerPerceptron(X)

out = MLP.run([2,5,1])

Y = [0,1]
Residual = Y-out
MSE = sum((Residual)**2)/len(Y)
print("MSE: ",MSE)

#calculate output error term
# Ok is the error for each neuron in the output layer only.
# Ok*(1-Ok) this is the derivative of the sigmoid function
deltao = (Residual)*out*(1-out)
print("Delta: ", deltao)

# calculate Hidden layer error term

def bp(self,x,y):
    """Run a single (x,y) pair with the backpropagation algorithm"""
    x = np.array(x,dtype = object)
    y = np.array(y,dtype = object)

#Challenge: Write Backpropagation algorithm.
# Here you have it step by step
# 
# Step 1: Feed a sample to the neural network
# 
# Step 2: calculate the MSE
# 
# Step 3: calculate the output error terms
# 
# Step 4: calculate the error term of each unite on each layer

    for i in reversed(range(1,len(self.network)-1)):
        for h in range(len(self.network[i])):
            fwd_error = 0.0
            for k in range(self.layers[i+1]):
                fwd_error += 
            self.d[i][h] = 