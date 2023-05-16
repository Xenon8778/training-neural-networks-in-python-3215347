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
        x_sum = np.dot(np.append(x, self.bias), self.weights)
        return self.sigmoid(x_sum)

    def set_weights(self, w_init):
        """Set the weights. w_init is a python list with the weights."""
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        """Evaluate the sigmoid function for the floating point input x."""
        return 1/(1+np.exp(-x))


class MultilayerPerceptron:
    def __init__(self, layers, bias=1, eta=0.5):
        self.layers = np.array(layers, dtype=object)
        self.bias = bias
        self.eta = eta
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
    
    def bp(self,x,y):
        """Run a single (x,y) pair with the backpropagation algorithm"""
        x = np.array(x,dtype = object)
        y = np.array(y,dtype = object)
        #Challenge: Write Backpropagation algorithm.
        # Here you have it step by step
        # 
        # Step 1: Feed a sample to the neural network
        o = self.run(x) 
        
        # Step 2: calculate the MSE
        Resi = (y-o)
        MSE = sum(Resi**2)/len(y)
        
        # Step 3: calculate the output error terms
        self.d[-1] = o*(1-o)*Resi

        # Step 4: calculate the error term of each unite on each layer
        for i in reversed(range(1,len(self.network)-1)):
            for h in range(len(self.network[i])):
                fwd_error = 0.0
                for k in range(self.layers[i+1]):
                    fwd_error += self.network[i+1][k].weights[h]*self.d[i+1][k]
                self.d[i][h] = fwd_error*self.values[i][h]*(1-self.values[i][h])
        
        # Step 5 and 6: calculate the deltas and update the weights
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                for k in range(self.layers[i-1]+1):
                    if k == self.layers[i-1]:
                        delta = self.eta*self.d[i][j]*self.bias
                    else:
                        delta=self.eta*self.d[i][j]*self.values[i-1][k]
                    self.network[i][j].weights[k] += delta
        return MSE


#Test code
mlp = MultilayerPerceptron(layers = [2,2,1])
mlp.printWeights()
print("\n Training neural network as an XOR Gate...\n")
for i in range(3000):
    mse = 0.0
    mse += mlp.bp([0,0],[0])
    mse += mlp.bp([0,1],[1])
    mse += mlp.bp([1,0],[1])
    mse += mlp.bp([1,1],[0])
    mse = mse/4
    if (i%100 == 0):
        print (mse)

mlp.printWeights()

print('MLP:')
print("0 0 = {0:.10f}".format(mlp.run([0,0])[0]))
print("0 1 = {0:.10f}".format(mlp.run([0,1])[0]))
print("1 0 = {0:.10f}".format(mlp.run([1,0])[0]))
print("1 1 = {0:.10f}".format(mlp.run([1,1])[0]))