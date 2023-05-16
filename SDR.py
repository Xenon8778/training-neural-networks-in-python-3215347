import numpy as np

# 7 segment display recognition
# Will get 7 inputs plus a bias
# 10 outputs
# 1 hidden layers - 7
# [7 input | 7 | 10 output ]


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

        # Step 1: Feed a sample to the neural network
        out = self.run(x) 
        
        # Step 2: calculate the MSE
        Resi = (y-out)
        MSE = sum(Resi**2)/len(y)
        
        # Step 3: calculate the output error terms
        self.d[-1] = out*(1-out)*Resi

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
    

mlp = MultilayerPerceptron(layers = [7,7,10])
mlp.printWeights()
print("\n Training neural network as an SDR Classfier...\n")
epochs = 2000
for i in range(epochs):
    mse = 0.0
    mse += mlp.bp([1,1,1,1,1,1,0],[1,0,0,0,0,0,0,0,0,0])    #0 pattern
    mse += mlp.bp([0,1,1,0,0,0,0],[0,1,0,0,0,0,0,0,0,0])    #1 pattern
    mse += mlp.bp([1,1,0,1,1,0,1],[0,0,1,0,0,0,0,0,0,0])    #2 pattern
    mse += mlp.bp([1,1,1,1,0,0,1],[0,0,0,1,0,0,0,0,0,0])    #3 pattern
    mse += mlp.bp([0,1,1,0,0,1,1],[0,0,0,0,1,0,0,0,0,0])    #4 pattern
    mse += mlp.bp([1,0,1,1,0,1,1],[0,0,0,0,0,1,0,0,0,0])    #5 pattern
    mse += mlp.bp([1,0,1,1,1,1,1],[0,0,0,0,0,0,1,0,0,0])    #6 pattern
    mse += mlp.bp([1,1,1,0,0,0,0],[0,0,0,0,0,0,0,1,0,0])    #7 pattern
    mse += mlp.bp([1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,1,0])    #8 pattern
    mse += mlp.bp([1,1,1,1,0,1,1],[0,0,0,0,0,0,0,0,0,1])    #9 pattern
    mse = mse/10.0
    if (i%100 == 0):
        print (mse)

mlp.printWeights()
def pattern2num(x):
    x = list(1 if i==max(x) else 0 for i in x)
    match x:
      case [1,0,0,0,0,0,0,0,0,0]:
          return 0
      case [0,1,0,0,0,0,0,0,0,0]:
          return 1
      case [0,0,1,0,0,0,0,0,0,0]:
          return 2
      case [0,0,0,1,0,0,0,0,0,0]:
          return 3
      case [0,0,0,0,1,0,0,0,0,0]:
          return 4
      case [0,0,0,0,0,1,0,0,0,0]:
          return 5
      case [0,0,0,0,0,0,1,0,0,0]:
          return 6
      case [0,0,0,0,0,0,0,1,0,0]:
          return 7
      case [0,0,0,0,0,0,0,0,1,0]:
          return 8
      case [0,0,0,0,0,0,0,0,0,1]:
          return 9
      
print('MLP:')
print(mlp.run([1,1,1,1,1,1,1]))
print("8 = %d"%pattern2num(mlp.run([1,1,1,1,1,1,1])))
print("4 = %d"%pattern2num(mlp.run([0,1,1,0,0,1,1])))
print("1 = %d"%pattern2num(mlp.run([0,1,1,0,0,0,0])))