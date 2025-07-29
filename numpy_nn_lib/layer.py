import numpy as np

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons) # W
        self.biases = np.zeros((1,n_neurons)) # b

    def forward(self, inputs):
        self.inputs = inputs # x
        self.output = np.dot(inputs, self.weights) + self.biases # W.x + b

    def backward(self, dvalues): # dvalues = dL/dy
        self.dweights = np.dot(self.inputs.T, dvalues) # dL/dw
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)# dL/db
        self.dinputs = np.dot(dvalues, self.weights.T) # dL/dx -> pass into the previous layer