import numpy as np

class Dense:
    def __init__(self, n_inputs, n_neurons, l1, l2):
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons) # W
        self.biases = np.zeros((1,n_neurons)) # b
        self.l1 = l1
        self.l2 = l2 

    def forward(self, inputs):
        self.inputs = inputs # x
        self.output = np.dot(inputs, self.weights) + self.biases # W.x + b

    def backward(self, dvalues): # dvalues = dL/dy
        self.dweights = np.dot(self.inputs.T, dvalues) # dL/dw
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)# dL/db
        self.dinputs = np.dot(dvalues, self.weights.T) # dL/dx -> pass into the previous layer
        if self.l1 > 0: # Lasso regularization
            self.dweights += self.l1 * np.sign(self.weights) # dL/dw = dL/dw + l1 * sign(w)
        if self.l2 > 0: # Ridge regularization
            self.dweights += 2 * self.l2 * self.weights # dL/dw = dL/dw + 2*l2*w