import numpy as np

class RELU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(inputs, 0)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs<=0] = 0