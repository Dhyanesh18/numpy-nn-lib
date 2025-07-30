import numpy as np

class RELU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(inputs, 0)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs<=0] = 0


class Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs 
        self.output = 1/(1 + np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (self.output * (1 - self.output))


class Tanh:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output**2)


class SoftmaxCCE:
    """
    We are combining Softmax and CCE here, so that we wouldn't have to calculate Jacobian matrices
    """
    def forward(self, inputs, y_true):
        # Softmax
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.y_true = y_true

        # Categorical cross-entropy
        samples = len(inputs)
        clipped_output = np.clip(self.output, 1e-7, 1 - 1e-7)
        correct_confidences = clipped_output[range(samples), y_true]
        loss = -np.log(correct_confidences)
        return np.mean(loss)

    def backward(self):
        samples = len(self.output)
        self.dinputs = self.output.copy()
        self.dinputs[range(samples), self.y_true] -= 1
        self.dinputs /= samples
