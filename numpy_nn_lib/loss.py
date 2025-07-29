import numpy as np

class MSError:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred 
        self.y_true = y_true
        return np.mean((y_true-y_pred)**2)

    def backward(self):
        samples = len(self.y_true)
        self.dinputs = -2 * (self.y_true - self.y_pred) / samples
