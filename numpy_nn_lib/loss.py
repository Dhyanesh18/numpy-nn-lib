import numpy as np


class MSError:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred 
        self.y_true = y_true
        return np.mean((y_true-y_pred)**2)

    def backward(self):
        samples = len(self.y_true)
        self.dinputs = -2 * (self.y_true - self.y_pred) / samples


class MAE:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean(np.abs(y_true - y_pred))

    def backward(self):
        samples = len(self.y_true)
        self.dinputs = np.where(self.y_pred > self.y_true, 1, -1) / samples


class Huber:
    def __init__(self, delta=1.0):
        self.delta = delta

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        error = y_true - y_pred
        is_small = np.abs(error) <= self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)
        return np.mean(np.where(is_small, squared_loss, linear_loss))

    def backward(self):
        samples = len(self.y_true)
        error = self.y_true - self.y_pred
        is_small = np.abs(error) <= self.delta
        self.dinputs = np.where(is_small, -error, -self.delta * np.sign(error)) / samples



class BinaryCrossEntropy:
    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.y_true = y_true
        loss = - (y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred))
        return np.mean(loss)

    def backward(self):
        samples = len(self.y_true)
        self.dinputs = -(self.y_true / self.y_pred - (1 - self.y_true) / (1 - self.y_pred)) / samples



