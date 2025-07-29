import numpy as np
from numpy_nn_lib.layer import Dense
from numpy_nn_lib.activation import RELU
from numpy_nn_lib.loss import MSError
from numpy_nn_lib.optimizer import SGD

X = np.array([[1.0, 2.0], [0.5, -0.6], [-1.5, 2.0]])
y = np.array([[2.5], [-1.2], [3.3]])

# Network Architecture
dense1 = Dense(2,3)
activation1 = RELU()
dense2 = Dense(3,1)
loss_fn = MSError()
optimizer = SGD(learning_rate=0.1)


# Training Step
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)


# Compute Loss
loss = loss_fn.forward(dense2.output, y)

# Back prop
loss_fn.backward()
dense2.backward(loss_fn.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)


# Update params
optimizer.update_params(dense1)
optimizer.update_params(dense2)

print("Loss:", loss)

