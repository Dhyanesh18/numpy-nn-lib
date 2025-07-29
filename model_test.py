import numpy as np
from numpy_nn_lib.layer import Dense
from numpy_nn_lib.activation import RELU
from numpy_nn_lib.loss import MSError
from numpy_nn_lib.optimizer import SGD
from numpy_nn_lib.model import Model

X = np.array([[1.0, 2.0], [0.5, -0.6], [-1.5, 2.0]])
y = np.array([[2.5], [-1.2], [3.3]])

# Init the model
model = Model()
model.add(Dense(2,3))
model.add(RELU())
model.add(Dense(3,1))

# Compile the model with loss and optimizer
model.compile(
    loss = MSError(),
    optimizer=SGD(learning_rate=0.1)
)

# Train the model
model.train(X,y,epochs=50, print_every=10)

# Prediction time
preds = model.predict(X)
print("Predictions:\n", preds)
