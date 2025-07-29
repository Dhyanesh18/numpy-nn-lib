import numpy as np

class Model:
    """
    A simple sequential neural network model.

    Supports adding layers, compiling with loss and optimizer,
    training for regression tasks, and making predictions.

    Example:
        model = Model()
        model.add(Dense(2, 3))
        model.add(RELU())
        model.add(Dense(3, 1))
        model.compile(loss=MSError(), optimizer=SGD(learning_rate=0.1))
        model.train(X, y, epochs=1000)
        preds = model.predict(X)
    """

    def __init__(self):
        """
        Initialize an empty model.
        """
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        """
        Add a layer to the model.

        Args:
            layer: A layer object (e.g., Dense, Activation).
        """
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        """
        Configure the model for training.

        Args:
            loss: Loss function object (must have forward and backward).
            optimizer: Optimizer object (must have update_params).
        """
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X):
        """
        Perform a forward pass through all layers.

        Args:
            X: Input data.

        Returns:
            The model's final output.
        """
        output = X
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def backward(self):
        """
        Perform a backward pass through the network.

        Starts with loss gradient and propagates backward through all layers.
        """
        self.loss.backward()
        dinputs = self.loss.dinputs
        for layer in reversed(self.layers):
            layer.backward(dinputs)
            dinputs = layer.dinputs

    def update_params(self):
        """
        Update the weights and biases of trainable layers.

        Uses the configured optimizer.
        """
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                self.optimizer.update_params(layer)

    def train(self, X, y, epochs=50, print_every=10):
        """
        Train the model for a given number of epochs.

        Args:
            X: Input data.
            y: True target values.
            epochs: Number of training epochs.
            print_every: How often to print loss.
        """
        for epoch in range(1, epochs + 1):
            # Forward pass
            output = self.forward(X)

            # Compute loss
            loss = self.loss.forward(output, y)

            # Backward pass
            self.backward()

            # Update weights
            self.update_params()

            # Logging
            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Make predictions with the model (inference only).

        Args:
            X: Input data.

        Returns:
            The model's predicted output.
        """
        return self.forward(X)

    def evaluate(self, X, y):
        """
        Evaluate the model on given data.

        Args:
            X: Input data.
            y: True target values.

        Returns:
            A dictionary with loss and metrics.
        """
        predictions = self.forward(X)
        loss = self.loss.forward(predictions, y)

        # Optionally compute more metrics:
        mae = np.mean(np.abs(y - predictions))
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        print(f"Evaluation - Loss: {loss:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        return {
            'loss': loss,
            'mae': mae,
            'r2': r2
        }
