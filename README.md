# NumPyNNLib 

**A neural network library built from scratch with pure NumPy.**

This is a simple but fully working neural network framework, no TensorFlow or PyTorch black box here!  
It’s designed to **showcase my understanding of how deep learning works inside**: layers, activations, losses, optimizers, backprop, all built manually.

---

## Features

- Dense (fully connected) layers  
- Multiple activation functions (ReLU, Sigmoid, Tanh, Softmax)  
- Loss functions (MSE, MAE, Binary Cross-Entropy, Categorical Cross-Entropy, Huber)  
- Gradient backpropagation for each component  
- Simple optimizers (SGD, optional momentum)  
- Modular `Model` class with `add()`, `compile()`, `train()`, `evaluate()`, `predict()`  
- Extensible — add BatchNorm, Dropout, Convolutions, and more

---

## Why I built this

I wanted to **build my own** NN library, to **revise what i learnt a long time ago**:  
- How forward and backward passes flow through layers  
- How gradients are calculated for weights, biases, and inputs  
- How optimizers adjust parameters  
- Why activation functions and loss functions need to be handled carefully for stable training

---
