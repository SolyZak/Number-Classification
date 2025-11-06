# Number-Classification

MNIST Neural Network From Scratch 

This project implements a feed-forward neural network from scratch in NumPy, trained and tested on the MNIST handwritten digits dataset. It demonstrates how fundamental components of deep learning — forward propagation, backpropagation, activation functions, dropout, and optimizers — work behind the scenes without relying on high-level frameworks like TensorFlow or PyTorch.

Features

Manual implementation of:

Forward and Backward Propagation

Sigmoid and ReLU activations

Softmax output layer

Cross-Entropy loss

Dropout regularization

SGD and Momentum optimizers

Visualized training progress with Matplotlib

Achieves over 90% accuracy on MNIST test data with simple hyperparameters

How It Works

Data Loading & Preprocessing – Loads MNIST from Keras, normalizes pixel values, and one-hot encodes labels.

Neural Network Construction – Custom NeuralNetwork class with configurable hidden layers, activation functions, and dropout; supports ReLU or Sigmoid activations with Softmax output.

Training – Uses SGD or Momentum optimizers; performs batch training and logs loss, training accuracy, and test accuracy per epoch.

Visualization – Plots training loss and accuracy curves using Matplotlib.

Example Results

Typical output after ~200 epochs:

Epoch 200: Loss = 0.2435, Train Acc = 0.9365, Test Acc = 0.9268


This confirms the correctness and efficiency of the backpropagation and training loop.

Example Graphs

Left: Training Loss  Right: Training vs Testing Accuracy
(After running, Matplotlib automatically displays these graphs.)

Requirements
numpy
matplotlib
keras
tensorflow


Install dependencies:

pip install -r requirements.txt

How to Run
python mnist_nn.py


You’ll see printed logs of each epoch’s loss and accuracy, followed by training curves in a Matplotlib window.

Key Learnings

Built a functional neural network entirely from scratch using NumPy

Learned how optimizers, dropout, and activations affect performance

Gained deeper understanding of gradient flow and convergence

Contributors

Soliman Zakaria – Core development, architecture design, and training pipeline

License

MIT License — feel free to use, modify, and share with attribution.

Acknowledgments

MNIST dataset from Yann LeCun’s Database

Keras for dataset loading utilities
