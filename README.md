# MNIST Neural Network From Scratch

<p align="center">
  <strong>A feed-forward neural network built entirely from scratch using NumPy — no TensorFlow or PyTorch for training</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge" alt="Matplotlib"/>
</p>

---

## Overview

This project implements a complete neural network from scratch using only NumPy, trained on the MNIST handwritten digits dataset. It demonstrates how fundamental deep learning components — forward propagation, backpropagation, activation functions, dropout, and optimizers — work under the hood.

**Achieves 90%+ accuracy** on MNIST test data with simple hyperparameters.

## Implemented From Scratch

- Forward and Backward Propagation
- Sigmoid and ReLU activation functions
- Softmax output layer
- Cross-Entropy loss function
- Dropout regularization
- SGD and Momentum optimizers
- Training progress visualization

## How It Works

1. **Data Loading & Preprocessing** — Loads MNIST from Keras, normalizes pixel values, one-hot encodes labels
2. **Network Construction** — Custom `NeuralNetwork` class with configurable hidden layers, activations, and dropout
3. **Training** — Batch training with SGD or Momentum optimizer, logging loss and accuracy per epoch
4. **Visualization** — Plots training loss and accuracy curves using Matplotlib

## Results

Typical output after ~200 epochs:

```
Epoch 200: Loss = 0.2435, Train Acc = 0.9365, Test Acc = 0.9268
```

## Requirements

```
numpy
matplotlib
keras
tensorflow
```

```bash
pip install numpy matplotlib keras tensorflow
```

## Usage

```bash
python Code.py
```

Prints epoch-by-epoch loss and accuracy, then displays training curves in a Matplotlib window.

## Key Learnings

- Built a functional neural network entirely from scratch using NumPy
- Learned how optimizers, dropout, and activations affect performance
- Gained deeper understanding of gradient flow and convergence

## License

MIT License — feel free to use, modify, and share with attribution.

## Acknowledgments

- [MNIST dataset](http://yann.lecun.com/exdb/mnist/) by Yann LeCun
- Keras for dataset loading utilities
