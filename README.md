# Build a Neural Network from Scratch

**Abstract.** 
The widespread adoption of high-level deep learning libraries, while accelerating model development, has increasingly abstracted away the internal mechanics of neural networks, creating a gap between practical usage and fundamental understanding. 
To address this, this project presents a self-contained neural network framework implemented entirely from scratch without relying on automatic differentiation or pre-built deep learning modules. 
The implementation encompasses all essential components, including multi-layer architectures, diverse activation functions, regularization techniques, and state-of-the-art optimizers. 
Beyond serving as a pedagogical instrument that demystifies forward/backward propagation, gradient dynamics, and optimization landscapes, the framework demonstrates robust performance when applied to a multi-class classification task, successfully validating its correctness, numerical stability, and generalization across varied configurations. 
The extensible design and clean modularity further position it as a reliable baseline for educational purposes and future research exploration. 

## Features

### Core Architecture

- **Multiple Hidden Layers** — Build deep neural networks with flexible layer configurations
- **Kaiming Initialization** — Proper weight initialization for stable training (ReLU, Tanh, and GELU)

### Regularization Techniques

- **Weight Decay** — L2 regularization to prevent overfitting
- **Batch Normalization** — Reduces internal covariate shift with running mean/variance for inference
- **Dropout** — Inverted dropout implementation for robust regularization
- **Label Smoothing** — Softens one-hot labels to improve generalization

### Activation Functions

- **ReLU** — Rectified Linear Unit, the most widely used activation function
- **Tanh** — Hyperbolic tangent, zero-centered activation
- **GELU** — Gaussian Error Linear Unit, commonly used in Transformers

### Loss & Evaluation

- **Softmax + Cross-Entropy Loss** — Standard combination for multi-class classification
- **Categorical Accuracy** — Evaluation metric for classification tasks

### Optimization

- **SGD with Momentum** — Accelerated stochastic gradient descent
- **Adam Optimizer** — Adaptive moment estimation with bias correction
- **Mini-batch Training** — Efficient training with batch processing

## Installation

### Prerequisites

- Python 3.8+
- NumPy

### Install from Source

```bash
git clone <repository-url>
cd neural-network
pip install -e .
```

## Quick Start

Here's a minimal example to get you started with building and training a neural network:

```python
import numpy as np
from neural_network import model, dense, relu, batch_norm, dropout 
from neural_network.softmax_cross_entropy import softmax_cross_entropy_loss, softmax_cross_entropy_derivatives

# Build the model
nn = model()
nn.add(dense(input_size=784, output_size=256, init=1, weight_decay=0.001, optimizer=2))
nn.add(batch_norm(dims=256))
nn.add(relu())
nn.add(dropout(p=0.8))
nn.add(dense(input_size=256, output_size=10, init=1, weight_decay=0.001, optimizer=2))
nn.loss(softmax_cross_entropy_loss, softmax_cross_entropy_derivatives)

# Train the model
train_loss, train_acc, val_loss, val_acc = nn.fit(
    train_data=X_train,
    train_label=y_train,
    batch_size=64,
    epochs=50,
    learning_rate=0.001,
    val_data=X_val,
    val_label=y_val,
    val_mode=True
)

# Make predictions
predictions = nn.predict(X_test)
```
