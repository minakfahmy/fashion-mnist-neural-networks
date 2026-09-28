# Deep Learning on Fashion-MNIST

A PyTorch deep learning project exploring neural network classification, optimization, loss-landscape analysis, and unsupervised representation learning using the Fashion-MNIST dataset.

The project implements and evaluates Multi-Layer Perceptrons (MLPs), systematic hyperparameter tuning, architectural improvements with Batch Normalization and LeakyReLU, PCA-based visualization of SGD optimization trajectories, and an autoencoder for image reconstruction.

## Features

- Multi-Layer Perceptron (MLP) image classifier built with PyTorch
- Mini-batch Stochastic Gradient Descent (SGD)
- Systematic hyperparameter tuning using a validation set
- Batch Normalization and LeakyReLU architecture experiments
- Cross-entropy loss and L2 regularization
- PCA visualization of neural network parameter trajectories
- 3D neural network loss-landscape visualization
- Autoencoder for unsupervised representation learning
- Fashion-MNIST image reconstruction
- GPU acceleration with PyTorch/CUDA when available

## Technologies

- Python
- PyTorch
- NumPy
- scikit-learn
- Matplotlib
- torchvision
- Google Colab

## Neural Network Classification

The classification model is a fully connected neural network that maps a flattened Fashion-MNIST image to one of 10 clothing categories.

```text
28 × 28 Image
      ↓
784 Features
      ↓
Hidden Layer + ReLU
      ↓
Hidden Layer + ReLU
      ↓
      ...
      ↓
10 Output Logits
      ↓
Predicted Class
```

Multiple network configurations are evaluated by varying hyperparameters including:

- Number of hidden layers
- Number of hidden units
- Learning rate
- Mini-batch size
- Number of epochs
- L2 regularization strength

Model selection is performed using validation-set performance before evaluating the final model on the test set.

## Model Optimization

The baseline MLP is compared with an enhanced neural network architecture incorporating:

- Batch Normalization
- LeakyReLU activations
- SGD optimization
- L2 regularization

These experiments investigate how architectural choices affect training behavior and classification performance.

## Loss Landscape Visualization

To analyze the optimization process, neural network parameters are recorded throughout SGD training.

Because the complete parameter vector is high-dimensional, Principal Component Analysis (PCA) is used to project the parameter trajectory onto its two primary directions of variation.

The cross-entropy loss is then evaluated across this two-dimensional parameter space to construct a 3D loss landscape.

```text
High-Dimensional Model Parameters
              ↓
             PCA
              ↓
     Principal Components
          PC1     PC2
            \     /
             \   /
        Loss Landscape
              +
        SGD Trajectory
```

This visualization illustrates how SGD moves through the neural network parameter space toward regions of lower loss.

## Autoencoder

The project also implements a dense autoencoder for unsupervised representation learning.

### Architecture

```text
Input Image
   784
    ↓
   128
    ↓
    64
Bottleneck
    ↓
   128
    ↓
   784
Reconstructed Image
```

The encoder compresses each 784-dimensional image into a 64-dimensional latent representation.

The decoder reconstructs the original image from this compressed representation.

The autoencoder is trained using:

- Mean Squared Error (MSE) loss
- Adam optimizer
- ReLU hidden activations
- Sigmoid output activation

## Dataset

The project uses the Fashion-MNIST dataset, consisting of 28 × 28 grayscale images across 10 fashion categories.

Each image is flattened into a 784-dimensional vector for the fully connected neural networks.

## Results

The experiments evaluate:

- Classification accuracy
- Training and validation performance
- Hyperparameter configurations
- Architectural improvements
- SGD optimization trajectories
- Autoencoder reconstruction quality

> Add your final measured validation and test accuracies here after completing the experiments.

## Project Structure

```text
deep-learning-fashion-mnist/
│
├── notebooks/
│   └── fashion_mnist_deep_learning.ipynb
│
├── src/
│   └── fashion_mnist_models.py
│
├── results/
│   ├── loss_landscape.png
│   └── autoencoder_reconstructions.png
│
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/deep-learning-fashion-mnist.git
cd deep-learning-fashion-mnist
```

Install the dependencies:

```bash
pip install torch torchvision numpy scikit-learn matplotlib
```

## Skills Demonstrated

This project demonstrates practical experience with:

- Deep Learning
- PyTorch
- Neural Network Architecture
- Model Training & Evaluation
- Hyperparameter Optimization
- Computer Vision
- SGD & Adam Optimization
- Batch Normalization
- Representation Learning
- Autoencoders
- Principal Component Analysis (PCA)
- Loss Landscape Analysis
- Data Visualization

## Author

**Mina Fahmy**

GitHub: [minakfahmy](https://github.com/minakfahmy)
