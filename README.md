# Deep Neural Network from Scratch

This repository contains a **Deep Neural Network (DNN)** built entirely from scratch using **Python**. No pre-built machine learning libraries such as TensorFlow or PyTorch were used. The goal is to provide a clear understanding of how deep neural networks function at the mathematical and algorithmic level.

## 🧠 Project Overview

In this project, we create a simple neural network with multiple layers and demonstrate the following key concepts:

- **Weight and Bias Matrices**: Understand the dimensions and initialization of these matrices.
- **Forward Propagation**: Learn how input data is passed through the network and how activations are computed.
- **Backpropagation**: Understand how the gradients of the loss function with respect to weights and biases are computed using the chain rule.
- **Gradient Descent**: See how the parameters (weights and biases) are updated to minimize the loss and improve the model's accuracy.

## 🔧 Features

- **No external libraries**: The code is implemented from scratch without using libraries like TensorFlow, PyTorch, or Keras. 
- **Mathematical Explanations**: Brief explanations of the mathematical principles behind forward propagation, backpropagation, and gradient descent, including matrix dimensions.
- **High accuracy**: The model achieves **93% accuracy** on the training data and **92% accuracy** on the test data, demonstrating the effectiveness of this approach.

## 📚 How It Works

### 1. **Forward Propagation**
   The input data is passed through the network layer by layer. The activations for each layer are computed by multiplying the input by the weight matrix, adding the bias, and applying an activation function (e.g., ReLU or Sigmoid).

### 2. **Backpropagation**
   We calculate the **derivatives** of the loss function with respect to the weights and biases using the **chain rule** of calculus. These derivatives help us adjust the weights and biases to minimize the loss.

### 3. **Gradient Descent**
   Once the gradients are computed, we use **gradient descent** to update the weights and biases by subtracting a fraction (learning rate) of the gradients from them. This process is repeated iteratively to improve the model's performance.

### 4. Download the dataset: 
You can get the dataset for this project from Kaggle by visiting the following link:
https://www.kaggle.com/competitions/digit-recognizer/data

## 🔍 Model Evaluation

- **Training Accuracy**: The model achieves approximately **93% accuracy** on the training data.
- **Test Accuracy**: The model generalizes well to the test data with an accuracy of around **92%**.

## 💬 Contributing

If you find any issues or have suggestions for improvements, feel free to open an **issue** or submit a **pull request**.
