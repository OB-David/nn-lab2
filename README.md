# NN-PJ1: MNIST Classification

Implement MLP and CNN using a custom deep learning framework for handwritten digit classification on the MNIST dataset.

## Project Structure

```
mynn/                  # Neural network framework
├── op.py             # Linear, Conv2D, MultiCrossEntropyLoss
├── models.py         # Model_MLP, Model_CNN
├── optimizer.py      # SGD, MomentGD
├── lr_scheduler.py   # MultiStepLR, StepLR
├── runner.py         # Training pipeline
└── metric.py         # Evaluation metrics
draw_tools/          # Visualization tools
├── plot.py           # Training curves (loss, accuracy)
├── confusion.py      # Confusion matrix
├── heatmap.py        # Weight heatmap
├── misclassified.py  # Misclassified samples analysis
├── compare_train.py  # Multi-model comparison
└── draw.py           # Hand-drawn input tool
dataset/             # MNIST data
test_train.py        # Training script
test_model.py        # Testing script
```

## Core Implementation

| Module | Content |
|--------|---------|
| `op.py` | Linear layer, Conv2D, MultiCrossEntropyLoss (with built-in Softmax) |
| `models.py` | Model_MLP, Model_CNN |
| `optimizer.py` | SGD, MomentGD |
| `lr_scheduler.py` | MultiStepLR, StepLR |

## Quick Start

**Train Model**
```bash
python test_train.py
```
Modify model type (MLP/CNN), optimizer, learning rate scheduling, and path parameters in the script.

**Test Model**
```bash
python test_model.py
```
Specify the model weight path for evaluation.

## Dataset

[MNIST Handwritten Digit Dataset](http://yann.lecun.com/exdb/mnist/)

## Weights

https://huggingface.co/OB-David/MNIST