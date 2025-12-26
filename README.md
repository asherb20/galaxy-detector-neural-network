# Galaxy Detector Neural Network

## Task

**Binary image classification**

- Input: takes an image (grayscale or RGB)
- Output: Binary classification - galaxy present vs. no galaxy present

## Architecture

**Convolutional Neural Network (CNN)**

```
Input image (e.g., 128x128x3)
↓
Conv2D → ReLU → MaxPooling
↓
Conv2D → ReLU → MaxPooling
↓
Flatten
↓
Dense → ReLU
↓
Dense → Sigmoid (1 output neuron for binary classification)
```

## Resources

[Galaxy Zoo](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo)

## Roadmap

- [x] Write image preprocessor
- [x] Write dataset loader
- [ ] Write neural network
  - [x] Forward pass
  - [x] Loss function
  - [x] Backpropagation
  - [x] Gradient descent
  - [x] Training loop
  - [x] Loss tracking
