# Galaxy Detector Neural Network

## Streamlit App

[Galaxy Detector](https://galaxy-detector-neural-network-zgyjmg4jrhfrgu6y2iftmq.streamlit.app/)

## Task

**Binary image classification**

- Input: takes an image (grayscale or RGB)
- Output: Binary classification - galaxy present vs. no galaxy present

## Architecture

**Convolutional Neural Network (CNN)**

```
Input image (128x128x1 grayscale)
↓
Conv2D (32 filters) → Conv2D (32 filters) → MaxPooling
↓
Conv2D (64 filters) → Conv2D (64 filters) → MaxPooling
↓
AdaptiveAvgPool2d (4x4)
↓
Flatten
↓
Dense (128 neurons) → ReLU
↓
Dense (2 neurons) → Softmax (binary classification: galaxy/non-galaxy)
```

**Loss:** CrossEntropyLoss

**Optimizer:** Stochastic Gradient Descent (SGD)

## Resources

- [Galaxy Zoo](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo)
- [NASA Image & Video Library](https://images.nasa.gov/)
- [Lorem Picsum](https://picsum.photos/)
- [PyTorch Galaxy Datasets](https://github.com/patrikasvanagas/pytorch-galaxy-datasets)

## Roadmap

- [x] Write image preprocessor
- [x] Write dataset loader
- [x] Write neural network
  - [x] Forward pass
  - [x] Loss function
  - [x] Backpropagation
  - [x] Gradient descent
  - [x] Training loop
  - [x] Loss tracking
- [x] Write image extractor
- [x] Data augmentation
- [x] Convert to Convolutional Neural Network (CNN)
- [x] Curate larger training set
- [x] Improve training speed
- [x] Build simple UI for image upload then prediction
- [ ] Expand network to classify galaxy types
