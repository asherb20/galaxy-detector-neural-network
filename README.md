# Galaxy Detector Neural Network

## Task

**Binary image classification**

- Input: takes an image (grayscale or RGB)
- Output: Binary classification - galaxy present vs. no galaxy present

## Architecture

Large fully connected classifier (MLP)

## Resources

[Galaxy Zoo](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo)

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
- Curate larger training set
