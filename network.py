import numpy as np
from ds_loader import DatasetLoader
from preprocessor import ImagePreprocessor

class Network:
   def __init__(self, layers):
      self.layers = layers
      self.num_layers = len(self.layers)
      self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
      self.biases = [np.random.randn(y, 1) for y in layers[1:]]

   def sigmoid(self, z):
      return 1.0 / (1.0 + np.exp(-z))

   def forward_pass(self, x):
      a = x.reshape(-1, 1)
      for w, b in zip(self.weights, self.biases):
         z = np.dot(w, a) + b
         a = self.sigmoid(z)
      return a
   
   def binary_cross_entropy(self, y, a):
      return -(y * np.log(a + 1e-8) + (1 - y) * np.log(1 - a + 1e-8))

loader = DatasetLoader(galaxy_dir='./images/galaxies', non_galaxy_dir='./images/non_galaxies', preprocessor=ImagePreprocessor(), flatten=True)
data, labels = loader.load()
net = Network([16384, 64, 32, 1])
output = net.forward_pass(data[0])
print('Output:', output)
loss = net.binary_cross_entropy(labels[0], output)
print('Loss:', loss)