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
   
   def sigmoid_prime(self, z):
      return self.sigmoid(z) * (1 - self.sigmoid(z))
   
   def binary_cross_entropy(self, y, a):
      return -(y * np.log(a + 1e-8) + (1 - y) * np.log(1 - a + 1e-8))

   # iterate over each layer
   # calculate the activation for each neuron
   # return all activations and the weighted inputs (zs)
   def forward_pass(self, x):
      activation = x.reshape(-1, 1)
      activations = [activation]
      
      zs = []

      for w, b in zip(self.weights, self.biases):
         z = np.dot(w, activation) + b
         zs.append(z)
         activation = self.sigmoid(z)
         activations.append(activation)

      return activations, zs
   
   # output error
   # backrop through hidden layers
   def backpropagation(self, x, y):
      nabla_w = [np.zeros(w.shape) for w in self.weights]
      nabla_b = [np.zeros(b.shape) for b in self.biases]

      activations, zs = self.forward_pass(x)

      delta = (activations[-1] - y) * self.sigmoid_prime(zs[-1])
      nabla_b[-1] = delta
      nabla_w[-1] = np.outer(delta, activations[-2])

      for l in range(2, self.num_layers):
         z = zs[-l]
         sp = self.sigmoid_prime(z)
         delta = np.dot(self.weights[-l+1].T, delta) * sp
         nabla_b[-l] = delta
         nabla_w[-l] = np.outer(delta, activations[-l-1])

      return nabla_w, nabla_b

loader = DatasetLoader(galaxy_dir='./images/galaxies', non_galaxy_dir='./images/non_galaxies', preprocessor=ImagePreprocessor(), flatten=True)
data, labels = loader.load()
net = Network([16384, 64, 32, 1])
output = net.forward_pass(data[0])
print('Output:', output)
loss = net.binary_cross_entropy(labels[0], output)
print('Loss:', loss)
nabla_w, nabla_b = net.backpropagation(data, labels)
print('Delta weights:', nabla_w)
print('Delta biases:', nabla_b)