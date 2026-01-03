import numpy as np
from mlp.ds_loader import DatasetLoader
from mlp.preprocessor import ImagePreprocessor

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
      loss = -(y * np.log(a + 1e-8) + (1 - y) * np.log(1 - a + 1e-8))
      return float(loss.item()) # convert (1,1) array to scalar

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

      delta = activations[-1] - y
      nabla_b[-1] = delta
      nabla_w[-1] = np.outer(delta, activations[-2])

      for l in range(2, self.num_layers):
         z = zs[-l]
         sp = self.sigmoid_prime(z)
         delta = np.dot(self.weights[-l+1].T, delta) * sp
         nabla_b[-l] = delta
         nabla_w[-l] = np.outer(delta, activations[-l-1])

      return nabla_w, nabla_b
   
   # iterate over mini-batch
   # run backprop and accumulate gradients
   # update weights and biases
   def update_mini_batch(self, mini_batch, learn_rate):
      nabla_w = [np.zeros(w.shape) for w in self.weights]
      nabla_b = [np.zeros(b.shape) for b in self.biases]

      for x, y in mini_batch:
         delta_nabla_w, delta_nabla_b = self.backpropagation(x, y)
         nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
         nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

      self.weights = [w - (learn_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
      self.biases = [b - (learn_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

   # evaluate average loss and accuracy on dataset
   def evaluate(self, data):
      correct = 0
      total_loss = 0

      for x, y in data:
         activations, _ = self.forward_pass(x)
         a = activations[-1]
         total_loss += self.binary_cross_entropy(y, a)

         prediction = 1 if a >= 0.5 else 0
         if prediction == y:
            correct += 1

      avg_loss = total_loss / len(data)
      accuracy = correct / len(data)

      return avg_loss, accuracy

   # split dataset into mini-batches
   # call update_mini_batch for each mini-batch
   # repeat for multiple epochs
   # compute average loss for each epoch
   def train(self, training_data, epochs, mini_batch_size, learn_rate, validation_data=None):
      n = len(training_data)
      for epoch in range(epochs):
         np.random.shuffle(training_data)
         mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
         for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, learn_rate)

         avg_loss, accuracy = self.evaluate(training_data)
         metrics = f'Epoch {epoch + 1}: Loss = {avg_loss:.3f}: Train Acc = {(accuracy * 100):.1f}%'

         if validation_data:
            _, val_accuracy = self.evaluate(validation_data)
            metrics += f': Val Acc = {(val_accuracy * 100):.1f}%'

         print(metrics)

loader = DatasetLoader(galaxy_dir='./images/galaxies', non_galaxy_dir='./images/non_galaxies', preprocessor=ImagePreprocessor(), flatten=True, augment=True, variations=10)
training_data = loader.load()
split = int(0.8 * len(training_data))
training_set = training_data[:split]
validation_set = training_data[split:]
net = Network(layers=[16384, 64, 32, 1])
net.train(training_data=training_data, epochs=30, mini_batch_size=100, learn_rate=0.1, validation_data=validation_set)