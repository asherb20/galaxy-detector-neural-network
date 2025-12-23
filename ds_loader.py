import os
import numpy as np
from preprocessor import ImagePreprocessor

class DatasetLoader:
   def __init__(self, galaxy_dir, non_galaxy_dir, preprocessor, flatten=False):
      self.galaxy_dir = galaxy_dir
      self.non_galaxy_dir = non_galaxy_dir
      self.preprocessor = preprocessor
      self.flatten = flatten
      self.data = []
      self.labels = []

   # loop through directory
   # preprocess each image
   # append data
   # append label
   def append_data(self, dir, label):
      with os.scandir(dir) as entries:
         for entry in entries:
            if entry.is_file():
               img_data = self.preprocessor.preprocess(entry.path)
               if self.flatten:
                  img_data = img_data.flatten()
               self.data.append(img_data)
               self.labels.append(label)

   # generate random permutation of indices
   # apply permutation to data and labels
   def shuffle_data(self, data, labels):
      indices = np.arange(len(data))
      np.random.shuffle(indices)
      return data[indices], labels[indices]

   # append data and labels for each directory
   # return data and labels
   def load(self):
      self.append_data(self.galaxy_dir, 1)
      self.append_data(self.non_galaxy_dir, 0)

      return self.shuffle_data(np.array(self.data), np.array(self.labels))

# loader = DatasetLoader(galaxy_dir='./images/galaxies', non_galaxy_dir='./images/non_galaxies', preprocessor=ImagePreprocessor(), flatten=True)
# x, y = loader.load()
# print('data shape:', x.shape)
# print('label shape:', y.shape)
# print('1st 5 labels:', y[:5])