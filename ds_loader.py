import os
import numpy as np

class DatasetLoader:
   def __init__(self, galaxy_dir, non_galaxy_dir, preprocessor, flatten=False):
      self.galaxy_dir = galaxy_dir
      self.non_galaxy_dir = non_galaxy_dir
      self.preprocessor = preprocessor
      self.flatten = flatten
      self.data = []

   # loop through directory
   # preprocess each image
   # append tuple data
   def append_data(self, dir, label):
      with os.scandir(dir) as entries:
         for entry in entries:
            if entry.is_file():
               img_data = self.preprocessor.preprocess(entry.path)
               if self.flatten:
                  img_data = img_data.flatten()
               self.data.append((img_data, label))

   # append data for each directory
   # shuffle and return data
   def load(self):
      self.append_data(self.galaxy_dir, 1)
      self.append_data(self.non_galaxy_dir, 0)
      return self.data