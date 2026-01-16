import cv2
import numpy as np

class ImagePreprocessor:
   def __init__(self, size=(64, 64), color=cv2.COLOR_BGR2GRAY):
      self.size = size
      self.color = color

   def preprocess(self, img_path, augment=False):
      image = cv2.imread(img_path)
      image = cv2.resize(image, self.size)
      image = cv2.cvtColor(image, self.color)
      image = image / 255.0

      if augment:
         image = self.augment(image)

      image = image.reshape(self.size[0], self.size[1], 1)

      return image
   
   def augment(self, image):
      # random horizontal flip
      if np.random.rand() > 0.5:
         image = cv2.flip(image, 1)

      # random vertical flip
      if np.random.rand() > 0.5:
         image = cv2.flip(image, 0)

      # random rotation
      if np.random.rand() > 0.5:
         angle = np.random.randint(-30, 30)
         M = cv2.getRotationMatrix2D((self.size[0]//2, self.size[1]//2), angle, 1)
         image = cv2.warpAffine(image, M, self.size)

      # random brightness adjustment
      if np.random.rand() > 0.5:
         alpha = 1.0 + (np.random.rand() - 0.5) * 0.4  # scale factor 0.8 - 1.2
         beta = np.random.randint(-20, 20)  # shift -20 to +20
         image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta) / 255.0

      # random Gaussian noise
      if np.random.rand() > 0.5:
         noise = np.random.normal(0, 0.05, image.shape)
         image = np.clip(image + noise, 0, 1)

      return image