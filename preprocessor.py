import cv2

class ImagePreprocessor:
   def __init__(self, size=(128, 128), color=cv2.COLOR_BGR2GRAY):
      self.size = size
      self.color = color

   def preprocess(self, img_path):
      image = cv2.imread(img_path)
      image = cv2.resize(image, self.size)
      image = cv2.cvtColor(image, self.color)
      image = image / 255.0
      image = image.reshape(self.size[0], self.size[1], 1)

      return image