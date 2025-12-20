import cv2

class ImagePreprocessor:
   def __init__(self, image_path, size=(128, 128), color=cv2.COLOR_BGR2GRAY):
      self.image_path = image_path
      self.size = size
      self.color = color

   def preprocess(self):
      image = cv2.imread(self.image_path)
      # resize to 128x128
      image = cv2.resize(image, self.size)
      # convert to grayscale
      image = cv2.cvtColor(image, self.color)
      # normalize the image
      image = image / 255.0

      return image

image_preprocessor = ImagePreprocessor('./images/galaxies/b01708ea-06a2-40e1-8327-505d4e2f53d5.png')
image_data = image_preprocessor.preprocess()
print(image_data)
print(image_data.shape)  # Output: (128, 128)