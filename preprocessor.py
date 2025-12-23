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

# image_preprocessor = ImagePreprocessor()
# image_data = image_preprocessor.preprocess('./images/galaxies/b01708ea-06a2-40e1-8327-505d4e2f53d5.png')
# print(image_data.shape)
# image_vector = image_data.flatten()
# print(image_vector.shape)