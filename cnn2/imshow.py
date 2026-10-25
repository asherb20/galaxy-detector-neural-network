import matplotlib.pyplot as plt
import numpy as np
import torchvision
from loader import trainloader, batch_size, classes

def imshow(img):
   img = img / 2 + 0.5 # unnormalize
   npimg = img.numpy()
   plt.imshow(np.transpose(npimg, (1, 2, 0)))
   plt.show()

if __name__ == '__main__':
   # get some random training images
   dataiter = iter(trainloader)
   images, labels = next(dataiter)

   # show images
   imshow(torchvision.utils.make_grid(images))
   # print labels
   print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))