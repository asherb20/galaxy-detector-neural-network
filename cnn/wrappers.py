import os
import torch
from PIL import Image
from cnn.morphology import IGNORE

"""Dataset adapters; every item is (image, detect_label, morph_label)"""

GALAXY = 1
NON_GALAXY = 0
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def _extract_image(item):
   # GZ2 items are dicts, torchvision datasets return (image, target) tuples
   if isinstance(item, dict):
      return item['image']
   if isinstance(item, (tuple, list)):
      return item[0]
   return item


# Galaxy images (1 = galaxy) with their morphology label (or IGNORE)
class GalaxyWrapper(torch.utils.data.Dataset):
   def __init__(self, gz2_dataset, indices, morph_labels):
      self.gz2_dataset = gz2_dataset
      self.indices = list(indices)
      self.morph_labels = morph_labels

   def __len__(self):
      return len(self.indices)

   def __getitem__(self, idx):
      source_idx = self.indices[idx]
      image = _extract_image(self.gz2_dataset[source_idx])
      return image, GALAXY, int(self.morph_labels[source_idx])


# Non-galaxy images (0 = not galaxy); never train the morphology head
class NonGalaxyWrapper(torch.utils.data.Dataset):
   def __init__(self, dataset, indices=None):
      self.dataset = dataset
      self.indices = list(range(len(dataset))) if indices is None else list(indices)

   def __len__(self):
      return len(self.indices)

   def __getitem__(self, idx):
      image = _extract_image(self.dataset[self.indices[idx]])
      return image, NON_GALAXY, IGNORE


# Every image file under a folder (recursively), with no labels
class ImageFiles(torch.utils.data.Dataset):
   def __init__(self, root, transform=None):
      self.transform = transform
      self.paths = sorted(
         os.path.join(folder, name)
         for folder, _, names in os.walk(root)
         for name in names
         if name.lower().endswith(IMAGE_EXTENSIONS)
      )

   def __len__(self):
      return len(self.paths)

   def __getitem__(self, idx):
      with Image.open(self.paths[idx]) as image:
         image = image.convert('RGB')
      return self.transform(image) if self.transform else image
