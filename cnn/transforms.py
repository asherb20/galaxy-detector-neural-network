from torchvision import transforms

"""Image transforms shared by training (cnn/handler.py) and inference (app.py)"""

IMAGE_SIZE = 128
IN_CHANNELS = 3

# GZ2 galaxies sit in the middle of a 424px frame, so crop away most of the empty sky
GZ2_CROP_FRACTION = 0.6

NORMALIZE = transforms.Normalize(mean=[0.5] * IN_CHANNELS, std=[0.5] * IN_CHANNELS)  # [-1, 1]


# module-level callables (not lambdas) so DataLoader workers can pickle them on Windows
def to_rgb(image):
   return image.convert('RGB')


class CenterCropFraction:
   def __init__(self, fraction):
      self.fraction = fraction

   def __call__(self, image):
      width, height = image.size
      size = int(min(width, height) * self.fraction)
      return transforms.functional.center_crop(image, [size, size])


def _source_steps(crop_fraction):
   steps = [transforms.Lambda(to_rgb)]
   if crop_fraction is not None and crop_fraction < 1:
      steps.append(CenterCropFraction(crop_fraction))
   return steps


def train_transform(crop_fraction=None):
   """Augmented pipeline; galaxies look the same at any rotation or reflection"""
   return transforms.Compose(_source_steps(crop_fraction) + [
      transforms.RandomRotation(180, interpolation=transforms.InterpolationMode.BILINEAR),
      transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
      transforms.ToTensor(),
      NORMALIZE
   ])


def eval_transform(crop_fraction=None):
   """Deterministic pipeline; app.py uses it with no crop"""
   return transforms.Compose(_source_steps(crop_fraction) + [
      transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
      transforms.ToTensor(),
      NORMALIZE
   ])
