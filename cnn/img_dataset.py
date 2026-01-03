import os
from PIL import Image
from torch.utils.data import Dataset

"""Custom PyTorch Dataset for galaxy classification"""
class ImageDataset(Dataset):
    """
      Args:
         galaxy_dir (str): Path to directory with galaxy images
         non_galaxy_dir (str): Path to directory with non-galaxy images
         transform (callable, optional): Optional transform to be applied on images
    """
    def __init__(self, galaxy_dir, non_galaxy_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.load_images(galaxy_dir, label=1) # Load galaxy images (label = 1)
        self.load_images(non_galaxy_dir, label=0) # Load non-galaxy images (label = 0)
    
    """Load all images from a directory"""
    def load_images(self, directory, label):
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist")
            return
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.images.append(filepath)
                self.labels.append(label)
    
    def get_item(self, index):
        """Get image and label at index"""
        img_path = self.images[index]
        label = self.labels[index]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

# img_ds = ImageDataset('./images/galaxies/', './images/non_galaxies/')
# print(img_ds.images)
# print(img_ds.labels)