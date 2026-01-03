import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from ds_loader import DatasetLoader
from img_dataset import ImageDataset

# define relevant variables for the ML task
batch_size = 64
num_classes = 2
learning_rate = 0.001
num_epochs = 20
train_split = 0.8

# device will determine whether to run the training on GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the datasets
if __name__ == "__main__":
    # Define paths to image directories
    galaxy_dir = "../images/galaxies"
    non_galaxy_dir = "../images/non_galaxies"

   # Define image transformations
   # Resize to 128x128, convert to tensor, and normalize
    transform = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
      transforms.ToTensor(),  # Converts to tensor with values in [0, 1]
      transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    # Initialize dataset
    img_ds = ImageDataset(galaxy_dir, non_galaxy_dir, transform=transform)
    
    # Create data loaders
    ds_loader = DatasetLoader(img_ds, batch_size, train_split)
    train_loader, test_loader = ds_loader.create_loaders()

    # Display sample batch information
    print("\nSample batch from training set:")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")