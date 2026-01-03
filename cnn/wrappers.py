import torch

# Create wrapper to convert GZ2 to binary labels (1 = galaxy)
class GalaxyWrapper(torch.utils.data.Dataset):
   def __init__(self, gz2_dataset):
      self.gz2_dataset = gz2_dataset
   
   def __len__(self):
      return len(self.gz2_dataset)
   
   def __getitem__(self, idx):
      batch = self.gz2_dataset[idx]
      image = batch['image']
      # Label: 1 for galaxy (all GZ2 samples are galaxies)
      label = torch.tensor(1, dtype=torch.long)
      return image, label
   
# Create wrapper to assign non-galaxy labels (0 = not galaxy)
class NonGalaxyWrapper(torch.utils.data.Dataset):
   def __init__(self, fake_dataset):
      self.fake_dataset = fake_dataset
   
   def __len__(self):
      return len(self.fake_dataset)
   
   def __getitem__(self, idx):
      image, _ = self.fake_dataset[idx]
      # Label: 0 for non-galaxy
      label = torch.tensor(0, dtype=torch.long)
      return image, label