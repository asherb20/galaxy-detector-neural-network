import torch
from torch.utils.data import DataLoader

class DatasetLoader:
    def __init__(self, dataset=None, batch_size=32, train_split=0.8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_split = train_split

    """
    Create training and test data loaders
    
    Args:
        galaxy_dir (str): Path to galaxy images directory
        non_galaxy_dir (str): Path to non-galaxy images directory
        batch_size (int): Batch size for data loader
        train_split (float): Proportion of data to use for training (default: 0.8)
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    def create_loaders(self):
        print(f"Total images loaded: {len(self.dataset.images)}")
        print(f"Galaxy images: {sum(1 for l in self.dataset.labels if l == 1)}")
        print(f"Non-galaxy images: {sum(1 for l in self.dataset.labels if l == 0)}")
        
        # Split into train and test
        train_size = int(len(self.dataset.images) * self.train_split)
        test_size = len(self.dataset.images) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset.images, 
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        print(f"\nTrain set size: {len(train_dataset)}")
        print(f"Test set size: {len(test_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, test_loader