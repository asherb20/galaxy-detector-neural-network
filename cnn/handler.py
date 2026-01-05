import torch
import torch.nn as nn
from torchvision import transforms, datasets
from galaxy_datasets.pytorch import GZ2
from wrappers import GalaxyWrapper, NonGalaxyWrapper
from network import ConvNeuralNet
import time

# define relevant variables for the ML task
batch_size = 128
num_classes = 2
learning_rate = 0.001
num_epochs = 20
num_workers = 0

# device will determine whether to run the training on GPU or CPU
gpu_avail = torch.cuda.is_available()
device = torch.device('cuda' if gpu_avail else 'cpu')
if gpu_avail:
    print(f"using gpu: {torch.cuda.get_device_name(0)}")

# main training loop
if __name__ == "__main__":
   # define image transformations
    all_transforms = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
      transforms.ToTensor(),  # Converts to tensor with values in [0, 1]
      transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    # initialize Galaxy Zoo 2 dataset (galaxy images)
    gz2_dataset = GZ2(
      root='data/gz2',
      train=True,
      transform=all_transforms,
      download=False
    )

    # for quicker testing, use a subset of the dataset
    subset_indices = torch.randperm(len(gz2_dataset))[:1000].tolist()
    gz2_dataset = torch.utils.data.Subset(gz2_dataset, subset_indices)

    # initialize FakeData for non-galaxy images (random noise)
    fake_dataset = datasets.FakeData(
      size=len(gz2_dataset),  # Match GZ2 size for balance
      image_size=(1, 128, 128),  # Grayscale, 128x128
      num_classes=1,
      transform=all_transforms
    )

    # wrapped datasets
    galaxy_wrapped = GalaxyWrapper(gz2_dataset)
    non_galaxy_wrapped = NonGalaxyWrapper(fake_dataset)

    # combine datasets
    combined_dataset = torch.utils.data.ConcatDataset([galaxy_wrapped, non_galaxy_wrapped])
    
    # create data loaders
    train_loader = torch.utils.data.DataLoader(
      dataset=combined_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=(device.type == 'cuda')
    )

    # sample batch
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels: {labels}")
    print(f"Unique labels: {torch.unique(labels).tolist()}")  # Should be [0, 1]

    # initialize the CNN model
    model = ConvNeuralNet(num_classes)
    model.to(device)

    # set loss function with criterion
    criterion = nn.CrossEntropyLoss()

    # set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

    total_step = len(train_loader)
    print(f"\ntotal training steps per epoch: {total_step}")

    # pre-defined number of epochs to determine how many iterations to train the network on
    for epoch in range(num_epochs):
      epoch_start = time.time()
      # load_time = 0
      compute_time = 0

      # load in the data in batches using the train_loader object
      for i, (images, labels) in enumerate(train_loader):
        compute_batch_start = time.time()

        # move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        compute_time += time.time() - compute_batch_start

      epoch_time = time.time() - epoch_start
      print(f'epoch [{epoch + 1}/{num_epochs}], loss: {loss.item():.4f}, time: {epoch_time:.2f}s, compute: {compute_time:.2f}s')
    
    # evaluation
    model.eval()
    with torch.no_grad():
      correct = 0
      total = 0
      for images, labels in train_loader:
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
      
      print(f'accuracy: {100 * correct / total:.2f}%')
 
      