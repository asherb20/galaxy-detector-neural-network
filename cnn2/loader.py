import torch
# import torchvision
from torchvision.transforms import v2
# from galaxy_datasets import gz2
from galaxy_datasets.pytorch import GZ2

transform = v2.Compose([
   v2.ToImage(),
   v2.ToDtype(torch.float32, scale=True),
   v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 4

# trainset, labels = gz2(root='cnn2/data/gz2', train=True, download=True)
# print("trainset size:", len(trainset))
# print("trainset:", trainset)
# print("labels:", labels)

trainset = GZ2(root='data/gz2', train=True, download=False, transform=transform)
for batch in trainset:
   image = batch['image']
   label = batch['smooth-or-featured-gz2_smooth']
   print("image shape:", image.shape)
   print("label:", label)

# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# testset = GZ2(root='data/gz2', train=False, download=True, transform=transform)

# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)