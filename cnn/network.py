import torch.nn as nn


def conv_block(in_channels, out_channels):
   """Two 3x3 conv layers (each followed by BatchNorm + ReLU), then halve the spatial size"""
   return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2)
   )


class ConvNeuralNet(nn.Module):
   """Shared CNN backbone with two heads:
   - detect: galaxy vs. non-galaxy (2 logits)
   - morph: galaxy morphology (num_morph_classes logits), only meaningful for galaxies
   """
   def __init__(self, num_morph_classes, in_channels=3):
      super().__init__()
      # 128 -> 64 -> 32 -> 16 -> 8
      self.backbone = nn.Sequential(
         conv_block(in_channels, 32),
         conv_block(32, 64),
         conv_block(64, 128),
         conv_block(128, 256),
         nn.AdaptiveAvgPool2d(1),
         nn.Flatten(),
         nn.Dropout(0.3)
      )
      self.detect_head = nn.Linear(256, 2)
      self.morph_head = nn.Sequential(
         nn.Linear(256, 128),
         nn.ReLU(inplace=True),
         nn.Dropout(0.3),
         nn.Linear(128, num_morph_classes)
      )

   # Progresses data across layers
   def forward(self, x):
      features = self.backbone(x)
      return self.detect_head(features), self.morph_head(features)
