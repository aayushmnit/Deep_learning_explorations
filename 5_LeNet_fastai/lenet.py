# Import necessary packages
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LeNet(nn.Module):
    def __init__(self, n_class=10, size=28, in_channels=1):
        super().__init__()
        self.n_class = n_class
        self.size = size
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=20,
            kernel_size=5,
            stride=1,
            padding=2,
        )

        self.conv2 = nn.Conv2d(
            in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=2
        )

        self.fc1 = nn.Linear(
            in_features=int(50 * (np.square(self.size / 4))), out_features=500
        )

        self.fc2 = nn.Linear(in_features=500, out_features=n_class)

    def forward(self, x):
        # Applying first CONV => RELU => Pooling layer
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        # Applying second CONV => RELU => Pooling layer
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # Flatten to feed FC layer
        x = x.view(-1, self.num_flat_features(x))

        # First fully connected layer
        x = F.relu(self.fc1(x))

        # Applying FC layer and return
        return self.fc2(x)

    def num_flat_features(self, x):
        # Returning number of Flatten dimension
        size = x.size()[1:]
        num_features = 1
        # Multiplying all dimension except batch dimension
        for s in size:
            num_features *= s
        return num_features
