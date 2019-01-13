# Import necessary packages
import torch.nn as nn
import torch.nn.functional as F


class ShallowNet(nn.Module):
    def __init__(self, n_class=2, size=32, in_channels=3):
        super().__init__()
        self.n_class = n_class
        self.size = size
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.fc1 = nn.Linear(in_features=32*self.size*self.size,
                             out_features=self.n_class)

    def forward(self, x):
        # Applying first CONV => RELU layer
        x = F.relu(self.conv1(x))

        # Flatten to feed FC layer
        x = x.view(-1, self.num_flat_features(x))

        # Applying FC layer and return
        return self.fc1(x)

    def num_flat_features(self, x):
        # Returning number of Flatten dimension
        size = x.size()[1:]
        num_features = 1
        # Multiplying all dimension except batch dimension
        for s in size:
            num_features *= s
        return num_features
