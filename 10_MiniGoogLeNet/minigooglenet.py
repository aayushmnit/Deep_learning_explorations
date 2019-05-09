# Import necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MiniGoogLeNet(nn.Module):
    """
    A small and light weight model inspired by Inception architecture
    """

    def __init__(self, n_class, size, depth):
        super(MiniGoogLeNet, self).__init__()
        self.n_class = n_class
        self.size = size
        self.depth = depth
        self.conv0 = BasicConv2d(
            in_channels=depth, out_channels=96, kernel_size=3, stride=1, padding=1
        )

        # Inceptionx2 => Downsample
        self.inception1a = InceptionModule(in_channels=96, opK1x1=32, opK3x3=32)
        self.inception1b = InceptionModule(in_channels=64, opK1x1=32, opK3x3=48)
        self.downsample1a = DownsampleModule(in_channels=80, k=80)

        # Inceptionx4 => Downsample
        self.inception2a = InceptionModule(in_channels=160, opK1x1=112, opK3x3=48)
        self.inception2b = InceptionModule(in_channels=160, opK1x1=96, opK3x3=64)
        self.inception2c = InceptionModule(in_channels=160, opK1x1=80, opK3x3=80)
        self.inception2d = InceptionModule(in_channels=160, opK1x1=48, opK3x3=96)
        self.downsample2a = DownsampleModule(in_channels=144, k=96)

        # Inceptionx2 => Average Pool
        self.inception3a = InceptionModule(in_channels=240, opK1x1=176, opK3x3=160)
        self.inception3b = InceptionModule(in_channels=336, opK1x1=176, opK3x3=160)

        # Final FC layer
        self.fc4a = nn.Linear(in_features=336, out_features=n_class)

    def forward(self, x):
        # If x is 3 x 28 x 28
        x = self.conv0(x)  # 96 x 28 x 28

        # Inceptionx2 => Downsample
        x = self.inception1a(x)  # 64 x 28 x 28
        x = self.inception1b(x)  # 80 x 28 x 28
        x = self.downsample1a(x)  # 160 x 13 x 13

        # Inceptionx4 => Downsample
        x = self.inception2a(x)  # 160 x 13 x 13
        x = self.inception2b(x)  # 160 x 13 x 13
        x = self.inception2c(x)  # 160 x 13 x 13
        x = self.inception2d(x)  # 144 x 13 x 13
        x = self.downsample2a(x)  # 240 x 6 x 6

        # Inceptionx2 => Average Pool 
        x = self.inception3a(x)  # 336 x 6 x 6
        x = self.inception3b(x)  # 336 x 6 x 6
        x = F.avg_pool2d(x, kernel_size=7)  # 366 x 1 x 1

        # Dropout => Flatten => Dense(Fully-Connected)
        x = F.dropout(x, p=0.5)  # 366 x 1 x 1
        x = x.view(x.size(0), -1)  # 366
        x = self.fc4a(x) # 10

        return x


class InceptionModule(nn.Module):
    """
    Class to apply inception by merging a conv1x1 and conv3x3
    """

    def __init__(self, in_channels, opK1x1, opK3x3):
        super(InceptionModule, self).__init__()

        ## Defining 1x1 & 3x3 convolution
        self.conv0 = BasicConv2d(in_channels, opK1x1, kernel_size=1)
        self.conv1 = BasicConv2d(in_channels, opK3x3, kernel_size=3, padding=1)

    def forward(self, x):
        # Applying 1x1 & 3x3 convolution
        conv_1x1 = self.conv0(x)
        conv_3x3 = self.conv1(x)

        # Concatenating output
        outputs = [conv_1x1, conv_3x3]

        return torch.cat(outputs, 1)


class DownsampleModule(nn.Module):
    """
    Class to apply downsampling by merging a conv and max pooling layer
    """

    def __init__(self, in_channels, k):
        super(DownsampleModule, self).__init__()

        # Defining convolution layer
        self.conv0 = BasicConv2d(in_channels, k, kernel_size=3, stride=2)

    def forward(self, x):

        # Convolution layer
        conv_3x3 = self.conv0(x)

        # Pooling layer
        pool = F.max_pool2d(x, kernel_size=3, stride=2)

        # Concatenation
        outputs = [conv_3x3, pool]

        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):
    """
    Class to apply basic Conv => BN => RELU sequence
    """

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()

        # Define a CONV => BN => RELU
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        # Applying a CONV => BN => RELU
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

