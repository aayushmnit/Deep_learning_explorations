# Import necessary packages
import torch.nn as nn
import numpy as np


class AlexNet(nn.Module):
    def __init__(self, features, n_class=1000):
        super(AlexNet, self).__init__()
        self.features = features
        # (FC=>ACT=>BN=>DO)x2=>FC=>SOFTMAX
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_class)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def make_layers():
    layers = []
    in_channels = 3
    
    ## CONV=>ACT=>BN=>POOL=>DO
    layers += [nn.Conv2d(in_channels, out_channels = 96, kernel_size = 11, stride = 4, padding = 2)]
    layers += [nn.ReLU()]
    layers += [nn.BatchNorm2d(96)]
    layers += [nn.MaxPool2d(kernel_size = 3, stride = 2)]
    layers += [nn.Dropout(p = 0.25)]

    ## CONV=>ACT=>BN=>POOL=>DO
    layers += [nn.Conv2d(96, out_channels = 256, kernel_size = 5, stride = 1, padding = 2)]
    layers += [nn.ReLU()]
    layers += [nn.BatchNorm2d(256)]
    layers += [nn.MaxPool2d(kernel_size = 3, stride = 2)]
    layers += [nn.Dropout(p = 0.25)]
        
    ## ((CONV=>ACT=>BN)x3
    in_channels = 256
    for op in [384, 384, 256]:
        layers += [nn.Conv2d(in_channels, op, kernel_size = 3, padding = 1)]
        layers += [nn.ReLU()]
        layers += [nn.BatchNorm2d(op)]
        in_channels = op
    
    ## POOL=>DO
    layers += [nn.MaxPool2d(kernel_size = 3, stride = 2)]
    layers += [nn.Dropout(p=0.25)]
    return nn.Sequential(*layers)

def ALEXNet(**kwargs):
    model = AlexNet(make_layers(), **kwargs)
    return model

