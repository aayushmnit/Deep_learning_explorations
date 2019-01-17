# Import necessary packages
import torch.nn as nn
import numpy as np


class VGG(nn.Module):
    def __init__(self, features, size, n_class=10):
        super(VGG, self).__init__()
        self.features = features
        # FC=>ACT=>DO=>FC=>SOFTMAX
        self.classifier = nn.Sequential(
            nn.Linear(int(64 * np.square(size / 4)), 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, n_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(batch_norm=False):
    layers = []
    in_channels = 3
    ## ((CONV=>ACT=>BN)x2 =>POOL=>DO)x2
    for op in [32, 64]:
        ## For (CONV=>ACT=>BN)x2
        for _ in range(2):
            layers += [nn.Conv2d(in_channels, op, kernel_size=3, padding=1)]
            layers += [nn.ReLU(True)]
            if batch_norm:
                layers += [nn.BatchNorm2d(op)]
            in_channels = op
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Dropout(0.25)]
    return nn.Sequential(*layers)


def MiniVGGNet(batch_norm=False, **kwargs):
    model = VGG(make_layers(batch_norm=batch_norm), **kwargs)
    return model

