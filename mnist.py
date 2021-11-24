import torch

from model import MNISTModel, PreTrainedMNISTModel

import torch.nn as nn
from torchvision import datasets, transforms


def loaders(train_batch_size=32):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           lambda x: x.float(),
                ])),
        batch_size=train_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           lambda x: x.float(),
                       ])),
        batch_size=32, shuffle=True)
    return train_loader, test_loader


def model(pretrained=False):
    if pretrained:
        return PreTrainedMNISTModel()
    return MNISTModel()


def metric():
    return nn.CrossEntropyLoss()
