import torch

from model import MNISTModel, PreTrainedMNISTModel

import torch.nn as nn
from torchvision import datasets, transforms


def get_datasets(train_batch_size=32):
    train_dataset = datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           lambda x: x.float(),
                ]))


    test_dataset = datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           lambda x: x.float(),
                       ]))
    return train_dataset, test_dataset


def model(pretrained=False):
    if pretrained:
        return PreTrainedMNISTModel()
    return MNISTModel()


def metric():
    return nn.CrossEntropyLoss()
