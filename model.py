from torch import argmax
import torch.nn as nn
from gurobi_modules import NamedLinear, NamedConv2d, MILPNet
from torchvision.models import resnet50


class CIFAR10Model(nn.Module):
    def __init__(self, internal_dim=2000):
        nn.Module.__init__(self)
        self.conv = NamedConv2d(3, 6, 4)
        self.dense = nn.Linear(5046, internal_dim)
        self.milp_model = MILPNet(nn.Sequential(NamedLinear(internal_dim, 10)), w_range=0.1)

    def forward(self, x):
        h = self.forward_till_dense(x)
        y = self.milp_model(h)
        out = y
        return out

    def forward_till_dense(self, x):
        x = self.conv(x)
        r = nn.functional.relu(x)
        h = r.view(x.shape[0], -1)
        o = self.dense(h)
        o = nn.functional.relu(o)
        return o

    def predict(self, x):
        logits = self.forward(x)
        predictions = argmax(logits, dim=1)
        return predictions


class CIFAR10ModelDeep(nn.Module):
    def __init__(self, internal_dim=500):
        nn.Module.__init__(self)
        self.conv = NamedConv2d(3, 16, 1, padding=1)
        self.conv1 = NamedConv2d(16, 32, 3, 1, padding=1)
        self.conv2 = NamedConv2d(32, 64, 3, 1, padding=1)
        self.dense = nn.Linear(4*4*64, internal_dim)
        self.milp_model = MILPNet(nn.Sequential(NamedLinear(internal_dim, 10)), w_range=0.1)

    def forward(self, x):
        h = self.forward_till_dense(x)
        out = self.milp_model(h)
        return out

    def forward_till_dense(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2, 2)
        h = x.view(x.shape[0], -1)
        h = self.dense(h)
        o = nn.functional.relu(h)
        return o

    def predict(self, x):
        logits = self.forward(x)
        predictions = argmax(logits, dim=1)
        return predictions


class MNISTModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv = NamedConv2d(1, 3, 4)
        self.milp_model = MILPNet(nn.Sequential(NamedLinear(1875, 10)), w_range=0.1)

    def forward(self, x):
        h = self.forward_till_dense(x)
        y = self.milp_model(h)
        out = y
        return out

    def forward_till_dense(self, x):
        x = self.conv(x)
        r = nn.functional.relu(x)
        h = r.view(x.shape[0], -1)
        return h

    def predict(self, x):
        logits = self.forward(x)
        predictions = argmax(logits, dim=1)
        return predictions


class PreTrainedMNISTModel(nn.Module):
    def __init__(self):
        nn.Module__init__(self)
        # Load a pretrained resnet model from torchvision.models in Pytorch
        self.model = resnet50(pretrained=True)

        # Change the input layer to take Grayscale image, instead of RGB images.
        # Hence in_channels is set as 1 or 3 respectively
        # original definition of the first layer on the ResNet class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model = nn.Sequential(*list(self.model.children()[:-1]))

        # Change the output layer to output 10 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.milp_model = MILPNet(nn.Sequential(NamedLinear(num_ftrs, 10)), w_range=0.01)

    def forward(self, x):
        return self.milp_model(self.model(x))

    def forward_till_dense(self, x):
        return self.model(x)

    def predict(self, x):
        logits = self.forward(x)
        predictions = argmax(logits, dim=1)
        return predictions


class SimpleRegression(nn.Module):
    def __init__(self, input_dim, output_dim, w_range=0.1):
        nn.Module.__init__(self)
        assert output_dim < input_dim
        self.layer_1 = nn.Linear(input_dim, (input_dim + output_dim)//2)
        self.milp_model = MILPNet(nn.Sequential(NamedLinear((input_dim + output_dim)//2, output_dim) ),
                                  classification=False, w_range=w_range)

    def forward(self, x):
        h = self.forward_till_dense(x)
        y = self.milp_model(h)
        out = y
        return out

    def forward_till_dense(self, x):
        x = self.layer_1(x)
        r = nn.functional.relu(x)
        return r

    def predict(self, x):
        return self.forward(x)


class SingleLayerRegression(nn.Module):
    def __init__(self, input_dim, output_dim, w_range=0.1):
        nn.Module.__init__(self)
        self.milp_model = MILPNet(nn.Sequential(NamedLinear(input_dim, output_dim)) ,
                                  classification=False, w_range=w_range)
    def forward(self, x):
        y = self.milp_model(x)
        out = y
        return out

    def forward_till_dense(self, x):
        return x

    def predict(self, x):
        return self.forward(x)


class SimpleClassification(nn.Module):
    def __init__(self, input_dim, output_dim, w_range=0.1):
        nn.Module.__init__(self)
        intermediate = (input_dim + output_dim)//2
        self.layer_1 = nn.Linear(input_dim, intermediate)
        self.milp_model = MILPNet(nn.Sequential(NamedLinear(intermediate, output_dim)),
                                  classification=True, w_range=w_range)

    def forward(self, x):
        h = self.forward_till_dense(x)
        y = self.milp_model(h)
        out = y
        return out

    def forward_till_dense(self, x):
        x = self.layer_1(x)
        r = nn.functional.relu(x)
        return r

    def predict(self, x):
        logits = self.forward(x)
        return argmax(logits, dim=1)

