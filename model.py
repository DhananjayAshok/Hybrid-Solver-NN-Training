from torch import argmax
import torch.nn as nn
from gurobi_modules import NamedLinear, NamedConv2d, MILPNet


class MNISTModel(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv = NamedConv2d(1, 3, 4)
        self.milp_model = MILPNet(nn.Sequential(NamedLinear(1875, 10)), w_range=0.01)

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
        self.layer_1 = nn.Linear(input_dim, (input_dim + output_dim)//2)
        self.milp_model = MILPNet(nn.Sequential(NamedLinear((input_dim + output_dim)//2, output_dim)),
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
        return self.forward(x)