from torch import argmax
import torch.nn as nn
from gurobi_modules import NamedLinear, NamedConv2d


class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv = NamedConv2d(1, 3, 4)
        self.fc = NamedLinear(1875, 10)

    def forward(self, x):
        h = self.forward_till_dense(x)
        y = self.fc(h)
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