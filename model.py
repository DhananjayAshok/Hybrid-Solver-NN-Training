from torch import argmax
from binary_modules import *


class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv = BinarizeConv2d(1, 3, 4)
        self.fc = BinarizeLinear(1875, 10)

    def forward(self, x):
        x = self.conv(x)
        h = x.view(x.shape[0], -1)
        y = self.fc(h)
        y = Binarize(y)
        return y

    def predict(self, x):
        logits = self.forward(x)
        return argmax(logits, axis=1)

