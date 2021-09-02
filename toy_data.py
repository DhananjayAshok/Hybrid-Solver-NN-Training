import torch
import torch.nn as nn
from model import SimpleRegression, SingleLayerRegression, SimpleClassification
from torch.utils.data import Dataset, DataLoader


input_dim = 3
output_dim = 2


class IdentityDataset(Dataset):
    def __init__(self, n, input_dim=input_dim, output_dim=output_dim):
        self.n = n
        self.X = torch.rand(n, input_dim)
        self.y = self.X[:, :output_dim]

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.n

    def metric():
        return nn.MSELoss()

    def loaders(n=1000, input_dim=input_dim, output_dim=output_dim, batch_size=32):
        train_dataset = IdentityDataset(n, input_dim, output_dim)
        test_dataset = IdentityDataset(n, input_dim, output_dim)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, test_loader

    def model(input_dim=input_dim, output_dim=output_dim):
        return SimpleRegression(input_dim, output_dim, w_range=0.1)


class AffineDataset(Dataset):
    def __init__(self, n, input_dim, output_dim):
        self.n = n
        self.X = torch.rand(n, input_dim)
        self.y = 3*self.X[:, :output_dim] + 4

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.n

    def metric():
        return nn.MSELoss()

    def loaders(n=1000, input_dim=input_dim, output_dim=output_dim, batch_size=32):
        train_dataset = AffineDataset(n, input_dim, output_dim)
        test_dataset = AffineDataset(n, input_dim, output_dim)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, test_loader

    def model(input_dim=input_dim, output_dim=output_dim):
        return SimpleRegression(input_dim, output_dim, w_range=1000)


class PolynomialDataset(Dataset):
    def __init__(self, n, input_dim, output_dim):
        self.n = n
        self.X = torch.rand(n, input_dim)
        self.y = 3*self.X[:, :output_dim]**4 + 6*self.X[:, :output_dim]**3 + 2*self.X[:, :output_dim]**2 + \
                 self.X[:, :output_dim] + 9

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.n

    def metric():
        return nn.MSELoss()

    def loaders(n=1000, input_dim=input_dim, output_dim=output_dim, batch_size=32):
        train_dataset = PolynomialDataset(n, input_dim, output_dim)
        test_dataset = PolynomialDataset(n, input_dim, output_dim)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, test_loader

    def model(input_dim=input_dim, output_dim=output_dim):
        return SimpleRegression(input_dim, output_dim, w_range=10)


class FormulaDataset(Dataset):
    def __init__(self, n, input_dim, output_dim):
        self.n = n
        self.X = torch.rand(n, input_dim)
        self.y = 2*torch.exp(self.X[:, :output_dim]) + 3 * torch.sin(self.X[:, :output_dim]) * self.X[:, :output_dim]**5


    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.n

    def metric():
        return nn.MSELoss()

    def loaders(n=1000, input_dim=input_dim, output_dim=output_dim, batch_size=100):
        train_dataset = FormulaDataset(n, input_dim, output_dim)
        test_dataset = FormulaDataset(n, input_dim, output_dim)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, test_loader

    def model(input_dim=input_dim, output_dim=output_dim):
        return SimpleRegression(input_dim, output_dim, w_range=100)


class ThresholdDataset(Dataset):
    def __init__(self, n, input_dim):
        self.n = n
        self.X = torch.rand(n, input_dim)
        self.X = self.X - 0
        intermediate = self.X - torch.mean(self.X)
        self.y = torch.rand(n)
        for i in range(self.X.shape[0]):
            row_sum = sum(intermediate[i])
            if row_sum <= 0:
                self.y[i] = 0
            else:
                self.y[i] = 1
        self.y = self.y.type(torch.LongTensor)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.n

    def metric():
        return nn.CrossEntropyLoss()

    def loaders(n=1000, input_dim=input_dim, batch_size=100):
        train_dataset = ThresholdDataset(n, input_dim)
        test_dataset = ThresholdDataset(n, input_dim)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, test_loader

    def model(input_dim=input_dim, output_dim=2):
        return SimpleClassification(input_dim, output_dim, w_range=10)