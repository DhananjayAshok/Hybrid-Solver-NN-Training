import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from model import *
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd


input_dim = 3
output_dim = 2


class CIFAR10Dataset():
    def __init__(self):
        pass

    @staticmethod
    def metric():
        return nn.CrossEntropyLoss()

    @staticmethod
    def datasets():
        train_dataset = datasets.CIFAR10('data', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,0.5, 0.5), (0.5, 0.5, 0.5)),
                                           lambda x: x.float(),
                                       ]))

        test_dataset = datasets.CIFAR10('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5, 0.5), (0.5, 0.5, 0.5)),
            lambda x: x.float(),
        ]))
        return train_dataset, test_dataset

    @staticmethod
    def model():
        return CIFAR10ModelDeep()#CIFAR10Model()


class MNISTDataset():
    def __init__(self):
        pass

    @staticmethod
    def metric():
        return nn.CrossEntropyLoss()

    @staticmethod
    def datasets():
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

    @staticmethod
    def model(pretrained=False):
        if pretrained:
            return PreTrainedMNISTModel()
        return MNISTModel()


class AdultsDataset(Dataset):
    def __init__(self, train=True, test_ratio=0.2):
        self.train = train
        X_t, X_v, y_t, y_v = AdultsDataset.get_dataset(test_ratio=test_ratio)
        if self.train:
            self.X = X_t
            self.y = y_t
        else:
            self.X = X_v
            self.y = y_v

    def __getitem__(self, item):
        return self.X[item], int(self.y[item])

    def __len__(self):
        return len(self.y)

    @staticmethod
    def metric():
        return nn.CrossEntropyLoss()

    def loaders(batch_size=32):
        train_dataset = AdultsDataset()
        test_dataset = AdultsDataset()
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, test_loader

    def model(self=None):
        return SimpleClassification(input_dim=13, output_dim=2, w_range=0.0001)

    def get_dataset(test_ratio=0.2):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        df_temp = pd.read_csv(url)
        col_np = np.array(df_temp.columns).reshape(1, 15)
        data = np.append(df_temp.values, col_np, axis=0)
        cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
                "income"]
        df = pd.DataFrame(data=data, columns=cols)
        no_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
        for col in no_cols:
            df[col] = df[col].astype(float)
        cats = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country", "income"]
        for col in cats:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes
        df["education"] = df["education-num"].astype(int)
        df.drop("education-num", axis=1, inplace=True)
        X = df.drop("income", axis=1)
        y = df["income"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)
        X_train = torch.Tensor(X_train.values)
        X_test = torch.Tensor(X_test.values)
        y_train = torch.LongTensor(y_train.values)
        y_test = torch.LongTensor(y_test.values)
        return X_train, X_test, y_train, y_test

    def datasets(test_ratio=0.2):
        train_dataset = AdultsDataset(test_ratio=test_ratio)
        test_dataset = AdultsDataset(train=False, test_ratio=test_ratio)
        return train_dataset, test_dataset


class IdentityDataset(Dataset):
    def __init__(self, n, input_dim=input_dim, output_dim=output_dim):
        self.n = n
        self.X = torch.rand(n, input_dim)
        self.y = self.X[:, :output_dim]

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.n

    @staticmethod
    def metric():
        return nn.MSELoss()

    def datasets(n=1000, input_dim=input_dim, output_dim=output_dim, batch_size=32):
        train_dataset = IdentityDataset(n, input_dim, output_dim)
        test_dataset = IdentityDataset(n, input_dim, output_dim)
        return train_dataset, test_dataset

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

    @staticmethod
    def metric():
        return nn.MSELoss()

    def datasets(n=1000, input_dim=input_dim, output_dim=output_dim, batch_size=32):
        train_dataset = AffineDataset(n, input_dim, output_dim)
        test_dataset = AffineDataset(n, input_dim, output_dim)
        return train_dataset, test_dataset

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

    @staticmethod
    def metric():
        return nn.MSELoss()

    def datasets(n=1000, input_dim=input_dim, output_dim=output_dim, batch_size=32):
        train_dataset = PolynomialDataset(n, input_dim, output_dim)
        test_dataset = PolynomialDataset(n, input_dim, output_dim)
        return train_dataset, test_dataset

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

    @staticmethod
    def metric():
        return nn.MSELoss()

    def datasets(n=1000, input_dim=input_dim, output_dim=output_dim, batch_size=100):
        train_dataset = FormulaDataset(n, input_dim, output_dim)
        test_dataset = FormulaDataset(n, input_dim, output_dim)
        return train_dataset, test_dataset

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

    @staticmethod
    def metric():
        return nn.CrossEntropyLoss()

    def datasets(n=1000, input_dim=input_dim, batch_size=100):
        train_dataset = ThresholdDataset(n, input_dim)
        test_dataset = ThresholdDataset(n, input_dim)
        return train_dataset, test_dataset

    def model(input_dim=input_dim, output_dim=2):
        return SimpleClassification(input_dim, output_dim, w_range=10)