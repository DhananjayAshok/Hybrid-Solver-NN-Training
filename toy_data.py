import torch
import torch.nn as nn
from model import SimpleRegression
from torch.utils.data import Dataset, DataLoader

class IdentityDataset(Dataset):
    def __init__(self, n, input_dim, output_dim):
        self.n = n
        self.X = torch.rand(n, input_dim)
        self.y = self.X[:, :output_dim]

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.n

    def metric():
        return nn.MSELoss()

    def loaders(n=1000, input_dim=10, output_dim=5, batch_size=32):
        train_dataset = IdentityDataset(n, input_dim, output_dim)
        test_dataset = IdentityDataset(n, input_dim, output_dim)
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, test_loader

    def model(input_dim=10, output_dim=5):
        return SimpleRegression(input_dim, output_dim, w_range=1)
