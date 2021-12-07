import torch

from model import SimpleClassification

import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


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

    def metric():
        return nn.CrossEntropyLoss()

    def loaders(batch_size=32):
        train_dataset = AdultsDataset()
        test_dataset = AdultsDataset()
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, test_loader

    def model(self=None):
        return SimpleClassification(input_dim=13, output_dim=2, w_range=0.001)

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

    def get_datasets(test_ratio=0.2):
        train_dataset = AdultsDataset(test_ratio=test_ratio)
        test_dataset = AdultsDataset(train=False, test_ratio=test_ratio)
        return train_dataset, test_dataset


