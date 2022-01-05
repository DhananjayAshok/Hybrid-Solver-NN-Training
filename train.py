import pandas as pd
import torch
import os
from data import *
from algorithms import GradientDescent, SolverFineTuning, SolverGDHybrid
import argparse


def get_datasets(key):
    if key == "mnist":
        return MNISTDataset.datasets()
    elif key == "identity":
        return IdentityDataset.datasets()
    elif key == "affine":
        return AffineDataset.datasets()
    elif key == "polynomial":
        return PolynomialDataset.datasets()
    elif key == "formula":
        return FormulaDataset.datasets()
    elif key == "threshold":
        return ThresholdDataset.datasets()
    elif key == "adults":
        return AdultsDataset.datasets()
    elif key == "cifar10":
        return CIFAR10Dataset.datasets()


def get_model(key):
    if key == "mnist":
        return MNISTDataset.model()
    elif key == "identity":
        return IdentityDataset.model()
    elif key == "affine":
        return AffineDataset.model()
    elif key == "polynomial":
        return PolynomialDataset.model()
    elif key == "formula":
        return FormulaDataset.model()
    elif key == "threshold":
        return ThresholdDataset.model()
    elif key == "adults":
        return AdultsDataset.model()
    elif key == "cifar10":
        return CIFAR10Dataset.model()


def get_metric(key):
    if key == "mnist":
        return MNISTDataset.metric()
    elif key == "identity":
        return IdentityDataset.metric()
    elif key == "affine":
        return AffineDataset.metric()
    elif key == "polynomial":
        return PolynomialDataset.metric()
    elif key == "formula":
        return FormulaDataset.metric()
    elif key == "threshold":
        return ThresholdDataset.metric()
    elif key == "adults":
        return AdultsDataset.metric()
    elif key == "cifar10":
        return CIFAR10Dataset.metric()


def train_gd(batch_size=32, epochs=2, lr=0.05, early_stopping=None, max_points=None, optimizer=torch.optim.SGD,
             lr_scheduling=False):
    model = get_model(key)
    trainer = GradientDescent(model, metric, train_dataset, test_dataset, batch_size)
    trainer.configure(epochs=epochs, lr=lr, early_stopping=early_stopping, max_points=max_points,
                      optimizer=optimizer, lr_scheduling=lr_scheduling)
    trainer.train()
    print("Evaluating GD Accuracy Overall")
    return trainer.evaluate_accuracy()


def train_hybrid(batch_size=32, epoch_sequence=[20, 20], lr_sequence=[0.1, 0.05], early_stopping=2, max_points=None,
                 incorrect_subset=True):
    model = get_model(key)
    trainer = SolverGDHybrid(model, metric, train_dataset, test_dataset, batch_size)
    trainer.configure(epochs_sequence=epoch_sequence, lr_sequence=lr_sequence,
                      early_stopping=early_stopping, max_points=max_points, incorrect_subset=incorrect_subset)
    trainer.train()
    print("Evaluating Hybrid Accuracy Overall - Last GD Result, Final Result")
    return trainer.last_gd_res, trainer.evaluate_accuracy()


def test_gd_vs_hybrid():
    d = []
    columns = ["method", "epochs", "max_points", "accuracy"]
    lr = 0.01
    early_stopping = 2
    epochs = [20]
    max_points = [5_000, 10_000, 15_000, 20_000, 25_000]
    lr_sequence = [lr, lr/2]

    for epoch in epochs:
        for max_point in max_points:
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Starting epoch {epoch} with max points {max_point}/"
                  f"{len(train_dataset)} XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            epoch_sequence = [epoch // 2, epoch // 2]
            sgd_acc = train_gd(max_points=max_point, epochs=epoch, early_stopping=early_stopping)
            sgd_lrs_acc = train_gd(max_points=max_point, epochs=epoch, early_stopping=early_stopping, lr_scheduling=True)
            adam_acc = train_gd(max_points=max_point, epochs=epoch, early_stopping=early_stopping, optimizer=torch.optim.Adam)
            last_gd, hybrid_acc = train_hybrid(max_points=max_point, epoch_sequence=epoch_sequence, early_stopping=early_stopping)
            best_accuracy = max([sgd_acc, sgd_lrs_acc, adam_acc])
            d.append(["SGD", epoch, max_point, float(sgd_acc)])
            d.append(["SGD LRS", epoch, max_point, float(sgd_lrs_acc)])
            d.append(["Adam", epoch, max_point, float(adam_acc)])
            d.append(["hybrid last gd", epoch, max_point, float(last_gd)])
            d.append(["hybrid", epoch, max_point, float(hybrid_acc)])
            print(f"Intermediate Success: {last_gd > best_accuracy}")
            print(f"Overall Success: {hybrid_acc > best_accuracy}")
            print(f"Final Solver Failure: {last_gd > hybrid_acc}")
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    df = pd.DataFrame(data=d, columns=columns)
    print(df)
    df.to_csv(os.path.join(logfolder, f"{key}_gd_vs_hybrid.csv"), index=False)
    return df


def test_gd():
    d = []
    columns = ["optimizer", "lr_scheduling", "epochs", "max_points", "lr", "early_stopping", "accuracy"]
    lrs = [0.1, 0.01]
    early_stoppings = [2, 5]
    epochs = [20, 30, 40]
    max_points = [5_000, 10_000, 15_000, 20_000, 25_000]
    optimizers = [("SGD", torch.optim.SGD), ("Adam", torch.optim.Adam)]
    lr_scheduling = [True, False]

    for epoch in epochs:
        for max_point in max_points:
            for lr in lrs:
                for early_stopping in early_stoppings:
                    for opt in optimizers:
                        for lr_schedule in lr_scheduling:
                            name, optimizer = opt
                            print(
                                f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Starting epoch {epoch} with max points {max_point}/{len(train_dataset)} "
                                f"XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                            gd_acc = train_gd(max_points=max_point, epochs=epoch, early_stopping=early_stopping, lr=lr, optimizer=optimizer, lr_scheduling=lr_schedule)
                            d.append([name, lr_schedule, epoch, max_point, lr, early_stopping, float(gd_acc)])
    df = pd.DataFrame(data=d, columns=columns)
    print(df)
    df.to_csv(os.path.join(logfolder, f"{key}_gdTest.csv"), index=False)
    return df


logfolder = "logs"
key = "adults"
parser = argparse.ArgumentParser()
parser.add_argument('--key', metavar='K', type=str, help='key to run tests')
args = parser.parse_args()
if args.key is not None:
    key = args.key
train_dataset, test_dataset = get_datasets(key)
metric = get_metric(key)

if __name__ == "__main__":
    test_gd_vs_hybrid()