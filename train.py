import pandas as pd
import torch
import os
from data import *
from algorithms import GradientDescent, SolverGDHybridManual, SolverGDHybrid
import argparse
from time import time

regression_keys = ["identity", "affine", "polynomial", "formula"]
classification_keys = ["mnist", "adults", "cifar10"]
all_keys = regression_keys + classification_keys


def get_classification(key):
    if key in classification_keys:
        return True
    elif key in regression_keys:
        return False
    else:
        raise ValueError


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
    start_time = time()
    trainer.train()
    time_taken = time() - start_time
    print("Evaluating GD Accuracy Overall")
    if classification:
        return trainer.evaluate_accuracy(), time_taken
    else:
        return trainer.evaluate_l1()[0], time_taken


def train_hybrid(batch_size=32, epoch_sequence=[20, 20], lr_sequence=[0.1, 0.05], early_stopping=2, max_points=None,
                 incorrect_subset=True):
    model = get_model(key)
    trainer = SolverGDHybrid(model, metric, train_dataset, test_dataset, batch_size)
    trainer.configure(epochs_sequence=epoch_sequence, lr_sequence=lr_sequence,
                      early_stopping=early_stopping, max_points=max_points, incorrect_subset=incorrect_subset,
                      classification=classification)
    start_time = time()
    trainer.train()
    time_taken = time() - start_time
    print("Evaluating Hybrid Accuracy Overall - Last GD Result, Final Result")
    if classification:
        return trainer.last_gd_res, trainer.evaluate_accuracy(), time_taken
    else:
        return trainer.last_gd_res[0], trainer.evaluate_l1()[0], time_taken


def train_hybrid_manual(batch_size=32, epoch_sequence=[20, "s"], lr_sequence=[0.1, None], max_points=None,
                 incorrect_subset=True):
    model = get_model(key)
    trainer = SolverGDHybridManual(model, metric, train_dataset, test_dataset, batch_size)
    trainer.configure(epochs_sequence=epoch_sequence, lr_sequence=lr_sequence,
                      max_points=max_points, incorrect_subset=incorrect_subset,
                      classification=classification)
    start_time = time()
    trainer.train()
    time_taken = time() - start_time
    print("Evaluating Hybrid Accuracy Overall - Last GD Result, Final Result")
    if classification:
        return trainer.evaluate_accuracy(), time_taken
    else:
        return trainer.evaluate_l1()[0], time_taken


def test_gd_vs_hybrid_metric():
    d = []
    columns = ["method", "epochs", "max_points", "metric"]
    lr = 0.01
    early_stopping = 2
    epochs = [20]
    max_points = [5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 45_000, 50_000, 55_000, 60_000]
    lr_sequence = [lr, lr/2]

    for epoch in epochs:
        for max_point in max_points:
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Starting epoch {epoch} with max points {max_point}/"
                  f"{len(train_dataset)} XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            epoch_sequence = [epoch // 2, epoch // 2]
            sgd_metric, sgd_time = train_gd(max_points=max_point, epochs=epoch, early_stopping=early_stopping)
            sgd_lrs_metric, sgd_lrs_time = train_gd(max_points=max_point, epochs=epoch, early_stopping=early_stopping, lr_scheduling=True)
            adam_metric, adam_time = train_gd(max_points=max_point, epochs=epoch, early_stopping=early_stopping, optimizer=torch.optim.Adam)
            last_gd, hybrid_metric, hybrid_time = train_hybrid(max_points=max_point, epoch_sequence=epoch_sequence, early_stopping=early_stopping)
            best_metric = max([sgd_metric, sgd_lrs_metric, adam_metric])
            d.append(["SGD", epoch, max_point, float(sgd_metric)])
            d.append(["SGD LRS", epoch, max_point, float(sgd_lrs_metric)])
            d.append(["Adam", epoch, max_point, float(adam_metric)])
            d.append(["hybrid last gd", epoch, max_point, float(last_gd)])
            d.append(["hybrid", epoch, max_point, float(hybrid_metric)])
            print(f"Intermediate Success: {last_gd > best_metric}")
            print(f"Overall Success: {hybrid_metric > best_metric}")
            print(f"Final Solver Failure: {last_gd > hybrid_metric}")
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    df = pd.DataFrame(data=d, columns=columns)
    print(df)
    df.to_csv(os.path.join(logfolder, f"{key}_gd_vs_hybrid_metric.csv"), index=False)
    return df


def test_gd_vs_hybrid_cost():
    d = []
    columns = ["method", "epochs", "lr", "metric", "time"]
    lrs = [0.01, 0.1]
    early_stopping = None
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    max_point = 1000
    for epoch in epochs:
        for lr in lrs:
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Starting epoch {epoch} with max points {max_point}/"
                  f"{len(train_dataset)} XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            epoch_sequence = [epoch, "s"]
            lr_sequence = [lr, None]
            sgd_metric, sgd_time = train_gd(max_points=max_point, epochs=epoch, early_stopping=early_stopping)
            sgd_lrs_metric, sgd_lrs_time = train_gd(max_points=max_point, epochs=epoch, early_stopping=early_stopping, lr_scheduling=True)
            adam_metric, adam_time = train_gd(max_points=max_point, epochs=epoch, early_stopping=early_stopping, optimizer=torch.optim.Adam)
            hybrid_metric, hybrid_time = train_hybrid_manual(max_points=max_point,
                                                  epoch_sequence=epoch_sequence,
                                                  lr_sequence=lr_sequence)
            d.append(["SGD", epoch, lr, float(sgd_metric), sgd_time])
            d.append(["SGD LRS", epoch, lr, float(sgd_lrs_metric), sgd_lrs_time])
            d.append(["Adam", epoch, lr, float(adam_metric), adam_time])
            d.append(["hybrid", epoch, lr, float(hybrid_metric), hybrid_time])
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    df = pd.DataFrame(data=d, columns=columns)
    print(df)
    df.to_csv(os.path.join(logfolder, f"{key}_gd_vs_hybrid_cost.csv"), index=False)
    return df


def test_gd():
    d = []
    columns = ["optimizer", "lr_scheduling", "epochs", "max_points", "lr", "early_stopping", "metric"]
    lrs = [0.1, 0.01, 0.001]
    early_stoppings = [2, 5]
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    max_points = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
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
                            gd_metric = train_gd(max_points=max_point, epochs=epoch, early_stopping=early_stopping, lr=lr, optimizer=optimizer, lr_scheduling=lr_schedule)
                            d.append([name, lr_schedule, epoch, max_point, lr, early_stopping, float(gd_metric)])
    df = pd.DataFrame(data=d, columns=columns)
    print(df)
    df.to_csv(os.path.join(logfolder, f"{key}_gdTest.csv"), index=False)
    return df


logfolder = "logs"
key = "affine"

parser = argparse.ArgumentParser()
parser.add_argument('--key', metavar='K', type=str, help='key to run tests')
args = parser.parse_args()
if args.key is not None:
    key = args.key
assert key in all_keys
classification = get_classification(key)
train_dataset, test_dataset = get_datasets(key)
metric = get_metric(key)

if __name__ == "__main__":
    test_gd_vs_hybrid_cost()