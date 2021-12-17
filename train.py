import torch
import os
from data import *
from algorithms import GradientDescent, SolverFineTuning, SolverGDHybrid

logfolder = "logs"

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


key = "mnist"

train_dataset, test_dataset = get_datasets(key)
metric = get_metric(key)


def train_gd(batch_size=32, epochs=2, lr=0.05, early_stopping=None, max_points=None):
    model = get_model(key)
    trainer = GradientDescent(model, metric, train_dataset, test_dataset, batch_size)
    trainer.configure(epochs=epochs, lr=lr, early_stopping=early_stopping, max_points=max_points)
    trainer.train()
    print("Evaluating GD Accuracy Overall")
    return trainer.evaluate_accuracy()


def train_hybrid(batch_size=32, epoch_sequence=[20, 20], lr_sequence=[0.1, 0.05], early_stopping=2, max_points=None, incorrect_subset=True):
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
    lr = 0.1
    early_stopping = 2
    epochs = [10, 20, 30]
    max_points = [1000, 5000, 10_000]
    lr_sequence = [lr, lr/2]

    for epoch in epochs:
        for max_point in max_points:
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Starting epoch {epoch} with max points {max_point} XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            epoch_sequence = [epoch // 2, epoch // 2]
            gd_acc = train_gd(max_points=max_point, epochs=epoch, early_stopping=early_stopping)
            last_gd, hybrid_acc = train_hybrid(max_points=max_point, epoch_sequence=epoch_sequence, early_stopping=early_stopping)
            d.append(["gd", epoch, max_point, float(gd_acc)])
            d.append(["hybrid last gd", epoch, max_point, float(last_gd)])
            d.append(["hybrid", epoch, max_point, float(hybrid_acc)])
            print(f"Intermediate Success: {last_gd > gd_acc}")
            print(f"Overall Success: {hybrid_acc > gd_acc}")
            print(f"Final Solver Failure: {last_gd > hybrid_acc}")
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    df = pd.DataFrame(data=d, columns=columns)
    print(df)
    df.to_csv(os.path.join(logfolder, f"{key}_gd_vs_hybrid.csv"), index=False)
    return df


def test_gd():
    d = []
    columns = ["epochs", "max_points", "lr", "early_stopping", "accuracy"]
    lrs = [0.1, 0.01]
    early_stoppings = [1, 2, 5, None]
    epochs = [20, 30, 40]
    max_points = [1000, 5000, 10_000]

    for epoch in epochs:
        for max_point in max_points:
            for lr in lrs:
                for early_stopping in early_stoppings:
                    print(
                        f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Starting epoch {epoch} with max points {max_point} "
                        f"XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                    gd_acc = train_gd(max_points=max_point, epochs=epoch, early_stopping=early_stopping, lr=lr)
                    d.append([epoch, max_point, lr, early_stopping, float(gd_acc)])
    df = pd.DataFrame(data=d, columns=columns)
    print(df)
    df.to_csv(os.path.join(logfolder, f"{key}_gdTest.csv"), index=False)
    return df


if __name__ == "__main__":
    test_gd_vs_hybrid()