import torch
import mnist as m_train
from adults import AdultsDataset
from toy_data import IdentityDataset, AffineDataset, PolynomialDataset, FormulaDataset, ThresholdDataset
from algorithms import GradientDescent, SolverFineTuning, SolverGDHybrid
from evaluation import accuracy_evaluate
from model import *


def get_datasets(key):
    if key == "mnist":
        return m_train.get_datasets()
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
        return AdultsDataset.get_datasets()


def get_model(key):
    if key == "mnist":
        return m_train.model()
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


def get_metric(key):
    if key == "mnist":
        return m_train.metric()
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


key = "adults"

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
    print("Evaluating Hybrid Accuracy Overall")
    return trainer.evaluate_accuracy()


if __name__ == "__main__":
    lr = 0.1
    early_stopping = 2
    epochs = [5, 10, 20, 30]
    max_points = [10, 32, 100, 1000, 5000]
    lr_sequence = [lr, lr/2]

    for epoch in epochs:
        for max_point in max_points:
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Starting epoch {epoch} with max points {max_point} XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            epoch_sequence = [epoch // 2, epoch // 2]
            gd_acc = train_gd(max_points=max_point, epochs=epoch, early_stopping=early_stopping)
            hybrid_acc = train_hybrid(max_points=max_point, epoch_sequence=epoch_sequence, early_stopping=early_stopping)
            print(f"Overall Success: {hybrid_acc > gd_acc}")
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")