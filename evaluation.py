import torch
import pandas as pd
import matplotlib.pyplot as plt

class AccuracyMetric:
    def __call__(self, output, target):
        predictions = torch.argmax(output, axis=1)
        acc = (torch.sum(predictions == target) * 100) / len(target)
        return acc.item()


def metric_evaluate(model, test_loader, metric, metric_name="Metric", verbose=True):
    model.eval()
    val_losses = []
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        loss = metric(output, target)
        if isinstance(loss, float):
            val_losses.append(loss)
        else:
            val_losses.append(loss.item())
    val_losses = torch.Tensor(val_losses)
    if verbose:
        print(f"Mean {metric_name}: {torch.mean(val_losses)} Variance {metric_name}: {torch.std(val_losses)}")
    return torch.mean(val_losses), torch.std(val_losses)


def accuracy_evaluate(model, test_loader, verbose=True):
    metric = AccuracyMetric()
    return metric_evaluate(model, test_loader, metric, metric_name="Accuracy", verbose=verbose)


def get_incorrect_subset(model, train_dataset, limit=None):
    indices = []
    for i, data in enumerate(train_dataset):
        X, y = data
        X = X.view((1, ) + X.shape)
        pred = model.predict(X)
        if all(y == pred):
            pass
        else:
            indices.append(i)
        if limit is not None:
            if len(indices) > limit:
                break
    return torch.utils.data.Subset(train_dataset, indices)


def get_log_df(key):
    from train import logfolder
    df0 = pd.read_csv(logfolder + f"/{key}_gdTest.csv")
    df1 = pd.read_csv(logfolder + f"/{key}_gd_vs_hybrid.csv")
    return df0, df1


def plot_max_points_vs_accuracy_by_method(df, exclude_methods=[]):
    methods = df['method'].unique()
    epochs = df['epochs'].unique()
    for method in methods:
        if method in exclude_methods:
            continue
        for epoch in epochs:
            subset = df[df["method"] == method]
            subset = subset[subset["epochs"] == epoch]
            plt.plot(subset["max_points"], subset["accuracy"], label=f"{method}: {epoch} epochs")
    plt.legend()
    plt.xlabel("Max Points")
    plt.ylabel("Accuracy")
    plt.title(f"Max Points vs Accuracy by Method")
    plt.show()
    return


def plot_epochs_vs_accuracy_by_optimizer_and_max_points(df):
    optimizers = df["optimizer"].unique()
    lr_scheduling = df["lr_scheduling"].unique()
    max_points = df["max_points"].unique()
    for optimizer in optimizers:
        subset = df[df["optimizer"] == optimizer]
        # subset = subset[subset["lr_scheduling"] == lr_schedule]
        # subset = subset[subset["max_points"] == max_point]
        plt.plot(subset["epochs"], subset["accuracy"], label=f"{optimizer} "
                 )#f"{'with' if lr_schedule else 'without'} LRS")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Epochs vs Accuracy by Method and Max Points")
    plt.show()


def plot_0(df, exclusions=[]):
    plot_max_points_vs_accuracy_by_method(df, exclude_methods=exclusions)


def plot_1(df):
    plot_epochs_vs_accuracy_by_optimizer_and_max_points(df)
