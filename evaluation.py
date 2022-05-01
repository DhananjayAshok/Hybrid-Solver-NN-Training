import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    try:
        df0 = pd.read_csv(logfolder + f"/{key}_gdTest.csv")
    except FileNotFoundError:
        df0 = None
    try:
        df1 = pd.read_csv(logfolder + f"/{key}_gd_vs_hybrid_metric.csv")
    except FileNotFoundError:
        df1 = None
    try:
        df2 = pd.read_csv(logfolder + f"/{key}_gd_vs_hybrid_cost.csv")
    except FileNotFoundError:
        df2 = None
    if df0 is None and df1 is None and df2 is None:
        raise FileNotFoundError(f"None of the Files in {logfolder}/{key}_....csv could be found")
    return df0, df1, df2


def plot_max_points_vs_metric_by_method(df, exclude_methods=[], save_plot=False, key=None):
    if key is None:
        save_plot = False
    metric_col = "metric" if 'metric' in df.columns else "accuracy"
    methods = df['method'].unique()
    epochs = df['epochs'].unique()
    for method in methods:
        if method in exclude_methods:
            continue
        for epoch in epochs:
            subset = df[df["method"] == method]
            subset = subset[subset["epochs"] == epoch]
            plt.plot(subset["max_points"], subset[metric_col], label=f"{method}: {epoch} epochs")
    plt.legend()
    plt.xlabel("Max Points")
    plt.ylabel("Metric")
    title = f"{key if key is not None else ''} Max Points vs Metric by Method"
    plt.title(title)
    if not save_plot:
        plt.show()
    else:
        from train import logfolder
        plt.savefig(f"{logfolder}/{title}.jpg")
    plt.clf()
    return


def plot_metric_averages_by_method(df, exclude_methods=[], save_plot=False, key=None):
    if key is None:
        save_plot = False
    metric_col = "metric" if 'metric' in df.columns else "accuracy"
    methods = df['method'].unique()
    methods_to_plot = []
    averages = []
    stds = []
    for method in methods:
        if method in exclude_methods:
            continue
        methods_to_plot.append(method)
        averages.append(df[df["method"] == method][metric_col].mean())
        stds.append(df[df["method"] == method][metric_col].std())

    use_stds = not pd.isna(stds).any()
    plt.bar(x=methods_to_plot, height=averages, yerr=stds if use_stds else None)
    plt.xlabel("Methods")
    plt.ylabel("Avg Metric")
    title = f"{key if key is not None else ''} Metric Averages by Method"
    plt.title(title)
    if not save_plot:
        plt.show()
    else:
        from train import logfolder
        plt.savefig(f"{logfolder}/{title}.jpg")
    plt.clf()
    return


def plot_epochs_vs_time_by_method(df, exclude_methods=[], save_plot=False, key=None):
    if key is None:
        save_plot = False
    time_column = "time"
    methods = df['method'].unique()
    for method in methods:
        if method in exclude_methods:
            continue
        subset = df[df["method"] == method]
        plt.plot(subset["epochs"], subset[time_column], label=f"{method}")

    plt.xlabel("Epochs")
    plt.ylabel("Time")
    title = f"{key if key is not None else ''} Epochs vs Time by Method"
    plt.title(title)
    plt.legend()
    if not save_plot:
        plt.show()
    else:
        from train import logfolder
        plt.savefig(f"{logfolder}/{title}.jpg")
    plt.clf()
    return


def plot_epochs_vs_metric_by_method(df, exclude_methods=[], save_plot=False, key=None):
    if key is None:
        save_plot = False
    metric_col = "metric" if 'metric' in df.columns else "accuracy"
    methods = df['method'].unique()
    lrs = df["lr"].unique()
    for method in methods:
        if method in exclude_methods:
            continue
        for lr in lrs:
            subset = df[df["method"] == method]
            subset = subset[subset["lr"] == lr]
            plt.plot(subset["epochs"], subset[metric_col], label=f"{method}: lr {lr}")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    title = f"{key if key is not None else ''} Epochs vs Metric by Method"
    plt.title(title)
    if not save_plot:
        plt.show()
    else:
        from train import logfolder
        plt.savefig(f"{logfolder}/{title}.jpg")
    plt.clf()
    return


def plot_epochs_vs_metric_by_optimizer_and_max_points(df, save_plot=False, key=None):
    if key is None:
        save_plot = False
    metric_col = "metric" if 'metric' in df.columns else "accuracy"
    optimizers = df["optimizer"].unique()
    lr_scheduling = df["lr_scheduling"].unique()
    max_points = df["max_points"].unique()
    for optimizer in optimizers:
        subset = df[df["optimizer"] == optimizer]
        # subset = subset[subset["lr_scheduling"] == lr_schedule]
        # subset = subset[subset["max_points"] == max_point]
        plt.plot(subset["epochs"], subset[metric_col], label=f"{optimizer} "
                 )#f"{'with' if lr_schedule else 'without'} LRS")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    title = f"{key if key is not None else ''} Epochs vs Metric by Method and Max Points"
    plt.title(title)
    if not save_plot:
        plt.show()
    else:
        from train import logfolder
        plt.savefig(f"{logfolder}/{title}.jpg")
    plt.clf()
    return


def plot_0(df, save_plot=False, key=None):
    plot_epochs_vs_metric_by_optimizer_and_max_points(df, save_plot=save_plot, key=key)


def plot_1(df, exclusions=[], save_plot=False, key=None):
    plot_max_points_vs_metric_by_method(df, exclude_methods=exclusions, save_plot=save_plot, key=key)


def plot_2(df, exclusions=[], save_plot=False, key=None):
    plot_metric_averages_by_method(df, exclude_methods=exclusions, save_plot=save_plot, key=key)


def plot_3(df, exclusions=[], save_plot=False, key=None):
    plot_epochs_vs_time_by_method(df, exclude_methods=exclusions, save_plot=save_plot, key=key)


def plot_4(df, exclusions=[], save_plot=False, key=None):
    plot_epochs_vs_metric_by_method(df, exclude_methods=exclusions, save_plot=save_plot, key=key)


def slicer(df, slicing={}):
    """
    slicing in format {col: value}
    """
    for col in slicing:
        if col not in df.columns:
            print(f"Column {col} not in dataframe columns")
            continue
        df = df[df[col] == slicing[col]]
    return df


def gen_all_plots(keys=None, exclusions=[],  save_plot=True, slicing_0={}, slicing_1={}, slicing_2={}):
    if keys is None:
        from train import all_keys
        keys = all_keys
    for key in tqdm(keys):
        try:
            df_0, df_1, df_2 = get_log_df(key)
            if df_0 is not None:
                df_0 = slicer(df_0, slicing_0)
                if key == "cifar10":
                    pass
                else:
                    plot_0(df_0, key=key, save_plot=save_plot)
            if df_1 is not None:
                df_1 = slicer(df_1, slicing_1)
                plot_1(df_1, key=key, exclusions=exclusions, save_plot=save_plot)
                plot_2(df_1, key=key, exclusions=exclusions, save_plot=save_plot)
            if df_2 is not None:
                df_2 = slicer(df_2, slicing_2)
                plot_3(df_2, key=key, exclusions=exclusions, save_plot=save_plot)
                plot_4(df_2, key=key, exclusions=exclusions, save_plot=save_plot)
        except FileNotFoundError:
            print(f"Could not find files for {key}")
    return


if __name__ == "__main__":
    from train import all_keys, regression_keys, classification_keys
    gen_all_plots(slicing_2={"lr": 0.01})
