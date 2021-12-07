import torch


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
