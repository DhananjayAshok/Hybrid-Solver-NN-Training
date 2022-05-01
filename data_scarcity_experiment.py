import torch
from data import IdentityDataset, AffineDataset, PolynomialDataset, FormulaDataset, ThresholdDataset, AdultsDataset, \
    MNISTDataset
from model import *


def get_loaders(key, train_batch_size=32):
    if key == "mnist":
        return MNISTDataset.loaders(train_batch_size=train_batch_size)
    elif key == "identity":
        return IdentityDataset.loaders()
    elif key == "affine":
        return AffineDataset.loaders()
    elif key == "polynomial":
        return PolynomialDataset.loaders()
    elif key == "formula":
        return FormulaDataset.loaders()
    elif key == "threshold":
        return ThresholdDataset.loaders()
    elif key == "adults":
        return AdultsDataset.loaders()


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


def train(model, optimizer, metric, data, target):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = metric(output, target)
    loss.backward()
    optimizer.step()
    return


def milp_train(model, metric, l1_metric, data, target):
    X = model.forward_till_dense(data)
    output = model(data)
    # target = encoder(target)
    # print(output.shape, target.shape)
    beforeloss = metric(output, target)
    beforeL1 = None
    afterL1 = None
    model.milp_model.initialize_mlp_model(w_range=0.1)
    if model.milp_model.classification:
        beforeAcc = torch.sum(torch.argmax(output, dim=1) == target)/len(target)
        min_accuracy = beforeAcc + 0.05
        model.milp_model.build_mlp_model(X, target, min_acc=min_accuracy)
    else:
        beforeL1 = l1_metric(output, target)
        l1 = torch.abs(output - target)
        l1= l1.mean()
        #l1 = None
        model.milp_model.build_mlp_model(X, target, max_loss=l1)
    #model.milp_model.report_mlp(verbose=False, constraint_loop_verbose=True)
    model.milp_model.solve_and_assign()
    #model.milp_model.report_mlp(verbose=False, constraint_loop_verbose=True)
    output = model(data)
    loss = metric(output, target)
    if model.milp_model.classification:
        afterAcc = torch.sum(torch.argmax(output, dim=1) == target)/len(target)
    else:
        afterL1 = l1_metric(output, target)
    print(f"Before loss was {beforeloss.item()}")
    print(f"Now loss is {loss.item()}")
    if model.milp_model.classification:
        print(f"Before Accuracy was {beforeAcc*100}%")
        print(f"Now Accuracy is {afterAcc*100}%")
    else:
        print(f"Before L1 was {beforeL1.item()}")
        print(f"Now L1 is {afterL1.item()}")
    return


def evaluate(model, metric, test_loader):
    model.eval()
    val_losses = []
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        loss = metric(output, target)
        val_losses.append(loss.item())
    val_losses = torch.Tensor(val_losses)
    print(f"Mean Loss: {torch.mean(val_losses)} Variance Loss: {torch.std(val_losses)}")


def acc_evaluate(model, test_loader):
    model.eval()
    val_losses = []
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        predictions = torch.argmax(output, axis=1)
        acc = (torch.sum(predictions == target)*100)/len(target)
        val_losses.append(acc.item())
    val_losses = torch.Tensor(val_losses)
    print(f"Mean Accuracy: {torch.mean(val_losses)} Variance Accuracy: {torch.std(val_losses)}")
    return torch.mean(val_losses), torch.std(val_losses)


def do_process(batch_size, epochs):
    print("\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print(f"Doing Process for batch size {batch_size} with epochs {epochs}")
    lr = 0.01
    key = "mnist"

    train_loader, test_loader = get_loaders(key, train_batch_size=batch_size)
    model = get_model(key)
    metric = get_metric(key)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_losses, train_counter, test_losses = [], [], []
    log_interval = 1500
    l1_metric = nn.L1Loss()

    data_0, target_0 = None, None
    data_1, target_1 = None, None
    i = 0
    for d, y in train_loader:
        if i == 0:
            data_0 = d
            target_0 = y
            i = i+1
        elif i == 1:
            data_1 = d
            target_1 = y
            i = i+1
        else:
            break
    for epoch in range(1, epochs + 1):
        train(model, optimizer, metric, data_0, target_0)

    for epoch in range(1, epochs + 1):
        train(model, optimizer, metric, data_1, target_1)

    baseline_mean, baseline_std = acc_evaluate(model, test_loader)

    model = get_model(key)
    metric = get_metric(key)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train(model, optimizer, metric, data_0, target_0)

    begin_mean, begin_std = acc_evaluate(model, test_loader)

    milp_train(model, metric, l1_metric, data_1, target_1)

    middle_mean, middle_std = acc_evaluate(model, test_loader)

    for epoch in range(1, epochs + 1):
        pass
        train(model, optimizer, metric, data_1, target_1)

    final_mean, final_std = acc_evaluate(model, test_loader)
    print(f"Experiment Success: {baseline_mean < final_mean}. Complete Success: {baseline_mean < middle_mean}")


if __name__ == "__main__":
    batch_sizes = [i+2 for i in range(15)]
    epochs = [i+2 for i in range(10)]
    for b in batch_sizes:
        for e in epochs:
            do_process(b, e)