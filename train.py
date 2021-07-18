import torch

from model import *
from torchvision import datasets, transforms

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       lambda x: x.float(),
            ])),
    batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       lambda x: x.float(),
                   ])),
    batch_size=32, shuffle=True)


model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(10 + 1)]
metric = nn.CrossEntropyLoss()
log_interval = 1500

def encoder(labels):
    true_labels = []
    for item in labels:
        base = [0.0 for i in range(10)]
        base[item] = 1.0
        true_labels.append(base)
    return torch.Tensor(true_labels)

def train(epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    #target = encoder(target)
    #print(output.shape, target.shape)
    loss = metric(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      #torch.save(model.state_dict(), '/results/model.pth')
      #torch.save(optimizer.state_dict(), '/results/optimizer.pth')

def milp_train():
    for batch_idx, (data, target) in enumerate(train_loader):
        X = model.forward_till_dense(data)
        output = model(data)
        # target = encoder(target)
        # print(output.shape, target.shape)
        beforeloss = metric(output, target)

        model.fc.build_mlp_model(X, target, )
        model.fc.solve_and_assign()
        output = model(data)
        loss = metric(output, target)
        print(f"Before loss was {beforeloss.item()}")
        print(f"Now loss is {loss.item()}")
        break

def eval():
    model.eval()
    val_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = metric(output, target)
        val_losses.append(loss.item())
    val_losses = torch.Tensor(val_losses)
    print(f"Mean Loss: {torch.mean(val_losses)} Variance Loss: {torch.std(val_losses)}")

for epoch in range(1, 10 + 1):
  train(epoch)
  pass

eval()

milp_train()

eval()