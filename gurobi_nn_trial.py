from gurobi_modules import NamedLinear, MILPNet
import torch
import torch.nn as nn

import sys
#print(sys.getrecursionlimit())
#sys.setrecursionlimit(1500)

output_dim = 2
input_dim = 1875
batch_size = 32
# dummy example of a NN with a single layer.
X = torch.rand(batch_size, input_dim)
Y = X[:, :output_dim] + 1
X_v = torch.rand(batch_size, input_dim)
Y_v = X_v[:, :output_dim] + 1

sequential_model = nn.Sequential(NamedLinear(input_dim, 10, activation="relu"), NamedLinear(10, output_dim))
model = MILPNet(sequential_model, classification=False)
model.build_mlp_model(X, Y)
model.solve_and_assign()
relu = nn.ReLU()
pred = sequential_model[1](relu(sequential_model[0](X)))

print("Train MSE Error", torch.mean(pred - Y)**2)

pred = sequential_model[1](relu(sequential_model[0](X_v)))

print("Test MSE Error", torch.mean(pred - Y_v)**2)
