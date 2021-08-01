from gurobi_modules import NamedLinear, MILPNet
import torch
import torch.nn as nn

import sys
#print(sys.getrecursionlimit())
#sys.setrecursionlimit(1500)

output_dim = 3
input_dim = 3
batch_size = 50
# dummy example of a NN with a single layer.
X = torch.rand(batch_size, input_dim)
Y = X[:, :output_dim]**2 + 2*X[:, :output_dim] + 1
X_v = torch.rand(batch_size, input_dim)
Y_v = X_v[:, :output_dim]**2 + 2*X_v[:, :output_dim] + 1

sequential_model = nn.Sequential(NamedLinear(input_dim, output_dim))
model = MILPNet(sequential_model, classification=False)
# 5

model.build_mlp_model(X, Y, max_loss=0.001)
model.solve_and_assign()
model.report_mlp(verbose=False, constraint_loop_verbose=True)
relu = nn.ReLU()
metric = nn.L1Loss()
pred = model(X)#sequential_model[1](relu(sequential_model[0](X)))

print(f"Train {metric} Error", metric(pred, Y))

pred =model(X_v)# sequential_model[1](relu(sequential_model[0](X_v)))

print(f"Test {metric} Error", metric(pred, Y_v))

