import gurobipy as gp
from gurobipy import GRB

import torch


output_dim = 10
input_dim = 12
batch_size = 32
# dummy example of a NN with a single layer.
X = torch.randint(0, 1, (batch_size, input_dim))
Y = X[:, :output_dim]
X_v = torch.randint(0, 1, (batch_size, input_dim))
Y_v = X_v[:, :output_dim]


W = torch.randint(-1, 1, (input_dim, output_dim))

m = gp.Model("MLP")
var_dict = {}
for i in range(input_dim):
    for j in range(output_dim):
        var_dict[(i, j)] = m.addVar(lb=-1, ub=1, vtype=GRB.INTEGER, name=f"w_{i},{j}")

for i in range(batch_size):
    x = X[i]
    y = Y[i]
    for j in range(output_dim):
        #y_j = sum over i wijxi
        constraint_string = " + ".join([f"var_dict[{(iii, j)}] * {x[iii]}" for iii in range(len(x)) ] ) + f" == {y[j]}"
        m.addConstr(eval(constraint_string), f"y_{j} datapoint {i}")

m.optimize()
for i in range(input_dim):
    for j in range(output_dim):
        W[i, j] = m.getVarByName(f"w_{i},{j}").x

pred = torch.rand(batch_size, output_dim)
for i in range(batch_size):
    pred[i] = torch.matmul(X[i], W)

print("Train MSE Error", torch.mean(pred - Y)**2)

pred = torch.rand(batch_size, output_dim)
for i in range(batch_size):
    pred[i] = torch.matmul(X_v[i], W)

print("Test MSE Error", torch.mean(pred - Y_v)**2)
