import gurobipy as gp
from gurobipy import GRB

import torch
import torch.nn as nn

import sys
#print(sys.getrecursionlimit())
#sys.setrecursionlimit(1500)

output_dim = 10
input_dim = 1875
batch_size = 32
# dummy example of a NN with a single layer.
X = torch.rand(batch_size, input_dim)
Y = X[:, :output_dim] + 1
X_v = torch.rand(batch_size, input_dim)
Y_v = X_v[:, :output_dim] + 1

class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = nn.Linear(input_dim, output_dim)
        self.m = None
        self.build_mlp_model() # defines self.m

    def forward(self, x):
        return self.model(x)

    def assign(self):
        with torch.no_grad():
            for j in range(self.output_dim):
                for i in range(self.input_dim):
                    self.model.weight[j, i] = self.m.getVarByName(f"w_{i},{j}").x
                self.model.bias[j] = self.m.getVarByName(f"b_{j}").x

    def build_mlp_model(self):
        """
        Encodes the entire network because one layer is the network here.
        :return:
        """
        m = gp.Model("MLP")
        var_dict = {}
        for j in range(output_dim):
            for i in range(input_dim):
                var_dict[(i, j)] = m.addVar(vtype=GRB.CONTINUOUS, name=f"w_{i},{j}")
            var_dict[j] = m.addVar(vtype=GRB.CONTINUOUS, name=f"b_{j}")
        for i in range(batch_size):
            x = X[i]
            y = Y[i]
            for j in range(output_dim):
                # y_j = sum over i wijxi
                weight_sum_string = " + ".join(
                    [f"var_dict[{(iii, j)}] * {x[iii]}" for iii in range(len(x))]) + f" == {y[j]}"
                constraint_string = f"var_dict[{j}] + " + weight_sum_string
                m.addConstr(eval(constraint_string), f"y_{j} datapoint {i}")
        self.m = m
        return m

    def solve_mlp_model(self):
        self.m.optimize()


model = SimpleNet(input_dim=input_dim, output_dim=output_dim)
model.solve_mlp_model()
model.assign()
print(model.model.weight)
pred = model(X)

print("Train MSE Error", torch.mean(pred - Y)**2)

pred = model(X_v)

print("Test MSE Error", torch.mean(pred - Y_v)**2)
