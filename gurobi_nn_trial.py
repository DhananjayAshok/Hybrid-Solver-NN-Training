from model import SingleLayerRegression
from gurobi_modules import MILPNet, NamedLinear
import torch
import gurobipy as gp
from gurobipy import GRB
import torch.nn as nn

from toy_data import ThresholdDataset

input_dim = 3
output_dim = 2
t = ThresholdDataset(100, input_dim)
# dummy example of a NN with a single layer.
X = t.X
Y = t.y


# 5
model = MILPNet(nn.Sequential(NamedLinear(input_dim, output_dim)), classification=True, w_range=10)
model.build_mlp_model(X, Y)
model.solve_and_assign()
#model.report_mlp(verbose=True, constraint_loop_verbose=True)