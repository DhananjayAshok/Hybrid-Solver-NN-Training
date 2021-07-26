import gurobipy as gp
from gurobipy import GRB

import torch


m = gp.Model("MIP")
# y = w2(w1x1)
x = 5
y = 100
w1 = m.addVar(lb=2, vtype=GRB.CONTINUOUS, name="w1")
w2 = m.addVar(lb=2, vtype=GRB.CONTINUOUS, name="w2")
a1 = m.addVar(vtype=GRB.CONTINUOUS, name="a1")
m.addConstr(a1 == x * w1, name="c1")
m.addConstr(y == w2 * a1)
try:
    m.optimize()
except:
    pass