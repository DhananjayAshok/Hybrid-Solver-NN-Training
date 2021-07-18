import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB

class MILPNet(nn.Module):
    """
    Class to hold the last few dense layers of a network.
    Model will be an instance of a Sequential model, i.e list of NamedLinear layers
    The names of the layers must be index numbers.
    """
    def __init__(self, model, classification=True):
        nn.Module.__init__(self)
        self.model = model
        self.classification = classification
        self.n_layers = len(self.model)
        self.m = None
        self.initialize_mlp_model() # defines self.m

    def forward(self, x):
        return self.model(x)

    def assign(self):
        with torch.no_grad():
            for l in range(self.n_layers):
                output_dim = self.model[l].out_features
                input_dim = self.model[l].in_features
                for j in range(output_dim):
                    for i in range(input_dim):
                        self.model[l].weight[j, i] = self.m.getVarByName(f"w_{l},{i},{j}").x
                    self.model[l].bias[j] = self.m.getVarByName(f"b_{l},{j}").x

    def initialize_mlp_model(self, w_range=10):
        """
        Sets up the mapping between the weights and biases of each layer with variables in a MILP model
        :return:
        """
        m = gp.Model("MLP")
        w_b_var_dict = {}
        for l in range(self.n_layers):
            output_dim = self.model[l].out_features
            input_dim = self.model[l].in_features
            for j in range(output_dim):
                for i in range(input_dim):
                    w_b_var_dict[(l, i, j)] = m.addVar(lb=float(self.model[l].weight[j, i])-w_range/2,
                                                       ub=float(self.model[l].weight[j, i])+w_range/2,
                                                       vtype=GRB.CONTINUOUS, name=f"w_{l},{i},{j}")
                w_b_var_dict[(l, j)] = m.addVar(vtype=GRB.CONTINUOUS, name=f"b_{l},{j}")
        self.w_b_var_dict = w_b_var_dict
        self.m = m
        self.m.params.NonConvex = 2


    def build_mlp_model(self, X, y):
        """
        Encodes the entire network because one layer is the network here.
        :return:
        """
        batch_size , n_units = X.shape
        inp_out_var_dict = {} # (n, l, j, 0) is the value of the forward pass pre activation of layer l, neuron j
        # when data point n is fed into the model. The input into layer l+1 is (n, l, j, 1) i.e post activation.
        for l in range(self.n_layers):
            output_dim = self.model[l].out_features
            input_dim = self.model[l].in_features
            activation = (self.model[l].activation is not None)
            for n in range(batch_size):
                output_constraint_string = None
                for j in range(output_dim):
                    # y_j = sum over i wijxi
                    weighted_sum_string = None
                    weighted_sum_constraint_string = None
                    activation_constraint_string = None
                    if l == 0:
                        assert n_units == input_dim
                        weighted_sum_string = " + ".join([f"self.w_b_var_dict[{(l, i, j)}] * {X[n][i]}"
                                                        for i in range(input_dim)]) +  f" + self.w_b_var_dict[{(l, j)}]"
                    else:
                        weighted_sum_string = " + ".join([f"self.w_b_var_dict[{(l, i, j)}] * inp_out_var_dict[{(n, l-1, i, 1)}]"
                                                          for i in range(input_dim)]) + f" + self.w_b_var_dict[{(l, j)}]"
                    inp_out_var_dict[(n, l, j, 0)] = self.m.addVar(vtype=GRB.CONTINUOUS, name=f"ws_{n},{l},{j},{0}")
                    inp_out_var_dict[(n, l, j, 1)] = self.m.addVar(vtype=GRB.CONTINUOUS, name=f"act_{n},{l},{j},{1}")
                    weighted_sum_constraint_string = weighted_sum_string + f"== inp_out_var_dict[{(n, l, j, 0)}]"

                    if activation:
                        activation_constraint_string = f"inp_out_var_dict[{(n, l, j, 1)}] == " \
                                                       f"gp.max_(inp_out_var_dict[{(n, l, j, 0)}], 0)"
                    else:
                        activation_constraint_string = f"inp_out_var_dict[{(n, l, j, 1)}] == inp_out_var_dict[{(n, l, j, 0)}]"

                    self.m.addConstr(eval(weighted_sum_constraint_string), f"WS_{l},{j} datapoint {n}")
                    self.m.addConstr(eval(activation_constraint_string), f"ACT_{l},{j} datapoint {n}")
                    if l == self.n_layers-1 and not self.classification:
                        self.m.addConstr(inp_out_var_dict[(n, l, j, 1)] == y[n][j], f"Y_{j} datapoint {n}")

                if l == self.n_layers - 1:
                    if self.classification:
                        correct_label = torch.argmax(y[n])
                        max_string = "gp._max(" + ", ".join([f"inp_out_var_dict[{(n, l, j, 1)}]"] for j in output_dim) \
                                     + ")"
                        output_constraint_string = max_string + f" == inp_out_var_dict[{(n, l, correct_label, 1)}]"
                        self.m.addConstr(eval(output_constraint_string), f"Output datapoint {n}")
        return


    def solve_mlp_model(self):
        self.m.optimize()

    def solve_and_assign(self):
        self.solve_mlp_model()
        self.assign()


class NamedLinear(nn.Linear):

    def __init__(self, *kargs, name="", activation="",  **kwargs):
        super(NamedLinear, self).__init__(*kargs, **kwargs)
        self.name = name
        self.activation = activation

class NamedConv2d(nn.Conv2d):

    def __init__(self, *kargs, name="", activation="",**kwargs):
        super(NamedConv2d, self).__init__(*kargs, **kwargs)
        self.name = name
        self.activation = activation