import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB

n_digits = 7

class MILPNet(nn.Module):
    """
    Class to hold the last few dense layers of a network.
    Model will be an instance of a Sequential model, i.e list of NamedLinear layers
    The names of the layers must be index numbers.
    """
    def __init__(self, model, classification=True, w_range=10):
        nn.Module.__init__(self)
        self.model = model
        self.classification = classification
        self.n_layers = len(self.model)
        self.m = None
        self.initialize_mlp_model(w_range=w_range) # defines self.m

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
                    w_b_var_dict[(l, i, j)].start = float(self.model[l].weight[j, i])
                w_b_var_dict[(l, j)] = m.addVar(vtype=GRB.CONTINUOUS, name=f"b_{l},{j}")
                w_b_var_dict[(l, j)].start = float(self.model[l].bias[j])
        self.w_b_var_dict = w_b_var_dict
        self.m = m
        #self.m.params.NonConvex = 2


    def build_mlp_model(self, X, y, max_loss=None):
        """
        Encodes the entire network because one layer is the network here.
        :return:
        """
        batch_size , n_units = X.shape
        inp_out_var_dict = {} # (n, l, j, 0) is the value of the forward pass pre activation of layer l, neuron j
        # when data point n is fed into the model. The input into layer l+1 is (n, l, j, 1) i.e post activation.
        self.constraints = {}
        for l in range(self.n_layers):
            output_dim = self.model[l].out_features
            input_dim = self.model[l].in_features
            activation = (self.model[l].activation == "relu")
            for n in range(batch_size):
                output_constraint_string = None
                for j in range(output_dim):
                    # y_j = sum over i wijxi
                    weighted_sum_string = None
                    weighted_sum_constraint_string = None
                    activation_constraint_string = None
                    if l == 0:
                        assert n_units == input_dim
                        weighted_sum_string = " + ".join([f"self.w_b_var_dict[{(l, i, j)}] * {tensor_round(X[n][i])}"
                                                        for i in range(input_dim)]) +  f" + self.w_b_var_dict[{(l, j)}]"
                    else:
                        weighted_sum_string = " + ".join([f"self.w_b_var_dict[{(l, i, j)}] * " + inp_out_var_dict[(n, l-1, i)]
                                                          for i in range(input_dim)]) + f" + self.w_b_var_dict[{(l, j)}]"
                    if activation:
                        inp_out_var_dict[(n, l, j)] = "gp.max_("+weighted_sum_string+", 0)"
                    else:
                        inp_out_var_dict[(n, l, j)] = weighted_sum_string

                    if l == self.n_layers-1 and not self.classification:
                        if max_loss is None:
                            target_string = inp_out_var_dict[(n, l, j)] + f" == {tensor_round(y[n][j])}"
                            #print(X[n])
                            #print(f"{y[n][j].data} ", y[n][j])
                            #print(target_string)
                            self.constraints[(j, n)] = (target_string, X[n], tensor_round(y[n][j]))
                            self.m.addConstr(eval(target_string), f"Y_{j} datapoint {n}")
                        else:
                            #loss = self.m.addVar(ub=max_loss, vtype=GRB.CONTINUOUS, name=f"Loss_{j}_{n}")
                            #self.loss_vars.append(loss)
                            lhs_target = inp_out_var_dict[(n, l, j)] + f"- {y[n][j]}"
                            self.m.addConstr(eval(lhs_target) <= max_loss, f"Y_{j} datapoint {n} bound above {max_loss}")
                            self.m.addConstr(-eval(lhs_target) <= max_loss, f"Y_{j} datapoint {n} bound below {max_loss}")
                if l == self.n_layers - 1:
                    if self.classification:
                        correct_label = torch.argmax(y[n])
                        correct_expression = inp_out_var_dict[(n, l, int(correct_label))]
                        for j in range(output_dim):
                            if j == correct_label:
                                pass
                            else:
                                self.m.addConstr(eval(correct_expression) + 0.0001 >= eval(inp_out_var_dict[(n, l, j)]),
                                                 f"Output datapoint {n} {j} vs {correct_label}")
        print(f"Finished Building Model")
        return


    def solve_mlp_model(self):
        self.m.optimize()
        self.report_mlp()

    def solve_and_assign(self):
        self.solve_mlp_model()
        self.assign()

    def report_mlp(self, verbose=False):
        print(self.m)
        if len(self.constraints) != 0:
            self.loop_constraints(verbose=verbose)
        if verbose:
            print("Printing Variables")
            for item in self.m.getVars():
                print(item)

    def loop_constraints(self, verbose=False):
        if self.constraints is not None:
            overall = len(self.constraints)
            true = 0
            for key_j, key_n in self.constraints:
                expression = self.constraints[(key_j, key_n)][0]
                result = round(utils_eval_expression(self, expression, cuteq=True), ndigits=n_digits)
                intended = self.constraints[(key_j, key_n)][2]
                if verbose:
                    print(f"{key_j, key_n}|| Result: {result} vs Intended {intended}")
                if(result == intended):
                    true += 1
            print(f"After assignment {(100*true)/overall}% of the constraints ({true}/{overall}) were true")

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


def utils_eval_expression(model, expression, cuteq=False):
    expression = expression.replace("]", "].x")
    expression = expression.replace("self", "model")
    if cuteq:
        expression = expression.split("==")[0]
    return eval(expression)

def utils_model_eval(model, key, cuteq = False):
    expression = model.constraints[key][0]
    return utils_eval_expression(model, expression, cuteq)

def tensor_round(arr, ndigits=n_digits):
    """
    Converts to float and performs round
    """
    return round(float(arr), ndigits)