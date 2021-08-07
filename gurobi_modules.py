import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB

n_digits = 7
epsilon = 0.0000001

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
        self.constraints = {}
        self.initialize_mlp_model(w_range=w_range) # defines self.m
        self.w_range = w_range
        self.report_mlp()

    def forward(self, x):
        return self.model(x)

    def constraint_type(self):
        if len(self.constraints) == 0:
            return None
        _, key = next(enumerate(self.constraints.keys()))
        if len(key) == 2:
            return "regreesion_eq"
        elif len(key) == 3 and key[2] == 0:
            return "regression_max_loss"
        elif len(key) == 3 and key[2] == 1:
            return "classification"

    def assign(self):
        if self.m.SolCount <= 0:
            print(f"Cannot Assign: MLP solver found no solutions")
            return
        with torch.no_grad():
            for l in range(self.n_layers):
                output_dim = self.model[l].out_features
                input_dim = self.model[l].in_features
                for j in range(output_dim):
                    for i in range(input_dim):
                        self.model[l].weight[j, i] = self.m.getVarByName(f"w_{l},{i},{j}").x
                    self.model[l].bias[j] = self.m.getVarByName(f"b_{l},{j}").x

    def assign_start(self):
        with torch.no_grad():
            for l in range(self.n_layers):
                output_dim = self.model[l].out_features
                input_dim = self.model[l].in_features
                for j in range(output_dim):
                    for i in range(input_dim):
                        self.model[l].weight[j, i] = self.m.getVarByName(f"w_{l},{i},{j}").start
                    self.model[l].bias[j] = self.m.getVarByName(f"b_{l},{j}").start

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
                m.update()
        self.w_b_var_dict = w_b_var_dict
        self.m = m
        self.m.update()
        #self.m.params.NonConvex = 2

    def build_mlp_model(self, X, y, max_loss=None, w_range=None):
        """
        Encodes the entire network because one layer is the network here.
        :return:
        """
        if w_range is None:
            w_range = self.w_range
        self.initialize_mlp_model(w_range=w_range)
        batch_size , n_units = X.shape
        inp_out_var_dict = {} # (n, l, j, 0) is the value of the forward pass pre activation of layer l, neuron j
        # when data point n is fed into the model. The input into layer l+1 is (n, l, j, 1) i.e post activation.
        self.constraints = {}
        for l in range(self.n_layers):
            output_dim = self.model[l].out_features
            input_dim = self.model[l].in_features
            activation = (self.model[l].activation == "relu")
            for n in range(batch_size):
                for j in range(output_dim):
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
                            self.constraints[(j, n)] = (target_string, X[n], tensor_round(y[n][j]))
                            self.m.addConstr(eval(target_string), f"Y_{j} datapoint {n}")
                        else:
                            #loss = self.m.addVar(ub=max_loss, vtype=GRB.CONTINUOUS, name=f"Loss_{j}_{n}")
                            #self.loss_vars.append(loss)
                            lhs_target = inp_out_var_dict[(n, l, j)] + f"- {tensor_round(y[n][j])}"
                            self.constraints[(j, n, 0)] = (inp_out_var_dict[(n, l, j)], X[n], tensor_round(y[n][j])
                                                           , max_loss)
                            self.m.addConstr(eval(lhs_target) <= max_loss, f"Y_{j} datapoint {n} bound above {max_loss}")
                            self.m.addConstr(-eval(lhs_target) <= max_loss, f"Y_{j} datapoint {n} bound below {max_loss}")
                if l == self.n_layers - 1:
                    if self.classification:
                        correct_label = y[n]
                        correct_expression = inp_out_var_dict[(n, l, int(correct_label))]
                        for j in range(output_dim):
                            if j == correct_label:
                                pass
                            else:
                                self.constraints[(j, n, 1)] = (correct_expression, inp_out_var_dict[(n, l, j)], X[n],
                                                               correct_label)
                                self.m.addConstr(eval(correct_expression) >= epsilon + eval(inp_out_var_dict[(n, l, j)]),
                                                 f"Output datapoint {n} {j} vs {correct_label}")
        print(f"Finished Building Model")
        self.m.update()
        return

    def solve_mlp_model(self):
        self.m.optimize()
        self.report_mlp()

    def solve_and_assign(self):
        self.solve_mlp_model()
        self.assign()

    def report_mlp(self, verbose=False, constraint_loop_verbose=False):
        print(self.m)
        if len(self.constraints) != 0:
            self.loop_constraints(eval_attr="start", verbose=constraint_loop_verbose)
        if self.m.SolCount <= 0:
            print(f"MLP solver has not found solutions")
        elif len(self.constraints) != 0:
            self.loop_constraints(verbose=constraint_loop_verbose)

        if verbose:
            print("Printing Variables")
            items = self.m.getVars()
            if len(items) == 0:
                pass
            else:
                self.m.printAttr("start")
                if hasattr(items[0], 'x'):
                    self.m.printAttr('x')

    def loop_constraints(self, eval_attr="x", verbose=False):
        if self.constraints is not None:
            overall = len(self.constraints)
            true = 0
            for key in self.constraints:
                if self.check_constraint(key, eval_attr=eval_attr, verbose=verbose):
                    true += 1
            if eval_attr == "x":
                print(f"After assignment {(100*true)/overall}% of the constraints ({true}/{overall}) were true")
            elif eval_attr == "start":
                print(f"Warm Start assignment {(100*true)/overall}% of the constraints ({true}/{overall}) were true")

    def check_constraint(self, key, eval_attr="x", verbose=False):
        c_type = self.constraint_type()
        if c_type is None:
            return False
        elif c_type == "regression_eq":
            return self.check_constraint_regression_eq(key, eval_attr=eval_attr, verbose=verbose)
        elif c_type == "regression_max_loss":
            return self.check_constraint_regression_max_loss(key, eval_attr=eval_attr, verbose=verbose)
        elif c_type == "classification":
            return self.check_constraint_classification(key, eval_attr=eval_attr, verbose=verbose)
        else:
            print(f"Unknown Constraint Type")
            return False

    def check_constraint_regression_eq(self, key, eval_attr="x", verbose=False):
        expression = self.constraints[key][0]
        result = round(utils_eval_expression_regression(self, expression, eval_attr, cuteq=True), ndigits=n_digits)
        intended = self.constraints[key][2]
        if verbose:
            print(f"{key[0], key[1]} {'SAT' if result == intended else 'UNSAT'}|| Result: {result} vs Intended {intended}")
        return intended == result

    def check_constraint_regression_max_loss(self, key, eval_attr="x", verbose=False):
        expression = self.constraints[key][0]
        output = round(utils_eval_expression_regression(self, expression, eval_attr, cuteq=False), ndigits=n_digits)
        intended = self.constraints[key][2]
        max_loss = self.constraints[key][3]
        abs_diff = abs(intended - output)
        if verbose:
            print(f"{key[0], key[1]} {'SAT' if abs_diff <= max_loss else 'UNSAT'}|| Result: {output} vs Intended {intended} (diff {abs_diff}) with "
                  f"max_loss {max_loss}")
        return abs_diff <= max_loss

    def check_constraint_classification(self, key, eval_attr="x", verbose=True):
        correct_expression = self.constraints[key][0]
        incorrect_expression = self.constraints[key][1]
        label = self.constraints[key][3]
        correct_eval = round(utils_eval_expression_regression(self, correct_expression, eval_attr, cuteq=False),
                             ndigits=n_digits)
        incorrect_eval = round(utils_eval_expression_regression(self, incorrect_expression, eval_attr, cuteq=False),
                               ndigits=n_digits)
        condition = correct_eval >= incorrect_eval + epsilon
        if verbose:
            print(
                f"{key[0], key[1]} {'SAT' if condition else 'UNSAT'}|| Label {key[0]} confidence {incorrect_eval}| correct label "
                f": {label} confidence {correct_eval} ")
        return condition


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


def utils_eval_expression_regression(model, expression, eval_attr="x", cuteq=False):
    expression = expression.replace("]", f"].{eval_attr}")
    expression = expression.replace("self", "model")
    if cuteq:
        expression = expression.split("==")[0]
    return eval(expression)


def utils_model_eval(model, key, cuteq = False):
    expression = model.constraints[key][0]
    return utils_eval_expression_regression(model, expression, cuteq)


def tensor_round(arr, ndigits=n_digits):
    """
    Converts to float and performs round
    """
    return round(float(arr), ndigits)