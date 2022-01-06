import torch
from evaluation import AccuracyMetric, metric_evaluate, get_incorrect_subset
from tqdm import tqdm
import time

val_cutoff = 32 * 10


class TrainingAlgorithm:
    def __init__(self, model, metric, train_dataset, test_dataset, batch_size):
        self.model = model
        self.metric = metric
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
        self.configured = False
        self._clean_configure_vars()

    def configure(self):
        raise ValueError

    def train(self, clean=False):
        if self.configured:
            to_ret = self._train()
        else:
            raise ValueError("Has not  been configured")
        if clean:
            self._clean_configure_vars()
        return to_ret

    def evaluate_loss(self):
        curr_loss_mean, curr_loss_var = metric_evaluate(self.model, self.test_loader, self.metric, "Loss")
        return curr_loss_mean

    def evaluate_accuracy(self):
        accuracy_metric = AccuracyMetric()
        curr_loss_mean, curr_loss_var = metric_evaluate(self.model, self.test_loader, accuracy_metric, "Accuracy")
        return curr_loss_mean

    def evaluate_l1(self):
        l1_metric = torch.nn.L1Loss()
        return metric_evaluate(self.model, self.test_loader, l1_metric, "L1")


class GradientDescent(TrainingAlgorithm):
    def __init__(self, model, metric, train_dataset, test_dataset, batch_size):
        TrainingAlgorithm.__init__(self, model, metric, train_dataset, test_dataset, batch_size)

    def configure(self, epochs, lr=None, early_stopping=None, max_points=None, early_stopping_max_points=val_cutoff, early_stop_batch=False, optimizer=torch.optim.SGD, lr_scheduling=False):
        self.epochs = epochs
        self.lr = lr
        self.early_stopping = early_stopping
        self.max_points = max_points
        self.early_stopping_max_points=early_stopping_max_points
        self.early_stopping_batch = early_stop_batch
        self.optimizer = optimizer
        self.lr_scheduling = lr_scheduling
        if early_stopping is not None and early_stopping_max_points is None:
            raise ValueError("Must specify early_stopping_max_points parameter as well")
        self.configured = True

    def _clean_configure_vars(self):
        self.epochs = None
        self.lr = None
        self.early_stopping = None
        self.max_points = None
        self.early_stopping_max_points = None
        self.early_stopping_batch = None
        self.optimizer = None
        self.lr_scheduling = None
        self.configured = False

    def _train(self):
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        if self.lr_scheduling:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.6, patience=2, verbose=True)
        if self.max_points is None:
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            self.train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.train_dataset, range(self.max_points)), batch_size=self.batch_size, shuffle=True)
        if self.early_stopping is not None:
            self.early_stopping_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.test_dataset, range(self.early_stopping_max_points)), batch_size=self.batch_size, shuffle=True)
        self.early_stopping_iters = 0
        self.best_loss = 10000
        for i in tqdm(range(self.epochs)):
            proceed = self.do_epoch()
            if not proceed:
                print(f"Batch Early Stopping After {i} iterations")
                return True
            if self.early_stopping is not None:
                if self.early_stopping_batch is None or not self.early_stopping_batch:
                    curr_loss_mean, curr_loss_var = metric_evaluate(self.model, self.early_stopping_loader, self.metric, "Loss", verbose=False)
                    if curr_loss_mean < self.best_loss:
                        self.best_loss = curr_loss_mean
                        self.early_stopping_iters = 0
                    else:
                        self.early_stopping_iters += 1
                    if self.early_stopping_iters >= self.early_stopping:
                        print(f"Epoch Early Stopping After {i} iterations")
                        return True
        return False

    def do_epoch(self):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.metric(output, target)
            loss.backward()
            self.optimizer.step()
            if self.early_stopping_batch:
                curr_loss_mean, curr_loss_var = metric_evaluate(self.model, self.early_stopping_loader, self.metric, "Loss", verbose=False)
                if curr_loss_mean < self.best_loss:
                    self.best_loss = curr_loss_mean
                    self.early_stopping_iters = 0
                else:
                    self.early_stopping_iters += 1
                if self.early_stopping_iters >= self.early_stopping:
                    return False
        return True


class SolverFineTuning(TrainingAlgorithm):
    def __init__(self, model, metric, train_dataset, test_dataset, batch_size):
        TrainingAlgorithm.__init__(self, model, metric, train_dataset, test_dataset, batch_size)

    def configure(self, n_iters=None, incorrect_subset=False):
        self.n_iters = n_iters
        self.incorrect_subset = incorrect_subset
        self.configured = True

    def _clean_configure_vars(self):
        self.n_iters = None
        self.incorrect_subset = False
        self.configured = False

    def _train(self):
        active_set = self.train_dataset
        if self.incorrect_subset:
            active_set = get_incorrect_subset(self.model, self.train_dataset, limit=self.batch_size)
        active_loader = torch.utils.data.DataLoader(active_set, batch_size=self.batch_size)
        data, target = next(iter(active_loader))
        if self.n_iters is None:
            self.n_iters = 1
        for i in tqdm(range(self.n_iters)):
            X = self.model.forward_till_dense(data)
            output = self.model(data)
            self.model.milp_model.initialize_mlp_model()
            if self.model.milp_model.classification:
                beforeAcc = torch.sum(torch.argmax(output, dim=1) == target) / len(target)
                min_accuracy = beforeAcc + 0.05
                self.model.milp_model.build_mlp_model(X, target, min_acc=min_accuracy)
            else:
                l1 = torch.abs(output - target)
                l1 = l1.mean()
                l1 = None
                self.model.milp_model.build_mlp_model(X, target, max_loss=l1)
            # model.milp_model.report_mlp(verbose=False, constraint_loop_verbose=True)
            self.model.milp_model.verbose = False # Force silence to improve time efficiency
            self.model.milp_model.solve_and_assign()
            # model.milp_model.report_mlp(verbose=False, constraint_loop_verbose=True))


class SolverGDHybridManual(TrainingAlgorithm):
    def __init__(self, model, metric, train_dataset, test_dataset, batch_size):
        TrainingAlgorithm.__init__(self, model, metric, train_dataset, test_dataset, batch_size)

    def configure(self, epochs_sequence, n_iters=None, incorrect_subset=None, lr_sequence=None,
                  max_points=None, classification=True, ):
        self.epoch_sequence = epochs_sequence
        self.lr_sequence = lr_sequence
        self.incorrect_subset = incorrect_subset if classification else False
        self.classification = classification
        self.sequence = []
        for i, e in enumerate(epochs_sequence):
            if isinstance(e, int):
                g = GradientDescent(self.model, self.metric, self.train_dataset, self.test_dataset, self.batch_size)
                g.configure(e, lr_sequence[i], max_points=max_points)
                self.sequence.append(g)
            elif e == "s" or e == "solver":
                s = SolverFineTuning(self.model, self.metric, self.train_dataset, self.test_dataset, self.batch_size)
                s.configure(n_iters=n_iters, incorrect_subset=self.incorrect_subset)
                self.sequence.append(s)
        self.configured = True

    def _clean_configure_vars(self):
        self.sequence = None
        self.classification = None
        self.configured = False

    def _train(self):
        for i, trainer in enumerate(self.sequence):
            if isinstance(trainer, GradientDescent):
                print(f"Begin Training Step {i+ 1}: Gradient Descent")
            elif isinstance(trainer, SolverFineTuning):
                print(f"Begin Training Step {i + 1}: Solver Fine Tuning")
            trainer.train()


class SolverGDHybrid(TrainingAlgorithm):
    def __init__(self, model, metric, train_dataset, test_dataset, batch_size):
        TrainingAlgorithm.__init__(self, model, metric, train_dataset, test_dataset, batch_size)
        self.last_gd_res = None

    def configure(self, epochs_sequence, n_iters=None, incorrect_subset=False, lr_sequence=None,
                  early_stopping=None, early_stopping_max_points=val_cutoff, early_stop_batch=False,
                  max_points=None, classification=True, optimizer=torch.optim.SGD, lr_scheduling=False):
        self.epoch_sequence = epochs_sequence
        self.lr_sequence = lr_sequence
        self.n_iters = n_iters
        self.incorrect_subset = incorrect_subset if classification else False
        self.classification = classification
        self.lr_scheduling = lr_scheduling
        self.sequence = []
        for i, e in enumerate(epochs_sequence):
            if isinstance(e, int):
                g = GradientDescent(self.model, self.metric, self.train_dataset, self.test_dataset, self.batch_size)
                g.configure(e, lr_sequence[i], max_points=max_points, early_stopping=early_stopping,
                            early_stopping_max_points=val_cutoff, early_stop_batch=early_stop_batch,
                            optimizer=optimizer, lr_scheduling=self.lr_scheduling)
                self.sequence.append(g)
            elif e == "s" or e == "solver":
                raise ValueError("Use SolverGDHybridManual Algorithm instead")
        self.configured = True

    def _clean_configure_vars(self):
        self.sequence = None
        self.classification = None
        self.n_iters = None
        self.selection_index = None
        self.configured = False

    def _train(self):
        for i, trainer in enumerate(self.sequence):
            if isinstance(trainer, GradientDescent):
                print(f"Begin Training Step {i+ 1}/{len(self.sequence)}: Gradient Descent")
            elif isinstance(trainer, SolverFineTuning):
                raise ValueError
            early_stop = trainer.train()
            print(f"Testing Results")
            self.evaluate_loss()
            if self.classification:
                tmp = self.evaluate_accuracy()
                if i == len(self.sequence) - 1:
                    self.last_gd_res = tmp
            else:
                tmp = self.evaluate_l1()
                if i == len(self.sequence) - 1:
                    self.last_gd_res = tmp
            if early_stop:
                print(f"Detected Early Stopping, calling Solver Fine Tuning")
                s = SolverFineTuning(self.model, self.metric, self.train_dataset, self.test_dataset,
                                     self.batch_size)
                s.configure(n_iters=self.n_iters, incorrect_subset=self.incorrect_subset)
                s.train()
