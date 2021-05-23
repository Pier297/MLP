from MLP.Optimizers import GradientDescent
from MLP.Network import Sequential
from MLP.Regularizers import _early_stopping
from MLP.LossFunctions import CrossEntropy, MSE
from MLP.experiments.utils import set_seed
from math import inf

# Given for each hyperparameter a list of possible values Returns a list of hyperparameters configurations, where each conf is a dictionary
def generate_hyperparameters(loss_func_values, lr_values, L2_values, momentum_values, hidden_layers_values, BATCH_SIZE_values):
    i = 0
    configurations = []
    for loss_func in loss_func_values:
        for lr in lr_values:
            for L2 in L2_values:
                for momentum in momentum_values:
                    for hidden_layers in hidden_layers_values:
                        for BATCH_SIZE in BATCH_SIZE_values:
                            configurations.append({"loss_function": loss_func, "lr": lr, "L2": L2, "momentum": momentum, "hidden_layers": hidden_layers, "BATCH_SIZE": BATCH_SIZE})
                            i += 1
    print(f"Generated {i} hyperparameters")
    if i > 100:
        raise ValueError("calm down.")
    return configurations
    
                                


class GridSearch:
    # Expects a list of dictionary containing an hyperparameters configuration:
    # [{
    #    loss_function: str,
    #    L2: float,
    #    lr: float,
    #    momentum: float,
    #    BATCH_SIZE: int
    # }]
    def __init__(self, hyperparameters, in_dimension, out_dimension):
        self.hyperparameters = hyperparameters
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

    # Returns the best set of hyperparameters based on the error on the Validation set
    def search(self, train_X, train_Y, val_X, val_Y, MAX_UNLUCKY_STEPS = 50, MAX_EPOCHS = 500, seed = 1):
        val_errors = [0 for i in range(len(self.hyperparameters))] # List of (val_error, n_epochs)
        # TODO: Do this in parallel
        k = 3
        for i, conf in enumerate(self.hyperparameters):
            set_seed(seed + i)
            model = Sequential()
            model.from_configuration(conf, self.in_dimension, self.out_dimension)
            if conf['loss_function'] == 'MSE':
                loss_func = MSE(conf['L2'])
            elif conf['loss_function'] == 'Cross Entropy':
                loss_func = CrossEntropy(conf['L2'])
            else:
                raise ValueError(f'Loss function {conf["loss_function"]} not supported.')
            
            optimizer = GradientDescent(loss_function=loss_func, lr=conf["lr"], momentum=conf["momentum"], BATCH_SIZE=conf["BATCH_SIZE"])
            best_conf = {}
            min_val_error = inf
            for j in range(k):
                (epochs, train_errors, train_accuracies, _val_errors, val_accuracies) = _early_stopping(model, optimizer, train_X, train_Y, val_X, val_Y, MAX_UNLUCKY_STEPS, MAX_EPOCHS)
                if _val_errors[epochs] < min_val_error:
                    best_conf = {'epochs': epochs, 'train_errors': train_errors, 'train_accuracies': train_accuracies, 'val_errors': _val_errors, 'val_accuracies': val_accuracies}
                    min_val_error = _val_errors[epochs]
                set_seed(seed + i + j)
                model = model.reset()
            print(f'{i} / {len(self.hyperparameters)}')
            # TODO: Avoid saving training, val errors
            val_errors[i] = (best_conf['epochs'], best_conf['train_accuracies'][best_conf['epochs'] - 1], best_conf["train_errors"], best_conf["val_errors"], best_conf["train_accuracies"], best_conf["val_accuracies"])
        
        # Find the lowest val error and return the associated hyperparameters conf
        id_best = 0
        lowest_val_errors = val_errors[0][1]
        for i, x in enumerate(val_errors):
            if x[1] < lowest_val_errors:
                lowest_val_errors = x[1]
                id_best = i
        
        best = self.hyperparameters[id_best]
        best["epochs"] = val_errors[id_best][0]
        return best, val_errors[id_best][2], val_errors[id_best][3], val_errors[id_best][4], val_errors[id_best][5]