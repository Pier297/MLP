
# Given for each hyperparameter a list of possible values Returns a list of hyperparameters configurations, where each conf is a dictionary
def generate_hyperparameters(loss_func_values, lr_values, l2_values, momentum_values, hidden_layers_values, hidden_layers_activations, batch_percentage):
    i = 0
    configurations = []
    for loss_func in loss_func_values:
        for lr in lr_values:
            for l2 in l2_values:
                for momentum in momentum_values:
                    for hidden_layers_activations_values in hidden_layers_activations:
                        for hidden_layers in hidden_layers_values:
                            for batch_percentage_values in batch_percentage:
                                i += 1
                                yield {"loss_function": loss_func, "lr": lr, "l2": l2, "momentum": momentum, "hidden_layers": hidden_layers, "batch_percentage": batch_percentage_values, "hidden_layers_activations": hidden_layers_activations_values}
    print(f"Generated {i} hyperparameters")
    if i > 200:
        raise ValueError("Error: too many hyperparameters configurations created.")