
# Given for each hyperparameter a list of possible values Returns a list of hyperparameters configurations, where each conf is a dictionary
def generate_hyperparameters(loss_func_values, lr_values, l2_values, momentum_values, hidden_layers_values, BATCH_SIZE_values):
    i = 0
    configurations = []
    for loss_func in loss_func_values:
        for lr in lr_values:
            for l2 in l2_values:
                for momentum in momentum_values:
                    for hidden_layers in hidden_layers_values:
                        for BATCH_SIZE in BATCH_SIZE_values:
                            configurations.append({"loss_function": loss_func, "lr": lr, "l2": l2, "momentum": momentum, "hidden_layers": hidden_layers, "BATCH_SIZE": BATCH_SIZE})
                            i += 1
    print(f"Generated {i} hyperparameters")
    if i > 200:
        raise ValueError("Error: too many hyperparameters configurations created.")
    return configurations