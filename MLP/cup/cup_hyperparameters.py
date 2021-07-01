adam_hyperparameters = {
    'loss_function_name'     : "MSE",
    'optimizer'              : "adam", #optimizer : "SGD",
    'in_dimension'           : 10,
    'out_dimension'          : 2,
    'validation_percentage'  : 0.20, # percentage of data into validation, remaining into training
    'mini_batch_percentage'  : 0.3,
    'max_unlucky_epochs'     : 100,
    'max_epochs'             : 500,
    'number_trials'          : 1,
    'validation_type'        : {'method': 'kfold', 'k': 5}, # validation_type:{'method': 'holdout'},
    'target_domain'          : None,
    'lr'                     : [0.0001, 0.0005, 0.0075, 0.05, 0.1], # 0.6
    'lr_decay'               : None,#[(0.01*1e-1, 200)], #[(0.0, 50)],
    'l2'                     : [0, 1e-6],
    'momentum'               : [0],
    'adam_decay_rate_1'      : [0.9],
    'adam_decay_rate_2'      : [0.999],
    'hidden_layers'          : [([('relu',20), ('relu', 15)],'linear')],
    'weights_init'           : [{'method': 'fanin'}],
    'print_stats'            : False
}
