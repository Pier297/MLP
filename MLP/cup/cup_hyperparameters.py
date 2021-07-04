adam_hyperparameters = {
    'loss_function_name'     : "MSE",
    'optimizer'              : "adam",
    'in_dimension'           : 10,
    'out_dimension'          : 2,
    'validation_percentage'  : 0.20, # percentage of data into validation, remaining into training
    'mini_batch_percentage'  : 0.3,
    'max_unlucky_epochs'     : 100,
    'max_epochs'             : 100,
    'number_trials'          : 1,
    'n_random_search'        : 2,
    'validation_type':       {'method': 'holdout'},
    'target_domain'          : None,
    'lr'                     : [5e-2],
    'lr_decay'               : None,
    'l2'                     : [0],
    'momentum'               : [0],
    'adam_decay_rate_1'      : [0.9],
    'adam_decay_rate_2'      : [0.999],
    'hidden_layers'          : [([('tanh',8), ('tanh', 8)],'linear')
                                ([('tanh',8), ('tanh', 7)],'linear')
                                ([('tanh',8), ('tanh', 6)],'linear')
                                ([('tanh',8), ('tanh', 5)],'linear')
                                ],
    'weights_init'           : [{'method': 'fanin'}],
    'print_stats'            : False,
    'additional_metric'      : 'MEE',
}

sgd_hyperparameters = {
    'loss_function_name'     : "MSE",
    'optimizer'              : "NAG",
    'in_dimension'           : 10,
    'out_dimension'          : 2,
    'validation_percentage'  : 0.20, # percentage of data into validation, remaining into training
    'mini_batch_percentage'  : 1,
    'max_unlucky_epochs'     : 100,
    'max_epochs'             : 100,
    'number_trials'          : 1,
    'n_random_search'        : 2,
    'validation_type':       {'method': 'holdout'},
    'target_domain'          : None,
    'lr'                     : [5e-3],
    'lr_decay'               : None,
    'l2'                     : [0],
    'momentum'               : [0.8],
    'hidden_layers'          : [([('tanh',8), ('tanh', 8)],'linear')
                                ([('tanh',8), ('tanh', 7)],'linear')
                                ([('tanh',8), ('tanh', 6)],'linear')
                                ([('tanh',8), ('tanh', 5)],'linear')
                                ],
    'weights_init'           : [{'method': 'fanin'}],
    'print_stats'            : False,
    'additional_metric'      : 'MEE',
}
