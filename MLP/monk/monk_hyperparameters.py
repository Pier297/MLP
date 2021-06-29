monk_config = {
    'loss_function_name'     : "Cross Entropy",
    'optimizer'              : "SGD",
    'in_dimension'           : 17,
    'out_dimension'          : 1,
    'target_domain'          : (0, 1), # (cross entropy)
    'validation_type'        : {'method': 'kfold', 'k': 7},
    #'validation_type'        : {'method': 'holdout', 'validation_percentage': 0.2},
    'max_unlucky_epochs'     : 300,
    'max_epochs'             : 500,
    'number_trials'          : 10,
    'print_stats'            : False
}

monk1_hyperparameters = {**monk_config,
    'mini_batch_percentage'  : 1.0,
    'lr'                     : [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2], # 0.6
    'lr_decay'               : [None], #[(0.0, 50)],
    'l2'                     : [0.0],
    'momentum'               : [0, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8],
    'hidden_layers'          : [([('tanh',4)],'sigmoid'),
                               ],
}

monk2_hyperparameters = {**monk_config,
    'mini_batch_percentage'  : 1.0,
    'lr'                     : [0.05, 0.2, 0.4, 0.6, 0.8], # 0.6
    'lr_decay'               : [None], #[(0.0, 50)],
    'l2'                     : [0.0],
    'momentum'               : [0, 0.2, 0.3, 0.4, 0.6, 0.8],
    'hidden_layers'          : [([('tanh',4)],'sigmoid'),
                               ],
}

monk3_hyperparameters = {**monk_config,
    'mini_batch_percentage'  : 1.0,
    'lr'                     : [0.05, 0.2, 0.4, 0.6, 0.8], # 0.6
    'lr_decay'               : [None], #[(0.0, 50)],
    'l2'                     : [0.0],
    'momentum'               : [0, 0.2, 0.3, 0.4, 0.6, 0.8],
    'hidden_layers'          : [([('tanh',5)],'sigmoid'),
                               ],
}

monk3_hyperparameters_reg = {**monk_config,
    'mini_batch_percentage'  : 1.0,
    'lr'                     : [0.05, 0.2, 0.4, 0.6, 0.8], # 0.6
    'lr_decay'               : [None], #[(0.0, 50)],
    'l2'                     : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'momentum'               : [0, 0.2, 0.3, 0.4, 0.6, 0.8],
    'hidden_layers'          : [([('tanh',5)],'sigmoid'),
                               ],
}