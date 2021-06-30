monk_config = {
    'loss_function_name'     : "Cross Entropy",
    'optimizer'              : "SGD",
    'in_dimension'           : 17,
    'out_dimension'          : 1,
    'target_domain'          : (0, 1), # (cross entropy)
    'validation_percentage': 0.2,
    'validation_type'        : {'method': 'kfold', 'k': 4},
    #'validation_type'        : {'method': 'holdout'},
    'max_unlucky_epochs'     : 400,
    'max_epochs'             : 200,
    'number_trials'          : 5,
    'print_stats'            : False,
    'weights_init'           : [{'method': 'fanin'}]
}

monk1_hyperparameters = {**monk_config,
    'mini_batch_percentage'  : 1.0,
    'lr'                     : [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6],# 1.0, 1.1, 1.2, 1.3],#, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], # 0.6
    'lr_decay'               : [None], #[(0.0, 50)],
    'l2'                     : [0.0],
    'momentum'               : [0.6],
    'hidden_layers'          : [([('tanh',5)],'sigmoid'),
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