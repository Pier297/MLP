monk_config = {
    'loss_function_name'     : "Cross Entropy",
    'optimizer'              : "NAG",
    'in_dimension'           : 17,
    'out_dimension'          : 1,
    'target_domain'          : (0, 1),
    'validation_type'        : {'method': 'kfold', 'k': 5},
    'max_unlucky_epochs'     : 100,
    'max_epochs'             : 300,
    'number_trials'          : 3,
    'print_stats'            : False,
    'weights_init'           : [{'method': 'normalized'}],
    'additional_metric'      : None
}

monk1_hyperparameters = {**monk_config,
    'mini_batch_percentage'  : 1.0,
    'lr'                     : [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    'lr_decay'               : [None],
    'l2'                     : [0.0],
    'momentum'               : [0.0, 0.2, 0.4,0.6, 0.8, 0.9],
    'hidden_layers'          : [([('tanh',3)],'sigmoid'),([('tanh',4)],'sigmoid'),([('tanh',5)],'sigmoid'),],
}

monk2_hyperparameters = {**monk_config,
    'mini_batch_percentage'  : 1.0,
    'lr'                     : [0.1, 0.2, 0.4,0.6, 0.8, 1.0],
    'lr_decay'               : [None],
    'l2'                     : [0.0],
    'momentum'               : [0.0, 0.2, 0.4, 0.6,0.8, 0.9],
    'hidden_layers'          : [([('tanh',2)],'sigmoid'),([('tanh',4)],'sigmoid'),([('tanh',5)],'sigmoid'),],
}

monk3_hyperparameters = {**monk_config,
    'mini_batch_percentage'  : 1.0,
    'lr'                     : [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    'lr_decay'               : [None],
    'l2'                     : [0.0],
    'momentum'               : [0.0,0.2, 0.4, 0.6, 0.8, 0.9],
    'hidden_layers'          : [([('tanh',6)],'sigmoid'), ([('tanh',9)],'sigmoid'), ([('tanh',12)],'sigmoid'),
                                ([('relu',6)],'sigmoid'), ([('relu',9)],'sigmoid'), ([('relu',12)],'sigmoid')],
}

monk3_hyperparameters_reg = {**monk_config,
    'mini_batch_percentage'  : 1.0,
    'lr'                     : [0.1, 0.2, 0.4,0.6, 0.8, 1.0],
    'lr_decay'               : [None],
    'l2'                     : [1e-3, 1e-4],
    'momentum'               : [0.0, 0.2, 0.4, 0.6,0.8, 0.9],
    'hidden_layers'          : [([('tanh',6)],'sigmoid'), ([('tanh',9)],'sigmoid'), ([('tanh',12)],'sigmoid')],
}
