adam_hyperparameters = {
    'loss_function_name'     : "MSE",
    'optimizer'              : "adam", #optimizer : "SGD",
    'in_dimension'           : 10,
    'out_dimension'          : 2,
    'validation_percentage'  : 0.20, # percentage of data into validation, remaining into training
    'mini_batch_percentage'  : [0.3],
    'max_unlucky_epochs'     : 200,
    'max_epochs'             : 1500,
    'number_trials'          : 1,
    'n_random_search'        : 36,
    'validation_type'        : {'method': 'kfold', 'k': 5},
    #'validation_type':       {'method': 'holdout'},
    'target_domain'          : None,
    'lr'                     : [7e-3, 4e-3, 2e-3, 1e-3, 5e-4],
    'lr_decay'               : None,
    'l2'                     : [0, 5e-4, 1e-4, 5e-5, 1e-5],
    'momentum'               : [0],
    'adam_decay_rate_1'      : [0.8, 0.85, 0.9],
    'adam_decay_rate_2'      : [0.9, 0.95, 0.999],
    'hidden_layers'          : [([('tanh',32), ('tanh', 32)],'linear'),
                                ([('tanh',16), ('tanh', 16)],'linear'),
                                ([('relu',32), ('relu', 32)],'linear'),],
    'weights_init'           : [{'method': 'fanin'}, {'method': 'linear', 'range': 0.7}],
    'print_stats'            : False,
    'additional_metric' : 'MEE'
}

sgd_hyperparameters = {
    'loss_function_name'     : "MSE",
    'optimizer'              : "NAG",
    'in_dimension'           : 10,
    'out_dimension'          : 2,
    'validation_percentage'  : 0.20, # percentage of data into validation, remaining into training
    'mini_batch_percentage'  : 1,
    'max_unlucky_epochs'     : 100,
    'max_epochs'             : 1200,
    'number_trials'          : 1,
    'n_random_search'        : 36,
    'validation_type'        : {'method': 'kfold', 'k': 5},
    #'validation_type':       {'method': 'holdout'},
    'target_domain'          : None,
    'lr'                     : [7e-3, 5e-3, 4e-3, 3e-3, 2e-3, 5e-4], # 0.6
    'lr_decay'               : None,
    'l2'                     : [0, 5e-5, 1e-5, 1e-4],
    'momentum'               : [0, 0.1, 0.2, 0.3, 0.4],
    'hidden_layers'          : [([('tanh',16), ('tanh', 16)],'linear'),
                                ([('tanh',32), ('tanh', 32)],'linear'),
                                ([('relu',32), ('relu', 32)],'linear'),
                                ([('relu',16), ('relu', 16)],'linear'),],
    'weights_init'           : [{'method': 'fanin'}, {'method': 'linear', 'range': 0.7}],
    'print_stats'            : False,
    'additional_metric' : 'MEE'
}


##### OLD ####

sgd_32_units_hyperparameters = {
    'loss_function_name'     : "MSE",
    'optimizer'              : "NAG",
    'in_dimension'           : 10,
    'out_dimension'          : 2,
    'validation_percentage'  : 0.20, # percentage of data into validation, remaining into training
    'mini_batch_percentage'  : 1,
    'max_unlucky_epochs'     : 100,
    'max_epochs'             : 1200,
    'number_trials'          : 1,
    'n_random_search'        : 12,
    #'validation_type'        : {'method': 'kfold', 'k': 5},
    'validation_type':       {'method': 'holdout'},
    'target_domain'          : None,
    'lr'                     : [4e-3], # 0.6
    'lr_decay'               : None,
    'l2'                     : [5e-5, 1e-5],
    'momentum'               : [0.1, 0.2, 0.3, 0.4],
    'hidden_layers'          : [([('tanh',32), ('tanh', 32)],'linear'),],
    'weights_init'           : [{'method': 'fanin'}, {'method': 'linear', 'range': 0.7}],
    'print_stats'            : False,
    'additional_metric' : 'MEE'
}




sgd_16_units_hyperparameters = {
    'loss_function_name'     : "MSE",
    'optimizer'              : "NAG",
    'in_dimension'           : 10,
    'out_dimension'          : 2,
    'validation_percentage'  : 0.20, # percentage of data into validation, remaining into training
    'mini_batch_percentage'  : 1,
    'max_unlucky_epochs'     : 100,
    'max_epochs'             : 1000,
    'number_trials'          : 3,
    'n_random_search'        : 24,
    #'validation_type'        : {'method': 'kfold', 'k': 5},
    'validation_type':       {'method': 'holdout'},
    'target_domain'          : None,
    'lr'                     : [3e-3, 2e-3], # 0.6
    'lr_decay'               : None,
    'l2'                     : [0, 5e-5, 1e-5, 1e-4],
    'momentum'               : [0, 0.1, 0.2, 0.3, 0.4],
    'hidden_layers'          : [([('tanh',16), ('tanh', 16)],'linear'),],
                                #([('tanh',32), ('tanh', 32)],'linear'),],
    'weights_init'           : [{'method': 'fanin'}],
    'print_stats'            : False,
    'additional_metric' : 'MEE'
}
""" adam_hyperparameters = {
    'loss_function_name'     : "MSE",
    'optimizer'              : "adam", #optimizer : "SGD",
    'in_dimension'           : 10,
    'out_dimension'          : 2,
    'validation_percentage'  : 0.20, # percentage of data into validation, remaining into training
    'mini_batch_percentage'  : [0.2, 0.3],
    'max_unlucky_epochs'     : 200,
    'max_epochs'             : 500,
    'number_trials'          : 1,
    'n_random_search'        : 36,
    #'validation_type'        : {'method': 'kfold', 'k': 5},
    'validation_type':       {'method': 'holdout'},
    'target_domain'          : None,
    'lr'                     : [1e-2, 5e-3, 1e-3, 5e-4],
    'lr_decay'               : None,
    'l2'                     : [0, 5e-5, 1e-5, 1e-4],
    'momentum'               : [0],
    'adam_decay_rate_1'      : [0.8],#[0.62742, 0.8175, 0.97960],
    'adam_decay_rate_2'      : [0.888],#[0.61609, 0.7723, 0.84638],
    'hidden_layers'          : [([('tanh',16), ('tanh', 16)],'linear'),
                                ([('tanh',32), ('tanh', 32)],'linear'),],
    'weights_init'           : [{'method': 'fanin'}],
    'print_stats'            : False,
    'additional_metric' : 'MEE'
}

sgd_hyperparameters = {
    'loss_function_name'     : "MSE",
    'optimizer'              : "NAG",
    'in_dimension'           : 10,
    'out_dimension'          : 2,
    'validation_percentage'  : 0.20, # percentage of data into validation, remaining into training
    'mini_batch_percentage'  : [0.3, 1],
    'max_unlucky_epochs'     : 200,
    'max_epochs'             : 500,
    'number_trials'          : 1,
    'n_random_search'        : 36,
    #'validation_type'        : {'method': 'kfold', 'k': 5},
    'validation_type':       {'method': 'holdout'},
    'target_domain'          : None,
    'lr'                     : [5e-3, 1e-3, 5e-4], # 0.6
    'lr_decay'               : None,
    'l2'                     : [0, 5e-5, 1e-5, 1e-4],
    'momentum'               : [0, 0.1, 0.3, 0.6, 0.9],
    'hidden_layers'          : [([('tanh',16), ('tanh', 16)],'linear'),
                                ([('tanh',32), ('tanh', 32)],'linear'),],
    'weights_init'           : [{'method': 'fanin'}],
    'print_stats'            : False,
    'additional_metric' : 'MEE'
} """







""" adam_hyperparameters = {
    'loss_function_name'     : "MSE",
    'optimizer'              : "adam", #optimizer : "SGD",
    'in_dimension'           : 10,
    'out_dimension'          : 2,
    'validation_percentage'  : 0.20, # percentage of data into validation, remaining into training
    'mini_batch_percentage'  : 0.3,
    'max_unlucky_epochs'     : 200,
    'max_epochs'             : 1000,
    'number_trials'          : 1,
    'n_random_search'        : 12,
    #'validation_type'        : {'method': 'kfold', 'k': 5},
    'validation_type':       {'method': 'holdout'},
    'target_domain'          : None,
    'lr'                     : [5e-2, 1e-3, 5e-4, 1e-4], # 0.6
    'lr_decay'               : None,#[(0.01*1e-1, 200)], #[(0.0, 50)],
    'l2'                     : [0],
    'momentum'               : [0],
    'adam_decay_rate_1'      : [0.9],
    'adam_decay_rate_2'      : [0.999],
    'hidden_layers'          : [([('tanh',32), ('tanh', 32)],'linear'),
                                ([('tanh',64), ('tanh', 32), ('tanh', 16)],'linear'),
                                ([('tanh',64), ('tanh', 64)],'linear')],#([('relu',128), ('relu', 128),('relu',64), ('relu', 64), ('relu', 32), ('relu', 16)],'linear'),
    'weights_init'           : [{'method': 'fanin'}],
    'print_stats'            : False
}

sgd_hyperparameters = {
    'loss_function_name'     : "MSE",
    'optimizer'              : "NAG",
    'in_dimension'           : 10,
    'out_dimension'          : 2,
    'validation_percentage'  : 0.20, # percentage of data into validation, remaining into training
    'mini_batch_percentage'  : 1,
    'max_unlucky_epochs'     : 250,
    'max_epochs'             : 1000,
    'number_trials'          : 1,
    'n_random_search'        : 12,
    #'validation_type'        : {'method': 'kfold', 'k': 5},
    'validation_type':       {'method': 'holdout'},
    'target_domain'          : None,
    'lr'                     : [5e-3, 1e-3, 9e-4], # 0.6
    'lr_decay'               : None,#[(0.01*1e-1, 200)], #[(0.0, 50)],
    'l2'                     : [0],
    'momentum'               : [0.8, 0.9],
    'hidden_layers'          : [([('tanh',32), ('tanh', 32)],'linear'),
                                ([('tanh',32), ('tanh', 32), ('tanh', 32)],'linear'),
                                ([('tanh',64), ('tanh', 64)],'linear')],#([('relu',128), ('relu', 128), ('relu',64), ('relu', 64), ('relu', 32), ('relu', 16)],'linear'),
    'weights_init'           : [{'method': 'fanin'}],
    'print_stats'            : False
} """

""" adam_hyperparameters = {
    'loss_function_name'     : "MSE",
    'optimizer'              : "adam", #optimizer : "SGD",
    'in_dimension'           : 10,
    'out_dimension'          : 2,
    'validation_percentage'  : 0.20, # percentage of data into validation, remaining into training
    'mini_batch_percentage'  : [0.3, 0.1, 0.5, 1],
    'max_unlucky_epochs'     : 100,
    'max_epochs'             : 500,
    'number_trials'          : 3,
    'validation_type'        : {'method': 'kfold', 'k': 5}, # validation_type:{'method': 'holdout'},
    'target_domain'          : None,
    'lr'                     : [0.0001, 0.0005, 0.0075, 0.05, 0.1], # 0.6
    'lr_decay'               : None,#[(0.01*1e-1, 200)], #[(0.0, 50)],
    'l2'                     : [0, 1e-6, 1e-5],
    'momentum'               : [0],
    'adam_decay_rate_1'      : [0.9, 0.8, 0.7, 0.6],
    'adam_decay_rate_2'      : [0.999, 0.888, 0.777, 0.666],
    'hidden_layers'          : [([('relu',32), ('relu', 32), ('relu', 32)],'linear'),],# ([('relu',16), ('relu', 16)],'linear'),([('tanh',16), ('relu', 16)],'linear'), ([('leaky-relu',16), ('relu', 16)],'linear')],
    'weights_init'           : [{'method': 'fanin'}],
    'print_stats'            : False
} """
