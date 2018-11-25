""" These dictionarys set most of the relevant settings and hyperparameters for the program. """

MCTS = {
    'c_puct': 1.0,
    'dir_alpha': 0.5,
    'dir_epsilon': 0.25
}

OPTIMISATION = {
    'num_batches': 1,
    'batch_size': 512,
    'num_epochs': 10
}

EVALUATION = {
    'num_games': 50,
    'num_simulations': 25,
    'temperature': 0,
}

SELF_PLAY = {
    'num_games': 50,
    'num_simulations': 25,
    'temperature': 1
}

NEURAL_NETWORKS = {
    'learning_rate': 0.02,
    'regularization_strength': 0.001,
    'num_filters_policy': 256,
    'num_filters_value': 256,
    'num_filters_tower': 256,
    'num_residual_blocks': 5,
    'kernel_size_tower': 3,
    'hidden_dim_value': 256
}
