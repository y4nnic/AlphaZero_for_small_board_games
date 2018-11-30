import numpy as np

#import logs
import config
import neural_networks


class AZModel:
    """
    An AZModel is used for the optimization phase and the evaluations for the MCTS. It contains
    the neural network and saves its versions after each training phase.

    Attributes:
        memory: An instance of PositionMemory that provides the training data.
        input_shape: The input shape for the neural network: (board_height, board_width, 3)
        num_possible_moves: Maximum number of possible moves for one state of the
            considered game.
        model_id: Unique name or number (string) for this model.
        load: If load is False, a neural network is initialized. Otherwise
            it needs to be loaded afterwords with the load() method.
        lr: The initial learning rate for the training process. The config.py file provides
            the value if lr == None.
        reg: The regularization strength for the training process. The config.py file provides
            the value if lr == None.
    """

    def __init__(self, memory, input_shape, num_possible_moves, model_id=None, load=False, lr=None, reg=None):
        """ Initializes the main components.

        Args:
            memory: An instance of PositionMemory that provides the training data.
            input_shape: The input shape for the neural network: (board_height, board_width, 3)
            num_possible_moves: Maximum number of possible moves for one state of the
                considered game.
            model_id: Unique name or number (string) for this model.
            load: If load is False, a neural network is initialized. Otherwise
                it needs to be loaded afterwords with the load() method.
            lr: The initial learning rate for the training process. The config.py file provides
                the value if lr == None.
            reg: The regularization strength for the training process. The config.py file provides
                the value if lr == None.
        """
        self.id = model_id
        self.num_possible_moves = num_possible_moves
        self.input_shape = input_shape

        self.lr = lr
        self.reg = reg

        self.model_counter = 0

        self.memory = memory
        self.neural_network = None

        if not load:
            self.neural_network = neural_networks.NeuralNetwork(
                self.input_shape,
                self.num_possible_moves,
                net_id=self.id,
                lr=self.lr,
                reg=self.reg
            )

        self.X = None
        self.y_outcomes = None
        self.y_probabilities = None

        # initiate logger
        #self.logger = logs.get_logger()

    def evaluate(self, state):
        """ The model evaluates a given state and returns the neural network's
         value estimate and policy estimate. """
        state = state[np.newaxis, :, :, :]

        value, policy_logits = self.neural_network.predict(state)

        value = value[0, 0]
        policy_logits = policy_logits[0, :]
        policy = softmax(policy_logits)

        return value, policy

    def save(self, version_id):
        """ Saves the current version of the model (.json + .h5) in the
         directory "saved/neural_networks/".

        Args:
             version_id: Id of the current version (file: "model_<version_id>").
                -> <model_id>_<iteration>
        """
        model_json = self.neural_network.to_json()
        path = "saved/neural_networks/model_{}" .format(version_id)

        with open(path + ".json", "w") as file:
            file.write(model_json)

        print("Saved model ", version_id)

        self.neural_network.save_weights(path + ".h5")

    def load(self, version):
        """ Loads the given version of the model from the directory
        "saved/neural_networks/".

        Args:
            version: Iteration of the version of the model that is loaded.
        """
        self.model_counter = version
        version = str(version)
        path = "saved/neural_networks/model_{}_{}" .format(self.id, version)

        print("Loaded model {}" .format(self.id +"_"+version))
        self.neural_network = neural_networks.NeuralNetwork(
            self.input_shape,
            self.num_possible_moves,
            load_path=path + ".json",
            net_id=self.id,
            lr=self.lr,
            reg=self.reg
        )

    def train(self):
        """ Gathers the training data, trains the neural network and saves the model.
        The hyperparameters are set in config.py.
        """
        batch_size = config.OPTIMISATION['batch_size']
        num_epochs = config.OPTIMISATION['num_epochs']

        self.get_training_data()

        self.neural_network.fit(
            X=self.X,
            y={
                'value': self.y_outcomes,
                'policy': self.y_probabilities
            },
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=1
        )

        version_id = str(self.id) + "_" + str(self.model_counter)
        self.save(version_id)
        self.model_counter += 1

    def get_model_count(self):
        """ Returns the current version number of the model. """
        return self.model_counter

    def get_training_data(self):
        """ Receives the training data from the position memory and provides
        them for the neural network."""
        self.X, self.y_outcomes, self.y_probabilities = self.memory.get_training_data()
        print("model_X shape: {}".format(self.X.shape))
        print("model_y_outcomes: {}".format(self.y_outcomes.shape))
        print("model_y_probabilities: {}".format(self.y_probabilities.shape))


def softmax(x):
    """ Compute softmax values for each sets of scores in x. """
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)