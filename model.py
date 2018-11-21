import numpy as np

import logs
import config
import neural_networks


class AZModel:
    """ TODO docstring model """

    def __init__(self, memory, input_shape, num_possible_moves, model_id=None, load=False):
        """ TODO docstring __init__ """
        self.id = model_id # might be overwritten during loading
        self.num_possible_moves = num_possible_moves
        self.input_shape = input_shape

        self.model_counter = 0

        self.memory = memory
        self.neural_network = None

        if not load:
            self.neural_network = neural_networks.NeuralNetwork(
                self.input_shape,
                self.num_possible_moves,
                id=self.id
            )

        self.X = None
        self.y_outcomes = None
        self.y_probabilities = None

        # initiate logger
        #self.logger = logs.get_logger()

    def evaluate(self, state):
        """ TODO docstring evaluate """
        state = state[np.newaxis, :, :, :]

        value, policy_logits = self.neural_network.predict(state)

        value = value[0, 0]
        policy_logits = policy_logits[0, :]
        policy = softmax(policy_logits)

        return value, policy

    def save(self, version_id):
        """ TODO docstring save_model """
        model_json = self.neural_network.to_json()
        path = "saved/neural_networks/model_{}" .format(version_id)

        with open(path + ".json", "w") as file:
            file.write(model_json)

        print("Saved model ", version_id)

        self.neural_network.save_weights(path + ".h5")

    def load(self, version):
        """ TODO docstring load_model """
        # self.id = version_id.split('_')[1]
        self.model_counter = version
        version = str(version)
        path = "saved/neural_networks/model_{}_{}" .format(self.id, version)

        print("Loaded model {}" .format(self.id +"_"+version))
        self.neural_network = neural_networks.NeuralNetwork(
            self.input_shape,
            self.num_possible_moves,
            load_path=path + ".json",
            id=self.id
        )

    def train(self):
        """ TODO docstring train """
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
        """ TODO docstring get_model_counter """
        return self.model_counter

    def get_training_data(self):
        """ TODO docstring get_data """
        self.X, self.y_outcomes, self.y_probabilities = self.memory.get_training_data()
        print("model_X shape: {}".format(self.X.shape))
        print("model_y_outcomes: {}".format(self.y_outcomes.shape))
        print("model_y_probabilities: {}".format(self.y_probabilities.shape))


def softmax(x):
    """ Compute softmax values for each sets of scores in x. """
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)