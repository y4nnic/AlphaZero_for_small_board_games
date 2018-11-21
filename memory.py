import numpy as np
import os
import config

import logs


class Position:
    """ Objects of this class contain all relevant information corresponding to
    a board position that was generated during self-play.

    Attributes:
        state: The representation of the game state. This is the input for the
            neural network.
        legal_actions: A binary vector describing which actions are legal in this
            position. The action with index i is legal, if legal_actions[i] == 1.
        player: The player, whos turn it is in this board position. If 0, it is player
            one's (started the game) turn, if 1, it is the second player's turn.
        outcome: The outcome of the self-play game corresponding to this position:
             1: The current player won the game.
            -1: The opponent of the current player won the game.
        probabilities: The MCTS-policy corresponding to this position.
    """

    def __init__(self, state, legal_actions):
        """ Constructor. """
        self.state = state
        self.legal_actions = legal_actions

        self.player = state[0,0,2]
        self.outcome = 0
        self.probabilities = np.zeros(legal_actions.shape)

    def set_outcome(self, outcome):
        """ Sets the outcome. """
        self.outcome = outcome

    def set_probabilities(self, distribution):
        """ Sets the policy. """
        self.probabilities = distribution


class PositionMemory:
    """ The PositionMemory saves recent board positions occuring during self-play. The oldest
    positions are deleted, if the size of the memory surpasses a fixed maximum size.
    Rotations and symmetries are handled here as well. Additionaly, the PositionMemory
    provides the training data set for the model (according to the hyperparamters in config.yp)

    Attributes:
         states: A np.array containing all states (and their symmetries/rotations).
         outcomes: A np.array containing the outcomes corresponding to the states.
            (outcomes[i] is the policy corresponding to states[i]).
         probabilities A np.array containing the MCTS-policies corresponding to the states
            (probabilities[i] is the policy corresponding to states[i]).
         size:

    """

    def __init__(self, variant):
        """ Constructor. """
        self.states = None  # []
        self.outcomes = None  # []
        self.probabilities = None  # []
        self.size = 0
        self.variant = variant
        #self.logger = logs.get_logger()

    def add(self, position):
        """ This method adds a position to the memory. For an empty memory the np.arrays
        are initialized with the given position. Afterwards, rotations/symmetries are added
        depending on the game. If the size surpasses the maximum threshold, the first 100
        positions are discarded.

        Args:
            position: A Position object (as described above) that should be added to
                the memory.
        """
        if self.states is None:
            self.size += 1
            self.states = np.array(position.state, dtype=np.uint8)
            self.states = self.states[np.newaxis,:,:,:]
            self.outcomes = np.array(position.outcome)
            self.probabilities = np.array(position.probabilities)
            self.probabilities = self.probabilities[np.newaxis,:]
        else:
            self.size += 1
            self.states= np.append(self.states, position.state[np.newaxis, :, :, :], axis=0)
            self.outcomes = np.append(self.outcomes, position.outcome)
            self.probabilities = np.append(self.probabilities, position.probabilities[np.newaxis, :], axis=0)

        if self.variant == "TicTacToe":
            self.add_rotations_tic_tac_toe(position)
        if self.variant == "Connect4":
            self.add_mirror_connect_four(position)
        
        if self.size > 10000:
            self.states = self.states[100:]
            self.outcomes = self.outcomes[100:]
            self.probabilities = self.probabilities[100:]
            self.size -= 100
    
    def add_rotations_tic_tac_toe(self, position):
        """ This method adds three rotations (90, 180, 270 degree) of the state representation, together with
        adjusted policy vectors. """
        self.size += 3
        rotation_1 = np.rot90(position.state[np.newaxis,:,:], 1, (1,2))
        prob_1 = np.zeros(9)
        prob_1[0] = position.probabilities[2]
        prob_1[1] = position.probabilities[5]
        prob_1[2] = position.probabilities[8]
        prob_1[3] = position.probabilities[1]
        prob_1[4] = position.probabilities[4]
        prob_1[5] = position.probabilities[7]
        prob_1[6] = position.probabilities[0]
        prob_1[7] = position.probabilities[3]
        prob_1[8] = position.probabilities[6]
        
        self.states = np.append(self.states, rotation_1 , axis=0)
        self.outcomes = np.append(self.outcomes, position.outcome)
        self.probabilities = np.append(self.probabilities, prob_1[np.newaxis,:], axis=0)
        
        rotation_2 = np.rot90(position.state[np.newaxis,:,:], 2, (1,2))
        prob_2 = np.zeros(9)
        prob_2[0] = position.probabilities[8]
        prob_2[1] = position.probabilities[7]
        prob_2[2] = position.probabilities[6]
        prob_2[3] = position.probabilities[5]
        prob_2[4] = position.probabilities[4]
        prob_2[5] = position.probabilities[3]
        prob_2[6] = position.probabilities[2]
        prob_2[7] = position.probabilities[1]
        prob_2[8] = position.probabilities[0]
        
        self.states = np.append(self.states, rotation_2 , axis=0)
        self.outcomes = np.append(self.outcomes, position.outcome)
        self.probabilities = np.append(self.probabilities, prob_2[np.newaxis,:], axis=0)
        
        rotation_3 = np.rot90(position.state[np.newaxis,:,:], 3, (1,2))
        prob_3 = np.zeros(9)
        prob_3[0] = position.probabilities[6]
        prob_3[1] = position.probabilities[3]
        prob_3[2] = position.probabilities[0]
        prob_3[3] = position.probabilities[7]
        prob_3[4] = position.probabilities[4]
        prob_3[5] = position.probabilities[1]
        prob_3[6] = position.probabilities[8]
        prob_3[7] = position.probabilities[5]
        prob_3[8] = position.probabilities[2]
        
        self.states = np.append(self.states, rotation_3 , axis=0)
        self.outcomes = np.append(self.outcomes, position.outcome)
        self.probabilities = np.append(self.probabilities, prob_3[np.newaxis,:], axis=0)

    def add_mirror_connect_four(self, position):
        """ This method adds a mirror image of the position's state representation
        with a policy vector, that is adjusted accordingly. """
        self.size += 1
        mirror = position.state[np.newaxis,:,:]
        mirror = mirror[:,:,np.arange(6,-1,-1)]
        prob = position.probabilities[np.arange(6,-1,-1)]
        self.states = np.append(self.states, mirror, axis=0)
        self.outcomes = np.append(self.outcomes, position.outcome)
        self.probabilities = np.append(self.probabilities, prob[np.newaxis,:], axis=0)

    def save(self, id):
        """ This method saves the current memory in a new folder
        (path = saved/memory/<id>)."""

        path = "saved/memory/{}/" .format(id)

        if not os.path.exists(path):
            os.makedirs(path)

        size = np.array([self.size])
        np.save(
            file=path + "size.npy",
            arr=size
        )
        np.save(
            file=path + "saved_X.npy",
            arr=self.states
        )
        np.save(
            file=path + "saved_y_out.npy",
            arr=self.outcomes
        )
        np.save(
            file=path + "saved_y_prob.npy",
            arr=self.probabilities
        )

    def load(self, id):
        """ This method loads a memory that was saved before under
        saved/memory/<id>. """
        path = "saved/memory/{}/" .format(id)
        size = np.load(
            file=path + "size.npy"
        )
        self.size = size[0]
        self.states = np.load(
            file=path + "saved_X.npy"
        )
        self.outcomes = np.load(
            file=path + "saved_y_out.npy"
        )
        self.probabilities = np.load(
            file=path + "saved_y_prob.npy"
        )

    def get_training_data(self):
        """ The needed number of positions is computed based on the parameters in
        config.py. This number of positions is then randomly chosen from the PositionMemory.

        Returns:
            X:  A np.array containing the state representations for the training.
            y_outcomes: A np.array containing the targets for the value head.
            y_probabilites: A np.array containing the targets for the policy head.
        """
        num_batches = config.OPTIMISATION['num_batches']
        batch_size = config.OPTIMISATION['batch_size']
        num_positions = num_batches * batch_size

        print("num_positions: {}".format(num_positions))

        indices = np.arange(0,self.size,1)

        if self.size > num_positions:
            indices = np.random.choice(indices, num_positions)

        X = self.states[indices]
        y_outcomes = self.outcomes[indices]
        y_probabilities = self.probabilities[indices]

        return X, y_outcomes, y_probabilities

    def print_positions(self):
        """ Prints all positions in this memory. """
        np.set_printoptions(precision=3)

        if self.variant == "TicTacToe":
            for i in range(self.size):
                print("---------------------------------")
                print("Position {}" .format(i))
                player = self.states[i,0,0,2]
                if player == 0: player = 2
                print("Current player: {}" .format(player))
                if player == 2:
                    board = self.states[i, :, :, 1] + self.states[i, :,:, 0] * 2
                if player == 1:
                    board = self.states[i, :, :, 0] + self.states[i, :, :, 1] * 2
                print("{}" .format(board))
                print("Probabilities: \n {} | {} | {} " .format(
                    self.probabilities[i, 0],
                    self.probabilities[i, 1],
                    self.probabilities[i, 2]
                ))
                print("{} | {} | {}" .format(
                    self.probabilities[i, 3],
                    self.probabilities[i, 4],
                    self.probabilities[i, 5]
                ))
                print("{} | {} | {}" .format(
                    self.probabilities[i, 6],
                    self.probabilities[i, 7],
                    self.probabilities[i, 8]
                ))
                print("Outcome: {}" .format(self.outcomes[i]))

        if self.variant == "Connect4":
            for i in range(self.size):
                print("---------------------------------")
                print("Position {}" .format(i))
                player = self.states[i,0,0,2]
                if player == 0: player = 2
                print("Current player: {}" .format(player))
                if player == 2:
                    board = self.states[i, :, :, 1] + self.states[i, :,:, 0] * 2
                if player == 1:
                    board = self.states[i, :, :, 0] + self.states[i, :, :, 1] * 2
                print("{}" .format(board))
                print("Probabilities: \n {} | {} | {} | {} | {} | {} | {} " .format(
                    self.probabilities[i, 0],
                    self.probabilities[i, 1],
                    self.probabilities[i, 2],
                    self.probabilities[i, 3],
                    self.probabilities[i, 4],
                    self.probabilities[i, 5],
                    self.probabilities[i, 6],
                ))
                print("Outcome: {}" .format(self.outcomes[i]))