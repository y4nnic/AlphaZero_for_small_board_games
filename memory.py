import numpy as np
import os
import config

import logs


class Position:
    """ TODO docstring Position """

    def __init__(self, state, legal_actions):
        """ TODO docstring __init__ """
        self.state = state
        self.legal_actions = legal_actions

        self.player = state[0,0,2]
        self.outcome = 0
        self.probabilities = np.zeros(legal_actions.shape)

    def set_outcome(self, outcome):
        """ TODO docstring set_outcome """
        self.outcome = outcome

    def set_probabilities(self, distribution):
        """ TODO docstring set_probabilities """
        self.probabilities = distribution


class PositionMemory:
    """ TODO docstring Memory """

    def __init__(self):
        """ TODO docstring __init__ """
        self.states = None  # []
        self.outcomes = None  # []
        self.probabilities = None  # []
        self.size = 0
        #self.logger = logs.get_logger()

    def add(self, position):
        """ TODO docstring add """
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
        
        self.add_rotations(position)
        
        if self.size > 10000:
            self.states = self.states[100:]
            self.outcomes = self.outcomes[100:]
            self.probabilities = self.probabilities[100:]
            self.size -= 100
    
    def add_rotations(self, position):
        self.size += 4
        rotation_1 = np.rot90(position[np.newaxis,:,:], k=1, axis=(1,2))
        prob_1 = np.zeros(8)
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
        
        rotation_2 = np.rot90(position[np.newaxis,:,:], k=2, axis=(1,2))
        prob_2 = np.zeros(8)
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
        
        rotation_3 = np.rot90(position[np.newaxis,:,:], k=3, axis=(1,2))
        prob_3 = np.zeros(8)
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
                                                                                 
    def save(self, id):
        """ TODO docstring save_memory """
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
        """ TODO docstring load_memory """
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
        """ TODO docstring get_data """
        num_batches = config.OPTIMISATION['num_batches']
        batch_size = config.OPTIMISATION['batch_size']
        num_positions = num_batches * batch_size

        print("num_positions: {}".format(num_positions))
        #states = np.array(self.states)
        #outcomes = np.array(self.outcomes)
        #probabilities = np.array(self.probabilities)

        # memory_size = self.get_current_size()
        indices = np.arange(0,self.size,1)

        if self.size > num_positions:
            indices = np.random.choice(indices, num_positions)

        X = self.states[indices]
        y_outcomes = self.outcomes[indices]
        y_probabilities = self.probabilities[indices]

        return X, y_outcomes, y_probabilities

    def print(self):
        """ Prints all positions in this memory. """
        np.set_printoptions(precision=3)
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