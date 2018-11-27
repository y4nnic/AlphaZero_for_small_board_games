import numpy as np

import mcts
import logs
import config


class Agent:

    def play_move(self, number_simulations=None):
        pass

    def join_game(self, game):
        pass

    
    
class Player():
    def __init__(self, az_agent=None):
        self.game = None
        self.len_action_space = None
        self.az_agent = az_agent

    def join_game(self, game):
        """ Sets the game environment and the length of the action space.

        Args:
             game: See attributes.
        """
        self.game = game

    def play_move(self, number_simulations=None, temperature=None, opponent=None):
        current_position = self.game.get_current_position()
        
        action = input("choose from {} :".format(current_position.legal_actions))
        action = int(action)
        
        self.az_agent.tree_search.update_root(action)
        position = self.game.get_current_position()
        winning = self.game.execute_move(action)
        return winning, position
    
    
class RandomAgent(Agent):
    """ This agent plays random moves. The moves are sampled uniformly and at random
     from all legal actions of the current position.

    Attributes:
        game: The game environment which simulates moves and provides game states.
        len_action_space: The length of the game's action space.
        eval_agent: The opponent of this RandomAgent (only needed during games against
            AlphaZeroAgent).
    """

    def __init__(self, eval_agent=None):
        self.game = None
        self.len_action_space = None
        self.eval_agent = eval_agent

    def join_game(self, game):
        """ Sets the game environment and the length of the action space.

        Args:
             game: See attributes.
        """
        self.game = game
        self.len_action_space = game.get_len_action_space()

    def play_move(self, number_simulations=None, temperature=None, opponent=None):
        """ Samples a move uniformly and at random from all legal actions of the current
        position. If necessary (against AlphaZeroAgent), the root of the opposing agent's
        tree search is updated accordingly.

        Args:
             number_simulations: Irrelevant for this agent.
             temperature: Irrelevant for this agent.
             opponent: Irrelevant for this agent.

        Returns:
            winning: Only 1 if the selected action wins the game.
            position: The game state (memory.Position) before the selected action is executed.
        """
        current_position = self.game.get_current_position()

        distribution = current_position.legal_actions
        distribution = distribution/np.sum(distribution)

        action = np.random.choice(np.arange(self.len_action_space), p=distribution)
        if self.eval_agent is not None:
            self.eval_agent.tree_search.update_root(action)

        position = self.game.get_current_position()
        winning = self.game.execute_move(action)
        return winning, position


class AlphaZeroAgent(Agent):
    """
    Attributes:
        model: The model which will be used for the MCTS.
        game: The game environment which executes moves and provides game states.
        len_action_space: The length of the game's action space.
        tree_search: A MonteCarloTreeSearch object that handles the MCTS.
    """
    def __init__(self, model, variant):
        """ Model is set. The other attributes are only set when a game is joined. """
        self.model = model
        self.game = None
        self.len_action_space = None
        self.tree_search = None

        if variant == "TicTacToe":
            self.max_game_length = 9
        if variant == "Connect4":
            self.max_game_length = 42

        self.trajectories = {}
        # initialize logger
        # self.logger = logs.get_logger()

    def play_move(self, number_simulations=None, temperature=1.0, opponent=None, add_dirichlet=False):
        """ Selects a move based on MCTS. The root of the opponent's tree search is updated accordingly
         (only if the opponent is an AlphaZeroAgent).

         Args:
            number_simulations: Number of simulations before a move is selected.
            temperature: Controls the amount of exploration during the tree search (>= 0).
            opponent: The opposing player. Only necessary, if the opponent is an AlphaZeroAgent.
            add_dirichlet: If True, Dirichlet noise is added to the root's priors during MCTS.

        Returns:
            winning: Only 1 if the selected action wins the game.
            position: The game state (memory.Position) before the selected action is executed.
         """
        for i in range(number_simulations):
            self.tree_search.simulation(add_dirichlet=add_dirichlet)

        action, distribution = self.tree_search.play(temperature=temperature)

        position = self.game.get_current_position()
        position.set_probabilities(distribution)

        if opponent is not None:
            opponent.tree_search.update_root(action)

        winning = self.game.execute_move(action)

        return winning, position, action

    def join_game(self, game):
        """ The game enviroment and the length of the action space are set and the MCTS object
         is initialized accordingly. """
        self.game = game
        self.len_action_space = game.get_len_action_space()

        initial_position = self.game.get_current_position()
        self.tree_search = mcts.MonteCarloTreeSearch(
            model=self.model,
            game=self.game,
            initial_position=initial_position
        )

    def self_play(self, position_memory, game):
        """ The agent plays a game against itself and saves all positions in the memory.
        The outcome of a position:
            1 : Current player won the game.
           -1 : Current player lost the game.

        Args:
            position_memory: This memory.PositionMemory object saves all positions
                (game states, outcomes and MCTS-distributions).
            game: The game environment which provides game states and executes moves.
        """
        # reset game
        game.reset_game()
        self.join_game(game)

        winning = 0
        turn = 0
        positions = []
        trajectory = ""

        while winning is 0 and turn < self.max_game_length:
            winning, position, action = self.play_move(number_simulations=config.SELF_PLAY["num_simulations"], add_dirichlet=True)
            positions.append(position)
            trajectory += str(action)
            turn += 1
            # print("turn {}".format(turn))

        # self.logger.info("Player {} won the game after {} turns.".format(winner, turn))
        #print("Player {} won the game after {} turns.".format(winner, turn))

        if trajectory not in self.trajectories:
            self.trajectories[trajectory] = 1
        else:
            self.trajectories += 1

        player = position.player
        while turn > 0:
            turn -= 1
            position = positions[turn]
            winning *= -1
            outcome = winning
            
            # debugging
            # print("-----------------")
            # print("saving position")
            # print("outcome {}".format(outcome))
            # print("board")
            # current = position.state[0, 0, 2]
            # print("Current player: {}".format(current))
            # if player == 0:
            #    board = position.state[:, :, 1] + position.state[:, :, 0] * 2
            #if player == 1:
            #    board = position.state[:, :, 0] + position.state[:, :, 1] * 2
            #print("{}".format(board))
            
            player = 1 - player
            position.set_outcome(outcome)
            position_memory.add(position)

    def show_trajectories(self):
        """ This method prints information of the seen trajectories. """
        print("Number of seen trajectories: {}".format(np.sum(self.trajectories.values())))
        print("Number of unique trajectories: {}".format(len(self.trajectories)))
