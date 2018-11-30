import agent
import game
import memory
import model
import keras.backend as K

import config

import numpy as np


class Pipeline:
    """
    This class controls the self-play training process of an AlphaZeroAgent.

    Attributes:
        pipeline_id: A string that is used as a unique id for the corresponding run.

        variant: Either "TicTacToe" or "Connect4". The attributes input_shape, num_possible_moves,
            and game are set accordingly.

        input_shape: The input shape for the neural networks: (board_height, board_width, 3)
        num_possible_moves: Number of possible moves for the first turn of the game.
        game: Instance of the game environment corresponding to the variant attribute. It is
            used for the self-play phase.

        memory: A PositionMemory object. It saves the self-play positions, adds reflections
            and rotations, and provides training data for the model.

        model: An AZmodel object. It is used for the optimization phase and the evaluations
            for the MCTS.

        lr: The initial learning rate for the neural network's training process. The value is
            provided by config.py if not set manually (see __init__).
        reg: The regularization strength for the neural network's training process. The value
            is provided by config.py if not set manually (see __init__).

        agent: An instance of AlphaZeroAgent. It is optimized and evaluated in this pipeline.

        seen_trajectories: A vector that contains the total number of trajectories seen by the
            agent during the self-play phases. Each component represents the sum of all
            self-play phases for one iteration.
        unique_trajectories: Analogously to seen_trajectories this integer represents the total
            number of unique trajectories seen by the agent.

        win_ratio: This vector contains the win ratios of the agent during the evaluation phases.
            Each component represents the win rate of one evaluation phase.
        draw_ratio: This vector contains the draw ratios of the agent during the evaluation phases.
            Each component represents the draw rate of one evaluation phase.
        """

    def __init__(self, pipeline_id, variant="TicTacToe", lr=None, reg=None):
        """ Initializes all components of the self-play training pipelines. """
        # game environment
        if variant == "Connect4":
            self.input_shape = (6, 7, 3)
            self.num_possible_moves = 7
            self.game = game.Connect4Optimized()
        if variant == "TicTacToe":
            self.input_shape = (3, 3, 3)
            self.num_possible_moves = 9
            self.game = game.TicTacToeOptimized()

        self.variant = variant

        self.lr = lr
        if lr is None:
            self.lr = config.NEURAL_NETWORKS['learning_rate']

        self.reg = reg
        if reg is None:
            self.reg = config.NEURAL_NETWORKS['regularization_strength']

        # memory
        self.memory = memory.PositionMemory(variant=variant)

        # model
        self.pipeline_id = pipeline_id
        self.model = model.AZModel(
            memory=self.memory,
            input_shape=self.input_shape,
            num_possible_moves=self.num_possible_moves,
            model_id=self.pipeline_id,
            lr=self.lr,
            reg=self.reg
        )

        # agent
        self.agent = agent.AlphaZeroAgent(model=self.model, variant=variant)
        
        # trajectories counter
        self.seen_trajectories = None
        self.unique_trajectories = None
        
        # evaluation
        self.win_ratio = []
        self.draw_ratio = []

        # logger
        # self.logger = logs.get_logger()

    def run(self, num_iterations):
        """ This method executes the self-play training pipeline.

        Args:
            num_iterations: Number of iterations.

        Returns:
            win_ratio: This vector contains the win ratios of the agent during the evaluation phases.
                Each component represents the win rate of one evaluation phase.
            draw_ratio: This vector contains the draw ratios of the agent during the evaluation phases.
                Each component represents the draw rate of one evaluation phase.
            seen_trajectories: A vector that contains the total number of trajectories seen by the
            agent during the self-play phases. Each component represents the sum of all
                self-play phases for one iteration.
            unique_trajectories: Analogously to seen_trajectories this integer represents the total
                number of unique trajectories seen by the agent.
        """
        self.seen_trajectories = np.zeros(num_iterations)
        self.unique_trajectories = np.zeros(num_iterations)
        for iteration in range(num_iterations):
            #self.logger.info("Pipeline: ######## Iteration {} | Self-Play ########".format(iteration))
            print("iteration {} | self-play".format(iteration))
            self.self_play(iteration)

            #self.logger.info("Pipeline: ######## Iteration {} | Optimization ########".format(iteration))
            print("iteration {} | optimization".format(iteration))
            self.optimization(iteration)

            #self.logger.info("Pipeline: ######## Iteration {} | Evaluation ########".format(iteration))
            print("iteration {} | evaluation".format(iteration))
            self.evaluation(iteration)
            
        return self.win_ratio, self.draw_ratio, self.seen_trajectories, self.unique_trajectories

    def self_play(self, iteration):
        """ Executes the self-play phase. The number of games is provided by config.py.
         The memory is saved in saved/ at the end of the phase.

        Args:
             iteration: The current iteration of the self-play training process.
         """
        num_games = config.SELF_PLAY['num_games']
        for i in range(num_games):
            #self.logger.info("Pipeline: Self-Play - Game {}".format(i))
            self.agent.self_play(self.memory, self.game)

        memory_id = "position_memory_{}_ep_{}".format(str(self.id),iteration)
        print("saving memory {}".format(memory_id))
        self.memory.save(memory_id)

    def optimization(self, iteration):
        """ Executes the optimization phase. Beginning with iteration 5, the learning
         rate is decreased every fifth iteration (new_rate = old_rate/2).

        Args:
            iteration: The current iteration of the self-play training process.
        """
        if iteration > 0 and iteration%5 == 0:
            self.lr = self.lr/2
            K.set_value(
                self.model.neural_network.network.optimizer.lr,
                self.lr
            )
        self.model.train()

    def evaluation(self, iteration):
        """ Executes the evaluation phase. The agent plays against a random player. The
        agent plays as player one half of the time.

        Args:
            iteration: The current iteration of the self-play training process.
        """
        num_games = config.EVALUATION['num_games']
        wins = 0
        draws = 0

        player = 1

        for i in range(num_games):
            #self.logger.info("Pipeline: Evaluation - Game {}".format(i))
            winner = agent_vs_random(self.agent, player, self.variant)
            
            if winner == 0:
                draws += 1
            if winner == player:
                wins += 1
                # self.logger.info("agent won ({})".format(wins))
            player *= -1
        # self.logger.info("win ratio {}".format(wins/num_games))
        win_rate = wins/num_games
        draw_rate = draws/num_games
        self.win_ratio.append(win_rate)
        self.draw_ratio.append(draw_rate)
        print("agent vs random - win ratio {} - draw ratio {}".format(win_rate, draw_rate))
        
        self.seen_trajectories[iteration], self.unique_trajectories[iteration] = self.agent.show_trajectories()


def agent_vs_random(eval_agent, player, variant="TicTacToe"):
    """ Executes one game between eval_agent and a random player.

    Args:
        eval_agent: The AlphaZeroAgent instance.
        player: If player=-1, the agent plays as player two. Otherwise,
            the agent begins the game as player one.
        variant: The game variant. Either "TicTacToe" or "Connect4".
    """
    if variant == "TicTacToe":
        game_environment = game.TicTacToeOptimized()
        max_length = 9
    if variant == "Connect4":
        game_environment = game.Connect4Optimized()
        max_length = 42

    player_one = eval_agent
    player_two = agent.RandomAgent(eval_agent)

    if player == -1:
        player_two = eval_agent
        player_one = agent.RandomAgent(eval_agent)

    # reset game
    game_environment.reset_game()

    player_one.join_game(game_environment)
    player_two.join_game(game_environment)

    current_player = game_environment.current_player

    winning = 0
    turn = 0

    num_simulations = config.EVALUATION['num_simulations']

    while winning is 0 and turn < max_length:
        if current_player == 0:
            winning, _, _ = player_one.play_move(num_simulations, temperature=0)
        if current_player == 1:
            winning, _, _ = player_two.play_move(num_simulations, temperature=0)

        current_player = game_environment.current_player
        turn += 1

    if current_player == 0:
        winner = -1*winning
    else:
        winner = winning

    return winner


def agent_vs_agent(eval_agent, best_agent, player=1, variant="TicTacToe"):
    """ Executes one game between two agents (eval_agent and best_agent).

    Args:
        eval_agent: The AlphaZeroAgent instance that is evaluated.
        best_agent: The current best agent in this run.
        player: If player=-1, eval_agent plays as player two. Otherwise,
            eval_agent begins the game as player one.
        variant: The game variant. Either "TicTacToe" or "Connect4". """
    if variant == "TicTacToe":
        game_environment = game.TicTacToeOptimized()
        max_length = 9
    if variant == "Connect4":
        game_environment = game.Connect4Optimized()
        max_length = 42

    player_one = eval_agent
    player_two = best_agent

    if player == -1:
        player_two = best_agent
        player_one = eval_agent

    # reset game
    game_environment.reset_game()

    player_one.join_game(game_environment)
    player_two.join_game(game_environment)

    current_player = game_environment.current_player

    winning = 0
    turn = 0

    num_simulations = config.EVALUATION['num_simulations']

    while winning is 0 and turn < max_length:
        if current_player == 0:
            winning, _, _ = player_one.play_move(num_simulations, temperature=0, opponent=player_two)
            current_position = game_environment.get_current_position()
            board = 2*current_position.state[:,:,0] + 1*current_position.state[:,:,1]
        if current_player == 1:
            winning, _, _ = player_two.play_move(num_simulations, temperature=0, opponent=player_one)
            current_position = game_environment.get_current_position()
            board = 2*current_position.state[:,:,1] + 1*current_position.state[:,:,0]

        current_player = game_environment.current_player
        turn += 1
        
        print("===========")
        print(" {} | {} | {}".format(
            board[0,0],
            board[0,1],
            board[0,2]
        ))
        print(" {} | {} | {}".format(
            board[1,0],
            board[1,1],
            board[1,2]
        ))
        print(" {} | {} | {}".format(
            board[2,0],
            board[2,1],
            board[2,2]
        ))
        print("===========")

    if current_player == 0:
        winner = -1*winning
    else:
        winner = winning

    return winner


def agent_vs_player(az_agent, player=1, variant="TicTacToe"):
    """ Executes one game between an agent an a human player.

        Args:
            az_agent: The AlphaZeroAgent instance that is evaluated.
            player: If player=-1, the agent plays as player two. Otherwise,
                the agent begins the game as player one.
            variant: The game variant. Either "TicTacToe" or "Connect4". """
    # logger = logs.get_logger()
    if variant == "TicTacToe":
        game_environment = game.TicTacToeOptimized()
        max_length = 9
    if variant == "Connect4":
        game_environment = game.Connect4Optimized()
        max_length = 42

    player_one = az_agent
    player_two = agent.Player(az_agent)

    if player == -1:
        player_two = az_agent
        player_one = agent.Player(az_agent)

    # reset game
    game_environment.reset_game()

    player_one.join_game(game_environment)
    player_two.join_game(game_environment)

    current_player = game_environment.current_player

    winning = 0
    turn = 0

    num_simulations = config.EVALUATION['num_simulations']

    while winning is 0 and turn < max_length:
        
        if current_player == 0:
            winning, _, _ = player_one.play_move(num_simulations, temperature=0)
            current_position = game_environment.get_current_position()
            board = 2*current_position.state[:,:,0] + 1*current_position.state[:,:,1]
        if current_player == 1:
            winning, _, _ = player_two.play_move(num_simulations, temperature=0)
            current_position = game_environment.get_current_position()
            board = 2*current_position.state[:,:,1] + 1*current_position.state[:,:,0]
            
        current_player = game_environment.current_player
        turn += 1

        if variant == "TicTacToe":
            print("===========")
            print(" {} | {} | {}".format(
                board[0,0],
                board[0,1],
                board[0,2]
            ))
            print(" {} | {} | {}".format(
                board[1,0],
                board[1,1],
                board[1,2]
            ))
            print(" {} | {} | {}".format(
                board[2,0],
                board[2,1],
                board[2,2]
            ))
            print("-----------")
            value, policy = az_agent.model.evaluate(current_position.state)
            policy = policy*current_position.legal_actions
            policy = policy/np.sum(policy)
            print("value: {}" .format(value))
            print("policy:")
            print(" {} | {} | {}".format(
                policy[0],
                policy[1],
                policy[2]
            ))
            print(" {} | {} | {}".format(
                policy[3],
                policy[4],
                policy[5]
            ))
            print(" {} | {} | {}".format(
                policy[6],
                policy[7],
                policy[8]
            ))
            print("===========")
        if variant == "Connect4":
            print("===========")
            print(" {} | {} | {} | {} | {} | {} | {}".format(
                board[0, 0],
                board[0, 1],
                board[0, 2],
                board[0, 3],
                board[0, 4],
                board[0, 5],
                board[0, 6]
            ))
            print(" {} | {} | {} | {} | {} | {} | {}".format(
                board[1, 0],
                board[1, 1],
                board[1, 2],
                board[1, 3],
                board[1, 4],
                board[1, 5],
                board[1, 6]
            ))
            print(" {} | {} | {} | {} | {} | {} | {}".format(
                board[2, 0],
                board[2, 1],
                board[2, 2],
                board[2, 3],
                board[2, 4],
                board[2, 5],
                board[2, 6]
            ))
            print(" {} | {} | {} | {} | {} | {} | {}".format(
                board[3, 0],
                board[3, 1],
                board[3, 2],
                board[3, 3],
                board[3, 4],
                board[3, 5],
                board[3, 6]
            ))
            print(" {} | {} | {} | {} | {} | {} | {}".format(
                board[4, 0],
                board[4, 1],
                board[4, 2],
                board[4, 3],
                board[4, 4],
                board[4, 5],
                board[4, 6]
            ))
            print(" {} | {} | {} | {} | {} | {} | {}".format(
                board[5, 0],
                board[5, 1],
                board[5, 2],
                board[5, 3],
                board[5, 4],
                board[5, 5],
                board[5, 6]
            ))
            print("-----------")
            value, policy = az_agent.model.evaluate(current_position.state)
            policy = policy * current_position.legal_actions
            policy = policy / np.sum(policy)
            print("value: {}".format(value))
            print("policy:")
            print(" {} | {} | {} | {} | {} | {} | {}".format(
                policy[0],
                policy[1],
                policy[2],
                policy[3],
                policy[4],
                policy[5],
                policy[6]
            ))
            print(" {} | {} | {} | {} | {} | {} | {}".format(
                policy[7],
                policy[8],
                policy[9],
                policy[10],
                policy[11],
                policy[12],
                policy[13]
            ))
            print(" {} | {} | {} | {} | {} | {} | {}".format(
                policy[14],
                policy[15],
                policy[16],
                policy[17],
                policy[18],
                policy[19],
                policy[20]
            ))
            print(" {} | {} | {} | {} | {} | {} | {}".format(
                policy[21],
                policy[22],
                policy[23],
                policy[24],
                policy[25],
                policy[26],
                policy[27]
            ))
            print(" {} | {} | {} | {} | {} | {} | {}".format(
                policy[28],
                policy[29],
                policy[30],
                policy[31],
                policy[32],
                policy[33],
                policy[34]
            ))
            print(" {} | {} | {} | {} | {} | {} | {}".format(
                policy[35],
                policy[36],
                policy[37],
                policy[38],
                policy[39],
                policy[40],
                policy[41]
            ))
            print("===========")
    if current_player == 0:
        winner = -1*winning
    else:
        winner = winning
    print("Player {} won the game after {} turns.".format(winner, turn))
    return winner