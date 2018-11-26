import agent
import game
import memory
import model
import keras.backend as K

import logs
import config

import copy


class Pipeline:
    """ TODO docstring Pipeline """

    def __init__(self, id, variant="TicTacToe", lr=None, reg=None):
        """ TODO docstring __init__ """
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
        if lr == None:
            self.lr = config.NEURAL_NETWORKS['learning_rate']
        self.reg = reg

        # memory
        self.memory = memory.PositionMemory(variant=variant)

        # model
        self.id = id
        self.model = model.AZModel(
            memory=self.memory,
            input_shape=self.input_shape,
            num_possible_moves=self.num_possible_moves,
            model_id=self.id,
            lr=self.lr,
            reg=self.reg
        )

        # agent
        self.agent = agent.AlphaZeroAgent(model=self.model, variant=variant)

        # evaluation
        self.best_model_version = 0
        self.best_win_rate = 0
        self.best_draw_rate = 0
        self.evaluation_threshold = 0.55
        self.win_ratio = []
        self.draw_ratio = []
        self.win_ratio_agent = []
        self.draw_ratio_agent = []
        # logger
        # self.logger = logs.get_logger()

    def load(self):
        """ TODO docstring load"""
        pass

    def run(self, num_iterations):
        """ TODO docstring run """
        for iteration in range(num_iterations):
            #self.logger.info("Pipeline: ######## Iteration {} | Self-Play ########".format(iteration))
            print("iteration {} | self-play".format(iteration))
            self.self_play(iteration)

            #self.logger.info("Pipeline: ######## Iteration {} | Optimization ########".format(iteration))
            print("iteration {} | optimization".format(iteration))
            self.optimization(iteration)

            #self.logger.info("Pipeline: ######## Iteration {} | Evaluation ########".format(iteration))
            print("iteration {} | evaluation".format(iteration))
            self.evaluation()

        return self.win_ratio, self.draw_ratio

    def self_play(self, iteration):
        """ TODO docstring self_play """
        num_games = config.SELF_PLAY['num_games']
        for i in range(num_games):
            #self.logger.info("Pipeline: Self-Play - Game {}".format(i))
            #print("Game {}".format(i))
            self.agent.self_play(self.memory, self.game)

            #if (i + 1) % 25 is 0:
            #    memory_id = "position_memory_{}_ep_{}_{}".format(str(self.id),iteration, i + 1)
            #    # self.logger.info("saving memory {}".format(memory_id))
            #    print("saving memory {}".format(memory_id))
            #    self.memory.save(memory_id)
        memory_id = "position_memory_{}_ep_{}".format(str(self.id),iteration)
        print("saving memory {}".format(memory_id))
        self.memory.save(memory_id)

    def optimization(self, iteration):
        """ TODO docstring optimization """
        if iteration > 0 and iteration%5 == 0:
            self.lr = self.lr/2
            K.set_value(
                self.model.neural_network.network.optimizer.lr,
                self.lr
            )
        self.model.train()

    def evaluation(self):
        """ TODO docstring evaluation """
        num_games = config.EVALUATION['num_games']
        wins = 0
        draws = 0
        
        #best_model = model.AZModel(
        #            memory=self.memory,
        #            input_shape=self.input_shape,
        #            num_possible_moves=self.num_possible_moves,
        #            model_id=self.id
        #        )
        #best_model.load(self.best_model_version)
        #best_agent = agent.AlphaZeroAgent(model=best_model)
        player = 1

        for i in range(num_games):
            #self.logger.info("Pipeline: Evaluation - Game {}".format(i))
            winner = agent_vs_random(self.agent, player, self.variant)
            
            if winner == 0:
                draws += 1
            if winner == player:
                wins += 1
                # self.logger.info("agent won ({})".format(wins))
                #print("agent won ({})".format(wins))
            player *= -1
        # self.logger.info("win ratio {}".format(wins/num_games))
        win_rate = wins/num_games
        draw_rate = draws/num_games
        self.win_ratio.append(win_rate)
        self.draw_ratio.append(draw_rate)
        print("agent vs random - win ratio {} - draw ratio {}".format(win_rate, draw_rate))

        #wins = 0
        #draws = 0
        #player = 1
        #for i in range(num_games):
        #    #self.logger.info("Pipeline: Evaluation - Game {}".format(i))
        #    #print("game {}".format(i))
        #    winner = agent_vs_random(self.agent, player)
        #    winner = agent_vs_agent(self.agent, best_agent, player)
        #    
        #    if winner == 0:
        #        draws += 1
        #    if winner == player:
        #        wins += 1
        #        # self.logger.info("agent won ({})".format(wins))
        #        #print("agent won ({})".format(wins))
        #    player *= -1
        # self.logger.info("win ratio {}".format(wins/num_games))
        #win_rate = wins/num_games
        #draw_rate = draws/num_games
        #self.win_ratio_agent.append(win_rate)
        #self.draw_ratio_agent.append(draw_rate)
        #print("agent vs agent - win ratio {} - draw ratio {}".format(win_rate, draw_rate))

        #if win_rate > 0.55:
        #    self.best_model_version = self.agent.model.get_model_count()-1
        #    self.best_win_rate = win_rate
        #    self.best_draw_rate = draw_rate
        #s    print("New best agent!")
        #else:
            #self.agent=best_agent
            
       
        #if win_rate > self.best_win_rate:
        #    self.best_model_version = self.agent.model.get_model_count()-1
        #    self.best_win_rate = win_rate
         #   self.best_draw_rate = draw_rate
         #   print("New best agent!")
            
        #else:
        #    if (win_rate + 0.05 > self.best_win_rate) and (draw_rate-0.1 > self.best_draw_rate):
        #        self.best_model_version = self.agent.model.get_model_count()-1
        #        self.best_win_rate = win_rate
        #        self.best_draw_rate = draw_rate
        #        print("New best agent!")
        #    else:
                #best_model = model.AZModel(
                #    memory=self.memory,
                #    input_shape=self.input_shape,
                #    num_possible_moves=self.num_possible_moves,
                #    model_id=self.id
                #)
                #best_model.load(self.best_model_version)
                #best_agent = agent.AlphaZeroAgent(model=best_model)
                #self.agent=best_agent


def agent_vs_random(eval_agent, player, variant="TicTacToe"):
    # logger = logs.get_logger()
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
            winning, _ = player_one.play_move(num_simulations, temperature=0)
        if current_player == 1:
            winning, _ = player_two.play_move(num_simulations, temperature=0)

        current_player = game_environment.current_player
        turn += 1

    # logger.info("Player {} has won the game after {} turns.".format(winner, turn))
    #print("Player {} won the game after {} turns.".format(winner, turn))
    if current_player == 0:
        winner = -1*winning
    else:
        winner = winning

    return winner


def agent_vs_agent(eval_agent, best_agent, player, variant="TicTacToe"):
    # logger = logs.get_logger()
    # game_environment = game.Connect4(6, 7)
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
        if current_player == 1:
            winning, _ = player_one.play_move(num_simulations, temperature=0, opponent=player_two)
        if current_player == -1:
            winning, _ = player_two.play_move(num_simulations, temperature=0, opponent=player_one)

        current_player = game_environment.current_player
        turn += 1

    # logger.info("Player {} has won the game after {} turns.".format(winner, turn))
    #print("Player {} won the game after {} turns.".format(winner, turn))

    return winning
