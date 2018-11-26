import logs
import numpy as np
import config


# Optimization: http://www.moderndescartes.com/essays/deep_dive_mcts/
# nodes save statistics of children

class Node:
    """
    Represents a state in the search tree.

    Attributes:
        parent: Parent node in the search tree.

        last_action: Index of the action that lead to this state.

        children: This dictionary contains all expanded children of this node.

        legal_actions: A vector that represents all legal moves in this game state
            (1: legal move, 0: illegal move).

        len_action_space: The length of the action space of the game.

        state: The game state (see memory.Position).

        player: Represents which player's turn it is (1: player 1, 0: player 2).

        visits_children: Contains the visit counts for all children of this node.

        total_value_children: Contains the sum of all values for all children of this node.

        priors_children: Contains the prior probabilities (provided by the model) for all
            children of this node.
    """

    def __init__(self, state, legal_actions, last_action=None, parent=None):
        """ Initializes the node and its attributes.

        Args:
            state: See above.
            legal_actions: See above.
            last_action: See above. "None" for the root node.
            parent: See above. "None" for the root node.
        """
        self.parent = parent
        self.last_action = last_action
        self.children = {}
        self.legal_actions = legal_actions
        self.len_action_space = len(legal_actions)
        self.state = state

        self.player = state[2][0]

        self.visits_children = np.zeros(self.len_action_space, dtype=np.uint16)
        self.total_value_children = np.zeros(self.len_action_space, dtype=np.float32)
        self.priors_children = np.zeros(self.len_action_space, dtype=np.float32)

    def is_leaf(self):
        """ Returns True if the node is a leaf and False otherwise. """
        if sum(self.priors_children) == 0: return True
        return False

    def total_visits(self, temperature=1):
        """ Returns the sum of visit counts for all children of this node.
            The visit counts might be exponentiated with a given temperature > 0.
        """
        return np.sum(self.visits_children ** (1/temperature))

    def q_children(self):
        """ Computes and returns Q(s,a) for all actions in this state.
        """
        return self.total_value_children / (self.visits_children + 1)

    def u_children(self, c_puct=None, add_dirichlet=False):
        """ Computes and returns U(s,a) for all actions in this state.

        Args:
            c_puct: This constant controls the amount of exploration by scaling U.
            add_dirichlet: Dirichlet noise is added to U(s,a) if True.
        """
        if c_puct is None:
            c = config.MCTS['c_puct']
        else:
            c = c_puct

        if add_dirichlet:
            epsilon = config.MCTS['dir_epsilon']
            nu = np.random.dirichlet([config.MCTS['dir_alpha']] * self.len_action_space)
            U = (1-epsilon) * self.priors_children + epsilon * nu
            U = c * U
        else:
            U = c * self.priors_children

        U *= np.sqrt(self.total_visits() + 1)
        U /= 1 + self.visits_children

        return U

    def upper_confidence_bound(self, c_puct=None, add_dirichlet=False):
        """ Computes and returns the upper confidence bound for all actions in
        this state.

        Args:
            c_puct: This constant controls the amount of exploration by scaling U.
            add_dirichlet: Dirichlet noise is added to U(s,a) if True.
        """
        Q = self.q_children()
        U = self.u_children(c_puct, add_dirichlet)
        return (Q + U) * self.legal_actions

    def best_action(self, c_puct=None, add_dirichlet=False):
        """ Returns the action with the highest upper confidence bound

        Args:
            c_puct: This constant controls the amount of exploration by scaling U.
            add_dirichlet: Dirichlet noise is added to U(s,a) if True.
        """
        ranking = np.argsort(self.upper_confidence_bound(c_puct, add_dirichlet))
        i = self.len_action_space - 1
        while i > 0 and self.legal_actions[ranking[i]] == 0:
            i -= 1

        if i == 0:
            return np.where(self.legal_actions == 1)[0][0]
        else:
            return ranking[i]

    def expand(self, action, child_node):
        """ Adds a child to this node.

        Args:
            action: Index of the action corresponding to the new node.
            child_node: Node object that represents the child's game state and saves its statistics.
        """
        self.children[action] = child_node

    def backup(self, value, action):
        """ Updates the statistics of the child node corresponding to the action.

        Args:
             value: Estimated value (provided by the model) of the child node corresponding
                to the action.
             action: Index of the action.
        """
        self.visits_children[action] += 1
        self.total_value_children[action] += value

    def set_priors(self, probabilities):
        """ Sets the prior probabilities of all child nodes.

        Args:
            probabilities: Prior distribution over all possible moves (provided by the model).
        """
        for action, prior in enumerate(probabilities):
            self.priors_children[action] = prior


class MonteCarloTreeSearch:
    """
    Implementation of the Monte Carlo Tree Search (MCTS) algorithm.

    Attributes:
        root (Node): Current root node of the search tree. Needs to be set
            at the beginning of each MCTS run.
        c_puct (float): Constant determining the level of exploration.
    """

    def __init__(self, model, game, initial_position):
        """ Initializes a new search tree.

        Args:
            model (Model): The model which is used for evaluation.
            game (e.g. game.Game): The game environment which simulates moves and provides game states.
            initial_position (memory.Position): Simulations will start from this position of the game.
        """
        # initialize logger
        self.logger = logs.get_logger()

        self.model = model
        self.game = game
        self.len_action_space = self.game.get_len_action_space()

        # create root node
        self.root = Node(initial_position.state, initial_position.legal_actions)

        # initial evaluation
        self.evaluate(self.root)

    def simulation(self, add_dirichlet=False):
        """ Executes a full simulation that consists of three phases:

        Selection - Expansion (in self.select) - Evaluation - Backup

        Args:
            add_dirichlet: Dirichlet noise is added to U(s,a) if True.
         """
        self.logger.info("MCTS: ################ Selection ################")
        selected_action, leaf_node, winning_simulation, game_over = self.select(add_dirichlet=add_dirichlet)

        self.logger.info("MCTS: ################ Evaluation ################")
        if game_over == 1:
            value_estimate = -1*winning_simulation
        else:
            value_estimate = self.evaluate(leaf_node)

        self.logger.info("MCTS: ################ Backup ################")
        self.backup(leaf_node, value_estimate)

    def select(self, add_dirichlet=False):
        """ According to the statistics in the search tree
        actions are chosen until a leaf node s_L is reached.

        At each time step t < L the action which maximizes
        Q(s_t, a) + U(s_t, a) is selected.

        The leaf node s_L is expanded (added to the search tree as a new Node).

        Args:
            add_dirichlet: Dirichlet noise is added to U(s,a) if True.

        Returns:
            selected_action: Index of the action which lead to a leaf node.

            current_node: The expanded leaf node.

            simulation_winner: Contains the winner if the game ended during the
                simulation.

            game_over: Only True if the game ended during the simulation.
        """
        np.set_printoptions(precision=3)
        current_node = self.root
        selected_action = None
        winning_simulation = None
        game_over = False

        self.game.reset_simulation()

        # Debugging
        #self.logger.info("MCTS: ----- root node -----")
        #self.logger.info("MCTS: legal actions: {}".format(self.root.legal_actions))
        #self.logger.info("MCTS: children priors: {}".format(self.root.priors_children))
        #self.logger.info("MCTS: children visits: {}".format(self.root.visits_children))
        #self.logger.info("MCTS: children total values: {}".format(self.root.total_value_children))
        #self.logger.info("MCTS: position")
        #self.logger.info("MCTS: ---------------------")
        #print("select: #### Selection started ####")
        #print("select: root:")
        #print("select: legal actions: {}" .format(self.root.legal_actions))
        #print("select: priors: {}" .format(self.root.priors_children))
        #print("select: children visits: {}" .format(self.root.visits_children))
        #print("select: children total values: {}" .format(self.root.total_value_children))
        #position = self.game.get_current_position_simulation()
        #print("select: stones current player")
        #print(position.state[:, :, 0])
        #print("select: stones next player")
        #print(position.state[:, :, 1])
        #print("select: ---------------------------")

        while not current_node.is_leaf():
            selected_action = current_node.best_action(add_dirichlet=add_dirichlet)
            add_dirichlet = False # noise is only added to the root's priors

            winning_simulation, turn = self.game.simulate_move(selected_action)

            if selected_action not in current_node.children:
                self.expand(current_node, selected_action)

            next_node = current_node.children[selected_action]

            #print("select: selected action: {}" .format(selected_action))
            #self.logger.info("MCTS: selected action: {}".format(selected_action))
            #self.logger.info("MCTS: ----- current node -----")
            #self.logger.info("MCTS: legal actions: {}".format(next_node.legal_actions))
            #self.logger.info("MCTS: children priors: {}".format(next_node.priors_children))
            #self.logger.info("MCTS: children visits: {}".format(next_node.visits_children))
            #self.logger.info("MCTS: children total values: {}".format(next_node.total_value_children))

            #print("select: next node:")
            #print("select: legal actions: {}".format(next_node.legal_actions))
            #print("select: priors: {}".format(next_node.priors_children))
            #print("select: children visits: {}".format(next_node.visits_children))
            #print("select: children total values: {}".format(next_node.total_value_children))
            #position = self.game.get_current_position_simulation()
            #print("select: stones current player")
            #print(position.state[:,:,0])
            #print("select: stones next player")
            #print(position.state[:,:,1])
            #print("select: ---------------------------")

            current_node = next_node

            if winning_simulation == 1 or turn > self.game.board_size:
                # self.logger.info("MCTS: Player {} won simulated game." .format(simulation_winner))
                game_over = True # added for evaluation of won games (no need to estimate the value)
                break
                
        return selected_action, current_node, winning_simulation, game_over

    def expand(self, leaf_node, action):
        """ Creates a new Node from the current position of the game environment and
         adds it to the search tree.

         Args:
             leaf_node: Node object. Parent of the new node.
             action: Index of the action which results in the new game state.
         """
        position = self.game.get_current_position_simulation()

        new_child = Node(
            state=position.state,
            legal_actions=position.legal_actions,
            last_action=action,
            parent=leaf_node
        )

        leaf_node.expand(action, new_child)

    def evaluate(self, leaf_node):
        """ The node's game state is evaluated by the model and the priors are
        set accordingly.

        Args:
            leaf_node: Node object of the game state that is evaluated.

        Returns:
            value_estimate: The model's estimated value for the given game state
                (1: Current player will winning, -1: Current player will lose).
        """
        value_estimate, probabilities = self.model.evaluate(leaf_node.state)
        leaf_node.set_priors(probabilities)

        return value_estimate

    def backup(self, leaf_node, value):
        """ Backup of the value estimate from the evaluated node up to the root.

        Args:
            leaf_node: Last node of the trajectory that is updated.
            value: The model's estimated value for the leaf node's game state.
        """
        current_node = leaf_node
        last_action = current_node.last_action

        while current_node.parent is not None:
            current_node = current_node.parent
            current_node.backup(value, last_action)
            last_action = current_node.last_action
            value *= -1

    def play(self, temperature):
        """ Selects an action to be played according to the root node's statistics.

        For temperature = 0:
            The action that was visited the most is selected.
        For temperature > 0:
            The visit counts of the distribution are exponentiated by (1/temperature)
            before the selection.
        For temperature = 0:
            The probabilities match the relative frequency of visits.

        Args:
            temperature: The visit counts are exponentiated by this parameter
                before sampling.

        Returns:
             selected_action:
             distribution:
         """
        self.logger.info("MCTS: ################ Play ################")

        distribution = np.zeros(self.len_action_space)

        sum_priors = np.sum(self.root.priors_children)

        if sum_priors == 0:
            # random choice
            selected_action = np.random.choice(self.len_action_space, size=1)[0]

            return selected_action, distribution

        if temperature == 0:
            distribution = self.root.visits_children
            total_visits = np.sum(self.root.visits_children)

            if total_visits < 1:
                distribution = self.root.legal_actions
                total_visits = np.sum(distribution)

            distribution = distribution / total_visits
            selected_action = np.argmax(distribution)
        else:
            total_visits = self.root.total_visits(temperature)
            if total_visits < 1:
                distribution = self.root.legal_actions
                total_visits = np.sum(distribution)
            else:
                distribution = self.root.visits_children ** (1/temperature)

            distribution = distribution / total_visits

            selected_action = np.random.choice(self.len_action_space, size=1, p=distribution)[0]

        self.update_root(selected_action)

        #self.logger.info("MCTS: distribution: ")
        #self.logger.info("MCTS: {} | {} | {} " .format(distribution[0], distribution[1], distribution[2]))
        #self.logger.info("MCTS: {} | {} | {} " .format(distribution[3], distribution[4], distribution[5]))
        #self.logger.info("MCTS: {} | {} | {} " .format(distribution[6], distribution[7], distribution[8]))
        #self.logger.info("MCTS: selected action: {}" .format(selected_action))
        return selected_action, distribution

    def update_root(self, selected_action):
        """ The root node is changed to the current root's child corresponding to the
        selected action.

        Args:
            selected_action: Index of the selected action.
        """
        if selected_action not in self.root.children:
            self.game.reset_simulation()
            self.game.simulate_move(selected_action)
            self.expand(self.root, selected_action)
            self.evaluate(self.root.children[selected_action])

        next_node = self.root.children[selected_action]
        self.root = next_node