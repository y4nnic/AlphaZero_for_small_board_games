import numpy as np
import copy
import logs
import memory


class Game():

    def get_len_action_space(self):
        pass

    def reset_simulation(self):
        pass

    def reset_game(self):
        pass

    def execute_move(self, action):
        pass

    def get_current_position(self):
        pass

    def simulate_move(self, action):
        pass

    def get_current_position_simulation(self):
        pass


class TicTacToeOptimized(Game):
    """ TODO docstring TicTacToe"""

    def __init__(self):
        """ Initializes the game.

            Representation of the board (for one player):

            Bits (0 = least signigicant)
            28-30   -> first row
            24-26   -> second row
            20-22   -> third row
            16-18   -> first column
            12-14   -> second column
             8-10   -> third column
             4- 6   -> first diagonal (left top -> right bottom)
             0- 2   -> second diagonal (right top -> left bottom)

            Representation of the players:

            Player 1 -> 0
            Player 2 -> 1
        """
        self.moves = {
            0:0x40040040,
            1:0x20004000,
            2:0x10000404,
            3:0x04020000,
            4:0x02002022,
            5:0x01000200,
            6:0x00410001,
            7:0x00201000,
            8:0x00100110
        }

        self.player_1 = 0
        self.player_2 = 1

        self.board_length = 3
        self.board_size = self.board_length * self.board_length

        board_p1 = 0x00000000
        board_p2 = 0x00000000
        self.boards = [board_p2, board_p2]
        self.legal_moves = np.ones(9, dtype=np.uint8)
        self.current_player = self.player_1
        self.turn = 0

        self.boards_simulation = [board_p1, board_p2]
        self.legal_moves_simulation = np.ones(9, dtype=np.uint8)
        self.current_player_simulation = self.current_player
        self.turn_simulation = self.turn

    def get_len_action_space(self):
        return self.board_size

    def reset_simulation(self):
        """ Resets the simulation to the current state of the game. """
        self.boards_simulation = copy.deepcopy(self.boards)
        self.legal_moves_simulation = copy.deepcopy(self.legal_moves)
        self.current_player_simulation = self.current_player

    def reset_game(self):
        """ Resets the game. The board is cleared and it's player one's turn."""
        board_p1 = 0x00000000
        board_p2 = 0x00000000
        self.boards = [board_p1, board_p2]
        self.legal_moves = np.ones(9, dtype=np.uint8)
        self.current_player = 0

    def execute_move(self, action):
        """ TODO docstring execute_move """
        if self.legal_moves[action] == 1:
            self.boards[self.current_player] |= self.moves[action]
            self.legal_moves[action] = 0
        else:
            print("Illegal move!")

        winning = self.is_winning()
        self.current_player = 1 - self.current_player
        self.turn += 1
        return winning

    def is_winning(self):
        """ Checks the board for winning combinations. """
        winning = (self.boards[self.current_player] + 0x11111111) & 0x88888888
        if winning != 0:
            return 1
        return winning

    def get_current_position(self):
        """ Returns a Position object that contains the current game state. """
        pieces_p1 = self.board_to_array(self.boards[self.player_1])
        pieces_p2 = self.board_to_array(self.boards[self.player_2])

        if self.current_player == self.player_1:
            colour = np.ones((self.board_length, self.board_length, 1), dtype=np.uint8)
            state = np.concatenate((pieces_p1, pieces_p2, colour), axis=2)

        if self.current_player == self.player_2:
            colour = np.zeros((self.board_length, self.board_length, 1), dtype=np.uint8)
            state = np.concatenate((pieces_p2, pieces_p1, colour), axis=2)

        return memory.Position(state, self.legal_moves)

    def simulate_move(self, action):
        """ TODO docstring simulate move """
        if self.legal_moves_simulation[action] == 1:
            self.boards_simulation[self.current_player_simulation] |= self.moves[action]
            self.legal_moves_simulation[action] = 0
        else:
            print("Illegal move (simulation)!")

        winning = self.is_winning_simulation()
        self.current_player_simulation = 1 - self.current_player_simulation
        self.turn_simulation += 1

        return winning, self.turn_simulation

    def is_winning_simulation(self):
        """ Checks the simulation board for winning combinations. """
        winning = (self.boards_simulation[self.current_player_simulation] + 0x11111111) & 0x88888888
        if winning != 0:
            return 1
        return winning

    def get_current_position_simulation(self):
        """ Returns a Position object that contains the current game state. """
        pieces_p1 = self.board_to_array(self.boards_simulation[self.player_1])
        pieces_p2 = self.board_to_array(self.boards_simulation[self.player_2])

        if self.current_player_simulation == self.player_1:
            colour = np.ones((self.board_length, self.board_length, 1), dtype=np.uint8)
            state = np.concatenate((pieces_p1, pieces_p2, colour), axis=2)

        if self.current_player_simulation == self.player_2:
            colour = np.zeros((self.board_length, self.board_length, 1), dtype=np.uint8)
            state = np.concatenate((pieces_p2, pieces_p1, colour), axis=2)

        return memory.Position(state, self.legal_moves_simulation)

    def board_to_array(self, board_bin):
        board = np.zeros((3,3,1), dtype=np.uint8)

        board[0,0] = (board_bin & 0x40) >> 6
        board[0,1] = (board_bin & 0x4000) >> 14
        board[0,2] = (board_bin & 0x4) >> 2
        board[1,0] = (board_bin & 0x20000) >> 17
        board[1,1] = (board_bin & 0x2) >> 1
        board[1,2] = (board_bin & 0x200) >> 9
        board[2,0] = board_bin & 0b1
        board[2,1] = (board_bin & 0x1000) >> 12
        board[2,2] = (board_bin & 0x10) >> 4
        return board


class TicTacToe(Game):
    """ TODO docstring TicTacToe """

    def __init__(self):
        """ TODO docstring __init__ """
        self.player_1 = 1
        self.player_2 = -1

        self.winning_combinations = {
             0: [[0, 1, 2], [0, 3, 6], [0, 4, 8]],
             1: [[0, 1, 2], [1, 4, 7]],
             2: [[0, 1, 2], [2, 5, 8], [2, 4, 6]],
             3: [[3, 4, 5], [0, 3, 6]],
             4: [[3, 4, 5], [1, 4, 5], [0, 4, 8], [2, 4, 6]],
             5: [[3, 4, 5], [2, 5, 8]],
             6: [[6, 7, 8], [0, 3, 6], [2, 4, 6]],
             7: [[6, 7, 8], [1, 4, 7]],
             8: [[6, 7, 8], [2, 5, 8], [0, 4, 8]]
        }

        self.board_length = 3
        self.board_size = self.board_length * self.board_length

        self.board = np.zeros((self.board_size, 1), dtype=np.int8)
        self.legal_moves = np.ones(self.board_size, dtype=np.uint8)
        self.current_player = self.player_1
        self.turn = 0

        self.board_simulation = copy.deepcopy(self.board)
        self.legal_moves_simulation = np.ones((self.board_size), dtype=np.uint8)
        self.current_player_simulation = self.current_player
        self.turn_simulation = 0

    def get_len_action_space(self):
        """ TODO docstring get_len_action_space """
        return self.board_size

    def reset_simulation(self):
        """ TODO docstring reset_simulation """
        self.board_simulation = copy.deepcopy(self.board)
        self.legal_moves_simulation = copy.deepcopy(self.legal_moves)
        self.current_player_simulation = self.current_player
        self.turn_simulation = self.turn

    def reset_game(self):
        """ TODO docstring reset_game """
        self.board = np.zeros((self.board_size, 1), dtype=np.int8)
        self.legal_moves = np.ones(self.board_size, dtype=np.uint8)
        self.current_player = self.player_1
        self.turn = 0

    def execute_move(self, action):
        """ TODO docstring execute_move """
        if self.legal_moves[action] == 1:
            self.board[action] = self.current_player
            self.legal_moves[action] = 0
        else:
            print("++++++++++ ILLEGAL MOVE +++++++++++")
            print("illegal move {}" .format(action))
            print("game: legal_moves: {}" .format(self.legal_moves))
            print("game current position:")
            print(" {} | {} | {} " .format(self.board[0], self.board[1], self.board[2]))
            print(" {} | {} | {} " .format(self.board[3], self.board[4], self.board[5]))
            print(" {} | {} | {} " .format(self.board[6], self.board[7], self.board[8]))

        winner = self.is_winning(action)
        self.current_player *= -1
        self.turn += 1
        # print(" ############ ")
        # print(" {} | {} | {} " .format(self.board[0], self.board[1], self.board[2]))
        # print(" {} | {} | {} " .format(self.board[3], self.board[4], self.board[5]))
        # print(" {} | {} | {} " .format(self.board[6], self.board[7], self.board[8]))
        # print(" ############ ")
        return winner

    def is_winning(self, action):
        """ TODO docstring is_winning """
        combinations = self.winning_combinations[action]
        winning_sum = self.board_length * self.current_player

        for indices in combinations:
            if np.sum(self.board[indices]) == winning_sum:
                return self.current_player
        return 0

    def get_current_position(self):
        """ TODO docstring get_current_position """
        pieces_flat_p1 = np.array(self.board == self.player_1, dtype=np.uint8)
        pieces_flat_p2 = np.array(self.board == self.player_2, dtype=np.uint8)

        pieces_p1 = np.array([pieces_flat_p1[0:3], pieces_flat_p1[3:6,:], pieces_flat_p1[6:,:]])
        pieces_p2 = np.array([pieces_flat_p2[0:3], pieces_flat_p2[3:6,:], pieces_flat_p2[6:,:]])

        if self.current_player == self.player_1:
            colour = np.ones((self.board_length, self.board_length, 1), dtype=np.uint8)
            state = np.concatenate((pieces_p1, pieces_p2, colour), axis=2)

        if self.current_player == self.player_2:
            colour = np.zeros((self.board_length, self.board_length, 1), dtype=np.uint8)
            state = np.concatenate((pieces_p2, pieces_p1, colour), axis=2)

        return memory.Position(state, self.legal_moves)

    def simulate_move(self, action):
        """ TODO docstring simulate_move """
        if self.legal_moves_simulation[action] == 1:
            self.board_simulation[action] = self.current_player_simulation
            self.legal_moves_simulation[action] = 0
        else:
            print("++++++++++ SIMULATION: ILLEGAL MOVE +++++++++++")
            print("illegal move {}".format(action))
            print("simulation: legal_moves: {}".format(self.legal_moves))
            print("simulation current position:")
            print(" {} | {} | {} ".format(self.board[0], self.board[1], self.board[2]))
            print(" {} | {} | {} ".format(self.board[3], self.board[4], self.board[5]))
            print(" {} | {} | {} ".format(self.board[6], self.board[7], self.board[8]))

        winner = self.is_winning_simulation(action)
        self.current_player_simulation *= -1
        self.turn_simulation += 1

        return winner, self.turn_simulation

    def is_winning_simulation(self, action):
        """ TODO docstring is_winning_simulation """
        combinations = self.winning_combinations[action]
        winning_sum = self.board_length * self.current_player

        for indices in combinations:
            if np.sum(self.board_simulation[indices]) == winning_sum:
                return self.current_player
        return 0

    def get_current_position_simulation(self):
        """ TODO docstring get_current_position """
        pieces_flat_p1 = np.array(self.board_simulation == self.player_1, dtype=np.uint8)
        pieces_flat_p2 = np.array(self.board_simulation == self.player_2, dtype=np.uint8)

        pieces_p1 = np.array([pieces_flat_p1[0:3], pieces_flat_p1[3:6], pieces_flat_p1[6:9]])
        pieces_p2 = np.array([pieces_flat_p2[0:3], pieces_flat_p2[3:6], pieces_flat_p2[6:9]])

        if self.current_player_simulation == self.player_1:
            colour = np.ones((self.board_length, self.board_length, 1), dtype=np.uint8)
            state = np.concatenate((pieces_p1, pieces_p2, colour), axis=2)

        if self.current_player_simulation == self.player_2:
            colour = np.zeros((self.board_length, self.board_length, 1), dtype=np.uint8)
            state = np.concatenate((pieces_p2, pieces_p1, colour), axis=2)

        return memory.Position(state, self.legal_moves_simulation)


class Connect4(Game):

    def __init__(self, board_height, board_width):
        """ Initializes the board and creates the initial state."""
        self.player_1 = 1
        self.player_2 = -1

        self.board_height = board_height
        self.board_width = board_width

        self.board = np.zeros((self.board_height, self.board_width, 1), dtype=np.int8)
        self.legal_moves = np.ones(self.board_width, dtype=np.uint8)
        self.current_player = self.player_1
        self.turn = 0

        self.board_simulation = copy.deepcopy(self.board)
        self.legal_moves_simulation = copy.deepcopy(self.legal_moves)
        self.current_player_simulation = self.player_1
        self.turn_simulation = 0

        # initialize logger
        #self.logger = logs.get_logger()

    def get_len_action_space(self):
        """ TODO docstring get_len_action_space """
        return self.board_width

    def reset_simulation(self):
        """ TODO docstring reset_simulation """
        self.board_simulation = copy.deepcopy(self.board)
        self.legal_moves_simulation = copy.deepcopy(self.legal_moves)
        self.current_player_simulation = self.current_player
        self.turn_simulation = 0

    def reset_game(self):
        """ TODO docstring reset_game """
        self.board = np.zeros(self.board.shape, dtype=np.int8)
        self.legal_moves = np.ones(self.legal_moves.shape)
        self.current_player = self.player_1
        self.turn = 0

    def execute_move(self, action, simulation=False):
        """ Places the corresponding piece (1 or -1) on the given field.

        Should only be called if is_legal_move returns True.
        The attribute next_player is updated accordingly after the
        move is executed.

        Args:
             column: Integer index of the target field on the board.
        """
        # TODO refactoring: simulate_move and execute_move -> little less computation
        if not simulation:
            board = self.board
            current_player = self.current_player
            turn = self.turn

        if simulation:
            board = self.board_simulation
            current_player = self.current_player_simulation
            turn = self.turn_simulation

        board_height, _, _ = board.shape

        # finding the first free position
        # TODO np.where(column == 0)[-1]
        row = self.next_empty_row(action, board)

        # placing the piece
        board[row, action, 0] = current_player

        # check whether there are still empty fields in this column
        if turn >= self.board_height:
            self.update_legal_moves(row, action, simulation)

        # does the move win the game?
        winner = 0

        if turn >= 6:
            if self.is_winning_move(row, action, is_simulation=simulation):
                winner = current_player

        # change the current player
        if simulation:
            self.current_player_simulation *= -1
            self.turn_simulation += 1
        else:
            self.current_player *= -1
            self.turn += 1


        #if simulation:
            #self.logger.info("Game: simulated move: {}".format(action))
            #self.logger.info("Game: simulated board: \n {}".format(self.board_simulation[:, :, 0]))
        #else:
            #self.logger.info("Game: played move: {}".format(action))
            #self.logger.info("Game: board: \n {}".format(self.board[:, :, 0]))

        return winner

    def get_current_position(self, simulation=False):
        """ TODO docstring get_current_position """
        board = self.board
        legal_actions = self.legal_moves
        player = self.current_player

        if simulation:
            board = self.board_simulation
            legal_actions = self.legal_moves_simulation
            player = self.current_player_simulation

        pieces_player_1 = np.array(board == self.player_1)
        pieces_player_2 = np.array(board == self.player_2)

        if player == self.player_1:
            colour = np.ones((self.board_height, self.board_width, 1), dtype=np.uint8)
            state = np.concatenate((pieces_player_1, pieces_player_2, colour), axis=2)

        if player == self.player_2:
            colour = np.zeros((self.board_height, self.board_width, 1), dtype=np.uint8)
            state = np.concatenate((pieces_player_2, pieces_player_1, colour), axis=2)

        # state = np.concatenate((pieces_player_1, pieces_player_2, colour), axis=2)
        return memory.Position(state, legal_actions)

    def next_empty_row(self, column, board):
        """ Finds the next empty field in a given column of the board and
        returns its row index.

        Args:
            column: Integer index of the considered column of the board.

        Returns:
            row: Integer row index of the first free field in the
                considered column.
        """
        board_height, _, _ = board.shape

        row = board_height - 1
        while not board[row, column, 0] == 0 and row > 0:
            row -= 1

        return row

    def update_legal_moves(self, row, column, simulation=False):
        """ Checks whether the column is full and changes the legal_moves
                attribute accordingly.

                Attributes:
                    row (int): The row index of the last move.
                    column (int): The column index of the last move.
                """
        if row == 0:
            if not simulation:
                self.legal_moves[column] = 0
            if simulation:
                self.legal_moves_simulation[column] = 0

    def is_winning_move(self, row, column, is_simulation=False):
        """ Returns True if the considered move wins the game.

        Counts the relevant adjacent that have the same piece color
        as the current player.

        Args:
            row: Integer row index of considered move.
            column: Integer column index of considered move.
        """

        if self.is_winning_vertically(row, column, is_simulation):
            return True

        if self.is_winning_horizontally(row, column, is_simulation):
            return True

        if self.is_winning_diagonally(row, column, is_simulation):
            return True

        return False

    def is_winning_vertically(self, row, column, is_simulation):
        """ Returns True if the current move creates a vertical connected series
        of at least four adjacent pieces of the same color.

        Args:
            row: Integer row index of current move.
            column: Integer column index of current move.

        Returns:
            A Boolean that returns True if the condition (see above) is met.
        """

        count_up = self.count_adjacent_pieces(row, column, 1, 0, is_simulation)
        count_down = self.count_adjacent_pieces(row, column, -1, 0, is_simulation)
        count_vertical = count_up + 1 + count_down

        if count_vertical > 3:
            return True

        return False

    def is_winning_horizontally(self, row, column, is_simulation):
        """ Returns True if the current move creates a horizontally connected
        series of at least four adjacent pieces of the same color.

        Args:
            row: Integer row index of current move.
            column: Integer column index of current move.

        Returns:
            A Boolean that returns True if the condition (see above) is met.
        """
        count_right = self.count_adjacent_pieces(row, column, 0, 1, is_simulation)
        count_left = self.count_adjacent_pieces(row, column, 0, -1, is_simulation)
        count_horizontally = count_left + 1 + count_right

        if count_horizontally > 3:
            return True

        return False

    def is_winning_diagonally(self, row, column, is_simulation):
        """ Returns True if the current move creates a horizontally connected
        series of at least four adjacent pieces of the same color.

        Args:
            row: Integer row index of current move.
            column: Integer column index of current move.

        Returns:
            A Boolean that returns True if the condition (see above) is met.
        """
        count_down_right = self.count_adjacent_pieces(row, column, 1, 1, is_simulation)
        count_up_left = self.count_adjacent_pieces(row, column, -1, -1, is_simulation)
        count_diagonally = count_down_right + 1 + count_up_left

        if count_diagonally > 3:
            return True

        count_down_left = self.count_adjacent_pieces(row, column, 1, -1, is_simulation)
        count_up_right = self.count_adjacent_pieces(row, column, -1, 1, is_simulation)
        count_diagonally = count_down_left + 1 + count_up_right

        if count_diagonally > 3:
            return True

        return False

    def count_adjacent_pieces(self, row, column, direction_row, direction_column, is_simulation):
        """ Counts adjacent pieces of the same player as the starting piece.

        Counts the adjacent pieces that have the same piece color
        as the current player in the given direction (starting from given position).

        Args:
            row: Integer row index current move.
            column: Integer column index of current move.
            direction_row: Integer (-1,0 or 1) giving the iteration steps for next_row.
            direction_column: Integer (-1,0 or 1) giving the iteration steps for next_column.

        Returns:
            count: Integer giving the number of adjacent pieces of the same color
                in the given direction.
        """
        board_height, board_width, _ = self.board.shape

        board = self.board[:, :, 0]

        if is_simulation:
            board = self.board_simulation[:, :, 0]

        count = 0
        next_row = row + direction_row
        next_column = column + direction_column
        condition_row = lambda next_row: True
        condition_column = lambda next_column: True

        if direction_row == -1:
            condition_row = lambda next_row: next_row >= 0
        if direction_row == 1:
            condition_row = lambda next_row: next_row < board_height

        if direction_column == -1:
            condition_column = lambda next_column: next_column >= 0
        if direction_column == 1:
            condition_column = lambda next_column: next_column < board_width

        while condition_row(next_row) & condition_column(next_column):
            if not board[next_row, next_column] == self.current_player:
                break
            count += 1
            next_row += direction_row
            next_column += direction_column

        return count