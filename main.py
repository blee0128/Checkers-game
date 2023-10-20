"""Games or Adversarial Search (Chapter 5)"""

import copy
import itertools
import random
from collections import namedtuple
import time

import numpy as np

from utils import vector_add

GameState = namedtuple('GameState', 'to_move, utility, board, moves')
StochasticGameState = namedtuple('StochasticGameState', 'to_move, utility, board, moves, chance')


# ______________________________________________________________________________
# MinMax Search


def minmax_decision(state, game):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax_decision:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)))


# ______________________________________________________________________________


def expect_minmax(state, game):
    """
    [Figure 5.11]
    Return the best move for a player after dice are thrown. The game tree
	includes chance nodes along with min and max nodes.
	"""
    player = game.to_move(state)

    def max_value(state):
        v = -np.inf
        for a in game.actions(state):
            v = max(v, chance_node(state, a))
        return v

    def min_value(state):
        v = np.inf
        for a in game.actions(state):
            v = min(v, chance_node(state, a))
        return v

    def chance_node(state, action):
        res_state = game.result(state, action)
        if game.terminal_test(res_state):
            return game.utility(res_state, player)
        sum_chances = 0
        num_chances = len(game.chances(res_state))
        for chance in game.chances(res_state):
            res_state = game.outcome(res_state, chance)
            util = 0
            if res_state.to_move == player:
                util = max_value(res_state)
            else:
                util = min_value(res_state)
            sum_chances += util * game.probability(chance)
        return sum_chances / num_chances

    # Body of expect_minmax:
    return max(game.actions(state), key=lambda a: chance_node(state, a), default=None)


def alpha_beta_search(state, game):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_search:
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action

# The code was slidely modified to avoid having the AI run the whole game completely by itself
# created a new Checkers_game object the runs in this function alone to only provide a single move
# by the AI
def alpha_beta_cutoff_search(ostate, ogame, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    newboard = copy.deepcopy(ostate.board)
    game = Checkers_Game(newboard, ostate.to_move)
    state = game.initial
    player = game.to_move(ostate)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or (lambda state, depth: depth > d or game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action


# ______________________________________________________________________________
# Players for Games

# This is also slidely updated to avoid getting error if a user type in another choice outside the
# given options
def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: ")
    temp = game.actions(state)
    for count, i in enumerate(temp):
        print(count, i)
    print("")
    move = None
    if game.actions(state):
        check = True
        while check:
            move_string = input('Your move? ')
            try:
                x = int(move_string)
                if x not in range(0, len(temp)):
                    print("Invalid move, Try again.")
                else:
                    move = temp[int(move_string)]
                    check = False
            except ValueError:
                print("Invalid move, Try again.")

    else:
        print('no legal moves: passing turn to next player')

    return move

def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


def alpha_beta_player(game, state):
    return alpha_beta_search(state, game)


def minmax_player(game,state):
    return minmax_decision(state,game)


def expect_minmax_player(game, state):
    return expect_minmax(state, game)


# ______________________________________________________________________________
# Some Sample Games


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))

# created this Checkers_Game class based on the Game class
class Checkers_Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def __init__(self, board = [
            ["_", "o", "_", "o", "_", "o", "_", "o"],
            ["o", "_", "o", "_", "o", "_", "o", "_"],
            ["_", "o", "_", "o", "_", "o", "_", "o"],
            ["_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_"],
            ["x", "_", "x", "_", "x", "_", "x", "_"],
            ["_", "x", "_", "x", "_", "x", "_", "x"],
            ["x", "_", "x", "_", "x", "_", "x", "_"]
            ], to_move = 1
            ):
        self.board = board
        moves = self.get_all_moves(board,to_move)
        self.initial = GameState(to_move=to_move, utility=0, board=board, moves=moves)

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        return state.moves

    def result(self, state, move):
        """Return the state that results from making a move from a state."""

        newboard = self.action(state.board, move)
        # switch player to return the state
        if state.to_move == 1:
            to_move = 2
        else:
            to_move = 1

        utility_score = self.compute_utility(newboard, to_move)
        new_state = GameState(to_move = to_move, utility=utility_score, board=newboard, moves=self.get_all_moves(newboard,to_move))
        return new_state

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return state.utility != 0 or len(state.moves) == 0


    def compute_utility(self, board, player):
        """If player 1 wins with this move, return 1; if player 2 wins return -1; else return 0."""
        temp = self.get_all_moves(board, player)
        number_of_moves = len(temp)
        if number_of_moves == 0:
            # The reason player 1 will return -1 is because player 1 does not have any possible moves.
            # Therefore player 2 wins and will return -1
            if player == 1:
                return -1
            else:
                return +1
        else:
            return 0


    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        board = state.board

        for i in board:
            print(i)
        print()

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self):
        """Play an 2-person, move-alternating game."""
        """This method will have 3 different choice to play the game. The use can choose which options
        For the experiment, AI vs AI was used to quickly evaluate the performance of the game"""

        print("Choose 1 for player vs AI")
        print("Choose 2 for AI vs AI")
        print("Choose 3 for player vs player")

        check = True
        while check:
            x = input('Your choice? ')
            try:
                temp = int(x)
                if temp in range(1, 4):
                    check = False
                else:
                    print("Invalid move, Try again.")
            except ValueError:
                print("Invalid move")

        if temp == 1:
            state = self.initial
            turn = 0
            while True:
                for i in range(0, 2):
                    if i == 0:
                        print("Player turn")
                        move = query_player(self,state)
                        state = self.result(state, move)
                        if self.terminal_test(state):
                            self.display(state)
                            return self.utility(state, self.to_move(self.initial))
                    if i == 1:
                        print("AI turn")
                        move = alpha_beta_cutoff_search(state,self)
                        temp = self.actions(state)
                        for count, x in enumerate(temp):
                            print(count, x)
                        print("selected move: " + str(move))
                        state = self.result(state, move)
                        if self.terminal_test(state):
                            self.display(state)
                            return self.utility(state, self.to_move(self.initial))
        elif temp == 2:
            state = self.initial
            player1_time = []
            player2_time = []
            depth = 5
            while True:
                for i in range(0, 2):
                    if i == 0:
                        print("AI 1 turn")
                        self.display(state)
                        begin = self.milli_time()
                        # eval_fn=self.evaluate_function_1(state)
                        move = alpha_beta_cutoff_search(state, self, depth)
                        end = self.milli_time()
                        total_time = end - begin
                        player1_time.append(total_time)
                        temp = self.actions(state)
                        if len(temp) > 1:
                            print("Available moves: Select 0 to {}".format(len(temp) - 1))
                        elif len(temp) == 1:
                            print("Available moves: Select 0")
                        for count, x in enumerate(temp):
                            print(count, x)
                        print("selected move: " + str(move))
                        print("")
                        state = self.result(state, move)
                        if self.terminal_test(state):
                            self.display(state)
                            print("Depth :" + str(depth))
                            player_1_total = 0
                            for q in player1_time:
                                player_1_total += q
                            player_1_avg = player_1_total / len(player1_time)
                            player_2_total = 0
                            for r in player2_time:
                                player_2_total += r
                            player_2_avg = player_2_total / len(player2_time)
                            total_move = len(player1_time)+len(player2_time)
                            print("Number of moves: " + str(total_move))
                            print("Player 1 average decision time: " + str(player_1_avg) + "milliseconds")
                            print("Player 2 average decision time: " + str(player_2_avg) + "milliseconds")
                            return self.utility(state, self.to_move(self.initial))
                    if i == 1:
                        print("AI 2 turn")
                        self.display(state)
                        begin1 = self.milli_time()
                        move = alpha_beta_cutoff_search(state, self, depth)
                        end1 = self.milli_time()
                        total_time1 = end1 - begin1
                        player2_time.append(total_time1)
                        temp = self.actions(state)
                        if len(temp) > 1:
                            print("Available moves: Select 0 to {}".format(len(temp) - 1))
                        elif len(temp) == 1:
                            print("Available moves: Select 0")
                        for count, x in enumerate(temp):
                            print(count, x)
                        print("selected move: " + str(move))
                        print("")
                        state = self.result(state, move)
                        if self.terminal_test(state):
                            self.display(state)
                            print("Depth :" + str(depth))
                            player_1_total = 0
                            for q in player1_time:
                                player_1_total += q
                            player_1_avg = player_1_total / len(player1_time)
                            player_2_total = 0
                            for r in player2_time:
                                player_2_total += r
                            player_2_avg = player_2_total / len(player2_time)
                            total_move = len(player1_time) + len(player2_time)
                            print("Number of moves: " + str(total_move))
                            print("Player 1 average decision time: " + str(player_1_avg) + "milliseconds")
                            print("Player 2 average decision time: " + str(player_2_avg) + "milliseconds")
                            return self.utility(state, self.to_move(self.initial))


        elif temp == 3:
            state = self.initial
            turn = 0
            while True:
                for i in range(0, 2):
                    if i == 0:
                        print("Player 1 turn")
                        move = query_player(self,state)
                        state = self.result(state, move)
                        if self.terminal_test(state):
                            self.display(state)
                            return self.utility(state, self.to_move(self.initial))
                    if i == 1:
                        print("Player 2 turn")
                        move = query_player(self, state)
                        state = self.result(state, move)
                        if self.terminal_test(state):
                            self.display(state)
                            return self.utility(state, self.to_move(self.initial))
        else:
            print("Invalid input")

    def milli_time(self):
        """This method convert seconds to milliseconds"""
        return round(time.time() * 1000)

    def simple_move(self, board, y, x):
        """This method will see if the piece can perform a simple move given the piece row and column in the board"""
        # W is o in king mode
        # V is x in king mode
        list = []
        up = y - 1
        down = y + 1
        right = x + 1
        left = x - 1

        if board[y][x] == "o" or board[y][x] == "W" or board[y][x] == "V":
            if down < 8:
                if right < 8 and left >= 0:
                    if board[down][left] == "_":
                        list.append([(y, x), (down, left)])
                    if board[down][right] == "_":
                        list.append([(y, x), (down, right)])
                elif right > 7:
                    if board[down][left] == "_":
                        list.append([(y, x), (down, left)])
                elif left < 0:
                    if board[down][right] == "_":
                        list.append([(y, x), (down, right)])

        if board[y][x] == "x" or board[y][x] == "W" or board[y][x] == "V":
            if up >= 0:
                if right < 8 and left >= 0:
                    if board[up][left] == "_":
                        list.append([(y, x), (up, left)])
                    if board[up][right] == "_":
                        list.append([(y, x), (up, right)])
                elif right > 7:
                    if board[up][left] == "_":
                        list.append([(y, x), (up, left)])
                elif left < 0:
                    if board[up][right] == "_":
                        list.append([(y, x), (up, right)])
        return list

    def jump(self, board, y, x):
        """This method will see if the piece can perform a jump given the piece row and column in the board"""
        # W is o in king mode
        # V is x in king mode

        list = []
        up = y - 1
        down = y + 1
        right = x + 1
        left = x - 1
        upjump = y - 2
        downjump = y + 2
        rightjump = x + 2
        leftjump = x - 2

        if board[y][x] == "o":
            if downjump < 8:
                if rightjump > 7:
                    if (board[down][left] == "x" and board[downjump][leftjump] == "_") or (
                            board[down][left] == "V" and board[downjump][leftjump] == "_"):
                        list.append([(y, x), (downjump, leftjump)])
                elif leftjump < 0:
                    if (board[down][right] == "x" and board[downjump][rightjump] == "_") or (
                            board[down][right] == "V" and board[downjump][rightjump] == "_"):
                        list.append([(y, x), (downjump, rightjump)])
                else:
                    if (board[down][left] == "x" and board[downjump][leftjump] == "_") or (
                            board[down][left] == "V" and board[downjump][leftjump] == "_"):
                        list.append([(y, x), (downjump, leftjump)])
                    if (board[down][right] == "x" and board[downjump][rightjump] == "_") or (
                            board[down][right] == "V" and board[downjump][rightjump] == "_"):
                        list.append([(y, x), (downjump, rightjump)])

        if board[y][x] == "x":
            if upjump >= 0:
                if rightjump > 7:
                    if (board[up][left] == "o" and board[upjump][leftjump] == "_") or (
                            board[up][left] == "W" and board[upjump][leftjump] == "_"):
                        list.append([(y, x), (upjump, leftjump)])
                elif leftjump < 0:
                    if (board[up][right] == "o" and board[upjump][rightjump] == "_") or (
                            board[up][right] == "W" and board[upjump][rightjump] == "_"):
                        list.append([(y, x), (upjump, rightjump)])
                else:
                    if (board[up][left] == "o" and board[upjump][leftjump] == "_") or (
                            board[up][left] == "W" and board[upjump][leftjump] == "_"):
                        list.append([(y, x), (upjump, leftjump)])
                    if (board[up][right] == "o" and board[upjump][rightjump] == "_") or (
                            board[up][right] == "W" and board[upjump][rightjump] == "_"):
                        list.append([(y, x), (upjump, rightjump)])

        if board[y][x] == "W":
            if downjump < 8:
                if rightjump > 7:
                    if (board[down][left] == "x" and board[downjump][leftjump] == "_") or (
                            board[down][left] == "V" and board[downjump][leftjump] == "_"):
                        list.append([(y, x), (downjump, leftjump)])
                elif leftjump < 0:
                    if (board[down][right] == "x" and board[downjump][rightjump] == "_") or (
                            board[down][right] == "V" and board[downjump][rightjump] == "_"):
                        list.append([(y, x), (downjump, rightjump)])
                else:
                    if (board[down][left] == "x" and board[downjump][leftjump] == "_") or (
                            board[down][left] == "V" and board[downjump][leftjump] == "_"):
                        list.append([(y, x), (downjump, leftjump)])
                    if (board[down][right] == "x" and board[downjump][rightjump] == "_") or (
                            board[down][right] == "V" and board[downjump][rightjump] == "_"):
                        list.append([(y, x), (downjump, rightjump)])

            if upjump >= 0:
                if rightjump > 7:
                    if (board[up][left] == "x" and board[upjump][leftjump] == "_") or (
                            board[up][left] == "V" and board[upjump][leftjump] == "_"):
                        list.append([(y, x), (upjump, leftjump)])
                elif leftjump < 0:
                    if (board[up][right] == "x" and board[upjump][rightjump] == "_") or (
                            board[up][right] == "V" and board[upjump][rightjump] == "_"):
                        list.append([(y, x), (upjump, rightjump)])
                else:
                    if (board[up][left] == "x" and board[upjump][leftjump] == "_") or (
                            board[up][left] == "V" and board[upjump][leftjump] == "_"):
                        list.append([(y, x), (upjump, leftjump)])
                    if (board[up][right] == "x" and board[upjump][rightjump] == "_") or (
                            board[up][right] == "V" and board[upjump][rightjump] == "_"):
                        list.append([(y, x), (upjump, rightjump)])

        if board[y][x] == "V":
            if downjump < 8:
                if rightjump > 7:
                    if (board[down][left] == "o" and board[downjump][leftjump] == "_") or (
                            board[down][left] == "W" and board[downjump][leftjump] == "_"):
                        list.append([(y, x), (downjump, leftjump)])
                elif leftjump < 0:
                    if (board[down][right] == "o" and board[downjump][rightjump] == "_") or (
                            board[down][right] == "W" and board[downjump][rightjump] == "_"):
                        list.append([(y, x), (downjump, rightjump)])
                else:
                    if (board[down][left] == "o" and board[downjump][leftjump] == "_") or (
                            board[down][left] == "W" and board[downjump][leftjump] == "_"):
                        list.append([(y, x), (downjump, leftjump)])
                    if (board[down][right] == "o" and board[downjump][rightjump] == "_") or (
                            board[down][right] == "W" and board[downjump][rightjump] == "_"):
                        list.append([(y, x), (downjump, rightjump)])

            if upjump >= 0:
                if rightjump > 7:
                    if (board[up][left] == "o" and board[upjump][leftjump] == "_") or (
                            board[up][left] == "W" and board[upjump][leftjump] == "_"):
                        list.append([(y, x), (upjump, leftjump)])
                elif leftjump < 0:
                    if (board[up][right] == "o" and board[upjump][rightjump] == "_") or (
                            board[up][right] == "W" and board[upjump][rightjump] == "_"):
                        list.append([(y, x), (upjump, rightjump)])
                else:
                    if (board[up][left] == "o" and board[upjump][leftjump] == "_") or (
                            board[up][left] == "W" and board[upjump][leftjump] == "_"):
                        list.append([(y, x), (upjump, leftjump)])
                    if (board[up][right] == "o" and board[upjump][rightjump] == "_") or (
                            board[up][right] == "W" and board[upjump][rightjump] == "_"):
                        list.append([(y, x), (upjump, rightjump)])

        return list


    # player 1 is "o" or "W"
    # player 2 is "x" or "V"
    def get_all_moves(self, board, player):
        """This will provide all possible move by the specific player at their turn.
        The method will enumerate the entire board and look for all the moves. Will
        start by checking for jump. If no jump, then will check for simple move. If there
        exist a jump, will check for multijump"""
        moves = []

        for i in range(len(board)):
            for j in range(len(board[i])):
                if player == 1:
                    if board[i][j] == "o" or board[i][j] == "W":
                        jumptemp = self.jump(board, i, j)
                        if len(jumptemp) > 0:
                            for k in jumptemp:
                                moves.append(k)
                if player == 2:
                    if board[i][j] == "x" or board[i][j] == "V":
                        jumptemp = self.jump(board, i, j)
                        if len(jumptemp) > 0:
                            for k in jumptemp:
                                moves.append(k)

        queue = []
        if len(moves) > 0:
            for p in moves:
                queue.append(p)
            while len(queue) != 0:
                e = queue.pop(0)
                tempboard = copy.deepcopy(board)
                q = e
                t = q[0]
                r = q[1]
                y1 = t[0]
                x1 = t[1]
                y2 = r[0]
                x2 = r[1]
                tempboard[y2][x2] = tempboard[y1][x1]
                temp = self.jump(tempboard, y2, x2)
                if len(temp) > 0:
                    g = temp[-1]
                    c = g[-1]
                    q.append(c)
                    moves.append(q)
                    moves.remove(e)

        if len(moves) < 1:
            for i in range(len(board)):
                for j in range(len(board[i])):
                    if player == 1:
                        if board[i][j] == "o" or board[i][j] == "W":
                            temp = self.simple_move(board, i, j)
                            if len(temp) > 0:
                                for k in temp:
                                    moves.append(k)
                    if player == 2:
                        if board[i][j] == "x" or board[i][j] == "V":
                            temp = self.simple_move(board, i, j)
                            if len(temp) > 0:
                                for k in temp:
                                    moves.append(k)
        return moves


    def action(self, board, actionlist):
        """This will update the board based on the move given by the user. It will check if it is a simple move, jump or multijump and
        remove the pieces if a jump or multijump occurs. If any piece reach the other side, this function will convert it to a King"""
        if len(actionlist) == 2:
            initial = actionlist[0]
            final = actionlist[-1]
            row_diff = abs(final[0] - initial[0])
            col_diff = abs(final[1] - initial[1])
            row_mid = int((final[0] + initial[0]) / 2)
            col_mid = int((final[1] + initial[1]) / 2)

            if row_diff == 1:
                temp = board[initial[0]][initial[1]]
                board[initial[0]][initial[1]] = "_"
                board[final[0]][final[1]] = temp
            else:
                temp = board[initial[0]][initial[1]]
                mid_temp = board[row_mid][col_mid]
                if temp == "x" or temp == "V":
                    if mid_temp == "W":
                        board[initial[0]][initial[1]] = "_"
                        board[row_mid][col_mid] = "_"
                        board[final[0]][final[1]] = "V"
                    else:
                        board[initial[0]][initial[1]] = "_"
                        board[row_mid][col_mid] = "_"
                        board[final[0]][final[1]] = temp
                if temp == "o" or temp == "W":
                    if mid_temp == "V":
                        board[initial[0]][initial[1]] = "_"
                        board[row_mid][col_mid] = "_"
                        board[final[0]][final[1]] = "W"
                    else:
                        board[initial[0]][initial[1]] = "_"
                        board[row_mid][col_mid] = "_"
                        board[final[0]][final[1]] = temp

            for i in range(len(board[0])):
                if board[0][i] == "x":
                    board[0][i] = "V"

            for i in range(len(board[0])):
                if board[7][i] == "o":
                    board[7][i] = "W"

        elif len(actionlist) == 3:
            initial = actionlist[0]
            middle = actionlist[1]
            final = actionlist[-1]
            row_mid1 = int((middle[0] + initial[0]) / 2)
            col_mid1 = int((middle[1] + initial[1]) / 2)
            row_mid2 = int((middle[0] + final[0]) / 2)
            col_mid2 = int((middle[1] + final[1]) / 2)
            temp1 = board[initial[0]][initial[1]]
            mid_temp1 = board[row_mid1][col_mid1]
            if temp1 == "x" or temp1 == "V":
                if mid_temp1 == "W":
                    board[initial[0]][initial[1]] = "_"
                    board[row_mid1][col_mid1] = "_"
                    board[middle[0]][middle[1]] = "V"
                else:
                    board[initial[0]][initial[1]] = "_"
                    board[row_mid1][col_mid1] = "_"
                    board[middle[0]][middle[1]] = temp1
            if temp1 == "o" or temp1 == "W":
                if mid_temp1 == "V":
                    board[initial[0]][initial[1]] = "_"
                    board[row_mid1][col_mid1] = "_"
                    board[middle[0]][middle[1]] = "W"
                else:
                    board[initial[0]][initial[1]] = "_"
                    board[row_mid1][col_mid1] = "_"
                    board[middle[0]][middle[1]] = temp1

            temp2 = board[middle[0]][middle[1]]
            mid_temp2 = board[row_mid2][col_mid2]
            if temp2 == "x" or temp2 == "V":
                if mid_temp2 == "W":
                    board[middle[0]][middle[1]] = "_"
                    board[row_mid2][col_mid2] = "_"
                    board[final[0]][final[1]] = "V"
                else:
                    board[middle[0]][middle[1]] = "_"
                    board[row_mid2][col_mid2] = "_"
                    board[final[0]][final[1]] = temp2
            if temp2 == "o" or temp2 == "W":
                if mid_temp2 == "V":
                    board[middle[0]][middle[1]] = "_"
                    board[row_mid2][col_mid2] = "_"
                    board[final[0]][final[1]] = "W"
                else:
                    board[middle[0]][middle[1]] = "_"
                    board[row_mid2][col_mid2] = "_"
                    board[final[0]][final[1]] = temp2

        for i in range(len(board[0])):
            if board[0][i] == "x":
                board[0][i] = "V"

        for i in range(len(board[0])):
            if board[7][i] == "o":
                board[7][i] = "W"

        return board


if __name__ == '__main__':
    newgame = Checkers_Game()
    newgame.play_game()


