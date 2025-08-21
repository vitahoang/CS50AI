"""
Tic Tac Toe Player
"""

import math
import operator


X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # Initialize counters for X and O
    x = 0
    o = 0
    # Count the number of X's and O's on the board
    for row in board:
        for column in row:
            if column == X:
                x += 1
            if column == O:
                o += 1
    # Determine the current player based on counts
    if x == o == 0:
        return X
    if x > o:
        return O
    return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()  # Set to hold available actions
    for row in range(len(board)):
        for column in range(len(board[row])):
            if board[row][column] == EMPTY:
                actions.add((row, column))
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Explicit bounds checking for action indices
    row, col = action
    if not (0 <= row <= len(board) and 0 <= col <= len(board[row])):
        raise ValueError(
            f"Invalid action: row and column indices must be between 0 and 2, got ({row}, {col})")
    if board[row][col] is not EMPTY:
        raise ValueError("Invalid action: Cell already occupied")

    new_board = [row[:] for row in board]
    new_board[row][col] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    w = 3  # Number of consecutive marks needed to win

    def check_streak(row, col):
        # No streak found
        if board[row][col] == EMPTY:
            return None
        
        current_player = board[row][col]

        # check horizontal streaks
        for i in range(w):
            if col + i >= len(board[row]) \
                    or board[row][col + i] != current_player:
                break
            if i == w - 1:
                return current_player
        # check vertical streaks
        for i in range(w):
            if row + i >= len(board) \
                    or board[row + i][col] != current_player:
                break
            if i == w - 1:
                return current_player
        # check diagonal streaks
        if row + w <= len(board) and col + w <= len(board[row]):
            if all(board[row + i][col + i] == current_player for i in range(w)):
                return current_player
        if row + w <= len(board) and col - w >= -1:
            if all(board[row + i][col - i] == current_player for i in range(w)):
                return current_player
        return None

    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] == EMPTY:
                continue
            winner_player = check_streak(row, col)
            if winner_player is not None:
                return winner_player
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    actions_available = actions(board)
    if len(actions_available) == 0:
        return True
    if winner(board) is not None:
        return True

    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    if win == X:
        return 1
    elif win == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    current = player(board)
    best_score = -math.inf if current == X else math.inf
    best_action = None

    def max_value(board):
        if terminal(board):
            return utility(board)
        v = -math.inf
        for action in actions(board):
            v = max(v, min_value(result(board, action)))
        return v


    def min_value(board):
        if terminal(board):
            return utility(board)
        v = math.inf
        for action in actions(board):
            v = min(v, max_value(result(board, action)))
        return v

    less_than = operator.lt
    greater_than = operator.gt
    minx_maxo = min_value if current == X else max_value
    compare = greater_than if current == X else less_than

    for action in actions(board):
        score = minx_maxo(result(board, action))
        if compare(score, best_score):
            best_score = score
            best_action = action
    return best_action