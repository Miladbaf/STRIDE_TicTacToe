from pydantic import BaseModel, Field
from envs.tictactoe.tools import *

class AgentMemory(BaseModel):
    depth_scores: dict = Field(default_factory=dict, description="Stores scores calculated at different depths.")
    memo: dict = Field(default_factory=dict, description="Memoization of game state evaluations to optimize performance.")

class MinimaxAgent:
    def __init__(self, game, working_memory, player_marker, is_maximizing, log_file):
        """Initialize a Minimax agent for TicTacToe.

        Args:
            game: The TicTacToe game instance.
            working_memory: Dictionary containing game state information.
            player_marker (str): 'X' or 'O' representing the agent's marker.
            is_maximizing (bool): Player is maximizing its score or not.
            log_file (str): File path to write the log.
        """
        self.game = game
        self.player_marker = player_marker
        self.log_file = log_file
        self.is_maximizing = is_maximizing
        self.working_memory = working_memory

        self.initialize_log()

    def initialize_log(self):
        """Initialize the log file and write the initial log message."""
        with open(self.log_file, 'w') as f:
            f.write("")
        self.write_to_log(self.get_initial_log_message())

    def write_to_log(self, message):
        """Write a message to the log file."""
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")

    def get_initial_log_message(self):
        """Generate the initial log message based on game settings."""
        is_maximizing = self.game.player == self.game.maximizing_player
        action = "maximizing" if is_maximizing else "minimizing"
        max_depth = len(self.game.get_available_moves())
        max_depth_player = self.game.initial_player if max_depth % 2 == 1 else self.game.opponent_func(self.game.initial_player)
        depth_maxmin = "maximizing" if max_depth_player == self.game.maximizing_player else "minimizing"
        return (
            "==== USER ====\nI will play a game of TicTacToe. The goal is to align three of your marks in a horizontal, vertical, or diagonal row. I will use the Minimax algorithm to find the best move at each state of the game to play.\n"
            f"Before playing, I need to compute the scores corresponding to each depth via value iteration using the CalculateScores function. "
            f"Board has {max_depth} empty positions so I should go to depth {max_depth} and recursively score the game states until I reach depth 1. "
            f"At depth 0 player is {self.game.initial_player} and it is a {action} player. So at depth {max_depth} player is {max_depth_player} and it is a {depth_maxmin} player.\n"
        )
    
    def reset(self):
        """Reset the game to its initial state."""
        self.working_memory.memo.clear()
        self.working_memory.depth_scores.clear()
        self.game.reset()
    
    def get_available_moves(self):
        """Return a list of available moves on the board."""
        return [i for i in range(9) if self.game.board[i] is None]

    def score(self, player):
        """Evaluate the score for a player at a given game depth."""
        if self.game.win(player):
            return 1
        elif self.game.win(self.game.opponent_func(player)):
            return -1
        return 0

    def explore_states(self, current_depth, max_depth, depth_states, player, parent=None):
        """Recursive function to explore all possible game states and evaluate them."""
        op = ExploreStates(current_depth=current_depth, max_depth=max_depth, player=player, parent=parent)
        return op.execute(self, self.game, depth_states, self.working_memory)

    def calculate_scores(self, max_depth=9):
        """Compute scores for each state at each depth using the minimax algorithm with alpha-beta pruning."""
        op = CalculateScores(max_depth=max_depth)
        op.execute(game=self.game, working_memory=self.working_memory)

    def get_scores(self, depth):
        """Retrieve scores for the specified depth."""
        op = GetScores(depth=depth)
        return op.execute(self.game, self.working_memory)
    
    def get_action_score_tuples(self, state):
        """Retrieve action-score tuples for a given board state using precomputed minimax scores."""
        op = GetActionScoreTuples(state=state, player=self.game.player, opponent_marker=self.game.opponent_marker)
        return op.execute(self.working_memory)