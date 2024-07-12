from envs.connectn.tools import *
from pydantic import BaseModel, Field

class AgentMemory(BaseModel):
    depth_scores: dict = Field(default_factory=dict, description="Stores scores calculated at different depths.")
    memo: dict = Field(default_factory=dict, description="Memoization dictionary for storing previously computed results.")

class MinimaxBFSAgent:
    def __init__(self, game, working_memory, player_marker, is_maximizing, log_file):
        """Initialize a Minimax agent for Connect-N.

        Args:
            game: The Connect-N game instance.
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
        is_maximizing = self.game.player == self.game.maximizing_player
        action = "maximizing" if is_maximizing else "minimizing"
        max_depth = len(self.game.get_empty())
        max_depth_player = self.game.player if max_depth % 2 == 1 else self.game.opponent_func(self.game.player)
        depth_maxmin = "maximizing" if max_depth_player == self.game.maximizing_player else "minimizing"
        return (
            "==== USER ====\nI will play a game of Connect-N. The goal is to align {} of your marks in a row, either horizontally, vertically, or diagonally. I will use a Minimax algorithm to find the best move at each state of the game.\n"
            f"Before playing, I need to compute the scores corresponding to each depth via value iteration using the CalculateScores function. "
            f"Board has {max_depth} empty positions so I should go to depth {max_depth} and recursively score the game states until I reach depth 1. "
            f"At depth 0 player is {self.game.player} and it is a {action} player. So at depth {max_depth} player is {max_depth_player} and it is a {depth_maxmin} player.\n".format(self.game.n_in_row)
        )
    
    def reset(self):
        """Reset the game to its initial state."""
        self.working_memory.memo.clear()
        self.working_memory.depth_scores.clear()
        self.game.reset()

    def score(self, player):
        """Evaluate the score for a player at a given game depth."""
        if self.game.win(player):
            return 1
        elif self.game.win(self.game.opponent_func(player)):
            return -1
        return 0
    
    def get_ordered_moves(self):
        """Order available moves to prioritize center moves."""
        available_moves = self.game.get_available_moves()
        center_column = self.game.columns // 2
        ordered_moves = sorted(available_moves, key=lambda x: abs(x - center_column))
        return ordered_moves
    
    def explore_states(self, current_depth, max_depth, depth_states, player, parent=None):
        """Recursive function to explore all possible game states and evaluate them."""
        op = ExploreStates(current_depth=current_depth, max_depth=max_depth, player=player, parent=parent)
        return op.execute(self, self.game, depth_states, self.working_memory)

    def calculate_scores(self, max_depth=9):
        """Compute scores for each state at each depth using the minimax algorithm with alpha-beta pruning."""
        op = CalculateScoresN(max_depth=max_depth)
        op.execute(game=self.game, working_memory=self.working_memory)

    def get_scores(self, depth):
        """Retrieve scores for the specified depth."""
        op = GetScoresN(depth=depth)
        return op.execute(self.game, self.working_memory)
    
    def get_action_score_tuples(self, state):
        """Retrieve action-score tuples for a given board state using precomputed minimax scores."""
        op = GetActionScoreTuples(state=state, player=self.game.player, opponent_marker=self.game.opponent_marker)
        return op.execute(self.working_memory)