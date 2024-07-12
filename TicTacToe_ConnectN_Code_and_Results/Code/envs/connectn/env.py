import random
import time
from pydantic import BaseModel, Field, NonNegativeInt, validator
from typing import List, Optional

from pydantic import BaseModel, Field, validator
from typing import List, Optional
from pydantic.types import NonNegativeInt

class State(BaseModel):
    board: List[Optional[str]] = Field(..., description="A list representing the Connect-N board, containing 'X', 'O', or None")
    player: str = Field(..., description="Current player, either 'X' or 'O'")
    actions: List[NonNegativeInt] = Field(..., description="List of available action indices that are empty")
    action_schema: List[tuple] = Field(..., description="Schema describing valid actions for Connect-N")
    textual_descript: str = Field(..., description="Textual description of the current game state")
    current_agent: str = Field(..., description="Agent currently controlling the game")
    is_maximizing: bool = Field(..., description="Flag indicating if the current player is maximizing score")
    observation: str = Field(..., description="Observation from the current game state")
    is_done: bool = Field(..., description="Flag indicating if the game has ended")

    def is_valid_action(self, action: int) -> bool:
        """Ensure the action is valid only if the position is empty."""
        return action in self.actions and self.board[action] is None

class ConnectN:
    def __init__(self, name, rows, columns, n_in_row, starting_player='X', is_maximizing=True, log_file=None, working_memory=None):
        """Initialize the Connect-N game with custom settings.
        
        Args:
            rows (int): Number of rows in the board.
            columns (int): Number of columns in the board.
            n_in_row (int): Number of consecutive marks needed to win.
            starting_player (str): Player who starts the game, 'X' or 'O'.
            is_maximizing (bool): Player is maximizing its score or not.
        """
        self.name = name
        self.current_depth = 0
        self.rows = rows
        self.columns = columns
        self.n_in_row = n_in_row
        self.board = [None] * (rows * columns)
        self.starting_player = starting_player
        self.player = starting_player
        self.is_maximizing = is_maximizing
        self.is_done = False

        self.log_file = log_file
        self.working_memory = working_memory

        self.opponent_marker = 'O' if starting_player == 'X' else 'X'
        self.maximizing_player = starting_player if is_maximizing else self.opponent_func(starting_player)

        self.init_prompt = ("==== USER ====\n")
        self.init_prompt += "I will play a game of ConnectN. The goal is to align three of your marks in a horizontal, vertical, or diagonal row."

        is_maximizing = (self.player == self.maximizing_player)
        action = "maximizing" if is_maximizing else "minimizing"
        max_depth = len(self.get_empty())
        max_depth_player = self.starting_player if max_depth % 2 == 1 else self.opponent_func(self.starting_player)
        is_max = (max_depth_player == self.maximizing_player)
        depth_maxmin = "maximizing" if is_max else "minimizing"
        self.init_prompt += f" Board has {max_depth} empty positions so I should go to depth {max_depth} and recursively score the game states until I reach depth 1. At depth 0 player is {self.starting_player} and it is a {action} player. So at depth {max_depth} player is {max_depth_player} and it is a {depth_maxmin} player."
    
    description_of_problem_class = """
    ConnectN is a generalized version of Connect4, where two players alternate turns dropping colored discs into a vertically suspended grid. The objective is to form a horizontal, vertical, or diagonal line of N discs. The game introduces a gravity effect where discs drop to the lowest available position within a column, adding a unique strategic dimension to the gameplay.

    Components:
    - Players: Two players, typically referred to as Player X and Player O, who use different colored discs.
    - Board: A grid with configurable dimensions, with number of columns and number of rows given.
    - Discs: Each player has an ample supply of discs in their respective discs (X or O).

    Interaction protocol:
    - Players take turns starting with Player X.
    - On each turn, a player chooses a position in board to drop a disc into. The disc falls, affected by gravity, to the lowest available position within the column that empty position is in.
    - The game continues until a player forms a line of N discs in a row (horizontally, vertically, or diagonally) or the board is completely filled, resulting in a draw.

    Rules:
    1. Players must alternate turns, with Player X always going first.
    2. A player can only choose a position which is in a column that has available space.
    3. The game ends when one player forms a line of N discs or when all positions are filled without any player achieving this, which results in a draw.

    Goals of the players:
    - Player X: Strategize to connect N of their discs in a row vertically, horizontally, or diagonally before Player O.
    - Player O: Similarly, aim to connect N of their discs in a row while blocking Player X's attempts.

    Winning Conditions:
    - A player wins by aligning N of their discs in a row in any direction. (vertical, horizontal or diagonal)
    - The game results in a draw if the entire board is filled without either player achieving N in a row.

    Game Setup:
    1. The game starts with an empty board of the chosen dimensions.
    2. Player X makes the first move by dropping a disc into one of the available empty positions.

    Example Board:
    - Initial State for a 4x4 board (as used in Connect4):
    . . . .
    . . . .
    . . . .
    . . . .

    - After Player X places a disc in position number 13, affected by gravity:
    . . . .
    . . . .
    . . . .
    . X . .

    - After Player O places a disc in the position 12:
    . . . .
    . . . .
    . . . .
    O X . .

    - And so on, until either Player X or Player O connects N discs in a row, or the board is filled resulting in a draw.

    End-game Examples:
    1. Example board for Player X winning with a vertical line in the first column:
    X . . .
    X . O .
    X . O .
    X . O .

    2. Example board for Player X winning with a diagonal line:
    . X O X
    X O X O
    O X X O
    X O O X

    3. Example board for a tie where no empty spots are left and no lines of N are formed:
    X O X X
    O X X O
    X O X O
    O X O O

    Objective:
    Each player aims to strategically drop their discs in one of the available positions to form a line of N while preventing their opponent from doing the same. Anticipating the opponent's moves and effectively using the gravity-affected gameplay are critical to securing a victory.
    """
    
    def check_agents(self, agents):
        """Check if agents initiated correctly."""
        return len(list(agents.keys())) == 1 and list(agents.keys())[0] == "agent"
    
    def write_to_log(self, message):
        """Write a message to the log file."""
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
    
    def reset(self):
        """Reset the game to its initial state."""
        self.board = [None] * (self.rows * self.columns)
        self.player = self.starting_player  
        self.is_done = False  
        self.opponent_marker = 'O' if self.player == 'X' else 'X'  
        self.maximizing_player = self.player if self.is_maximizing else self.opponent_func(self.player)
        self.state = State(
            player=self.starting_player,
            actions=self.get_available_moves(),
            current_agent=self.starting_player,
            is_maximizing=self.is_maximizing,
            observation=self.get_textual_description(),
            action_schema=[("action", "integer", "the action chosen by the player, which should be in {}.".format(self.get_available_moves()))],
            board=[None] * (self.rows * self.columns),
            is_done=False,
            textual_descript=self.get_textual_description()
        )

    def get_textual_description(self):
        return (f"Now you are going to play a game of Connect-N, where two players alternate turns dropping "
                f"colored discs into a vertically suspended grid. The objective is to form a line of {self.n_in_row} "
                f"discs in a row, either horizontally, vertically, or diagonally. The current state of the board is "
                f"{self.board}, the current player is Player {self.player}, the number of discs required to win is "
                f"{self.n_in_row}. Your objective is to "
                f"strategically drop your discs to form a line of {self.n_in_row} discs while preventing your opponent "
                f"from doing the same. CalculateScoresN with max_depth={self.rows*self.columns}.")

    def opponent_func(self, player):
        """Return the opponent's marker."""
        return 'O' if player == 'X' else 'X'

    def switch_player(self):
        """Switch the current player and opponent."""
        self.player, self.opponent_marker = self.opponent_marker, self.player
    
    def get_state_struct(self):
        """Get the state of the game with specified parameters."""
        state = State(
            player=self.player,
            actions=self.get_available_moves(),
            board=self.board,
            current_agent=self.player,
            is_maximizing=self.is_maximizing,
            observation=self.get_textual_description(),
            action_schema=[("action", "integer", "the action chosen by the player, which should be in {}.".format(self.get_available_moves()))],            
            is_done=self.is_done,
            textual_descript=self.get_textual_description()
        ) 
        return state 

    def print_board(self):
        """Print the current state of the board."""
        for i in range(self.rows):
            print(' ' + ' | '.join([str(self.board[self.columns*i+j] if self.board[self.columns*i+j] is not None else ' ')
                                    for j in range(self.columns)]))
            if i < self.rows - 1:
                print('---+' * self.columns)
                
    def win(self, player):
        """Check if the specified player has won the game by achieving n_in_row marks in a line."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.rows):
            for c in range(self.columns):
                if self.board[r * self.columns + c] != player:
                    continue 
                for dr, dc in directions:
                    end_row = r + (self.n_in_row - 1) * dr
                    end_col = c + (self.n_in_row - 1) * dc
                    if 0 <= end_row < self.rows and 0 <= end_col < self.columns:
                        if all(self.board[(r + i * dr) * self.columns + (c + i * dc)] == player for i in range(self.n_in_row)):
                            return True
        return False
    
    def score(self, player):
        """Evaluate the score for a player at a given game depth."""
        if self.win(player):
            return 1
        elif self.win(self.opponent_func(player)):
            return -1
        return 0

    def over(self):
        """Check if the game is over."""
        self.is_done = self.win(self.player) or self.win(self.opponent_marker) or all(spot is not None for spot in self.board)
        return self.is_done

    def get_empty(self):
        """Return a list of all empty positions on the board."""
        return [i for i in range(self.rows * self.columns) if self.board[i] is None]
    
    def get_available_moves(self):
        """Return a list of the bottom-most available moves in each column."""
        available_moves = []
        for c in range(self.columns):
            for r in range(self.rows - 1, -1, -1):
                if self.board[r * self.columns + c] is None:
                    available_moves.append(r * self.columns + c)
                    break
        return available_moves
    
    def save_state(self):
        """Save the current game state in Connect-N."""
        return (tuple(self.board), self.player, self.opponent_marker)

    def restore_state(self, state):
        """Restore the game to a previous state in Connect-N."""
        self.board, self.player, self.opponent_marker = list(state[0]), state[1], state[2]
        self.maximizing_player = self.player if self.is_maximizing else self.opponent_func(self.player)

    def step(self, action):
        if self.board[action] is not None:
            raise ValueError("Invalid move")
        self.board[action] = self.player
        self.switch_player()
        self.update_state()
        self.state = State(
            player=self.starting_player,
            actions=self.get_available_moves(),
            board=self.board,
            current_agent=self.player,
            is_maximizing=self.is_maximizing,
            observation=self.get_textual_description(),
            action_schema=[("action", "integer", "the action chosen by the player, which should be in {}.".format(self.get_available_moves()))],            
            is_done=self.over(),
            textual_descript=self.get_textual_description()
        ) 
        return self.state

    def play_with_bfs_minimax(self, agent):
        """Play a game using the BFS minimax strategy provided by the agent."""
        start_time = time.time()
        max_depth = len(self.get_empty())
        agent.calculate_scores(max_depth)
        current_depth = 0
        agent.write_to_log("\nNow let's start to play the Connect-N game and choose action based on the calculated scores.")
        while not self.over():
            current_state = tuple(self.board)
            action_score_tuples = agent.get_action_score_tuples(current_state)
            valid_moves = self.get_available_moves()
            action_score_tuples = [(action, score) for action, score in action_score_tuples if action in valid_moves]
            agent.write_to_log(f"Question: This is depth {current_depth} of the Connect-N game. What action should I choose based on the current state of the game?")
            if action_score_tuples:
                is_maximizing = (self.player == self.maximizing_player)
                best_score = max(action_score_tuples, key=lambda x: x[1])[1] if is_maximizing else min(action_score_tuples, key=lambda x: x[1])[1]
                best_moves = [action for action, score in action_score_tuples if score == best_score]
                best_move = random.choice(best_moves)
                agent.write_to_log(f"Thought: I should first call GetScores to retrieve the scores for each action in depth {current_depth + 1}. It is my turn and I am a {'maximizing' if is_maximizing else 'minimizing'} player so I should choose the score with {'maximum' if is_maximizing else 'minimum'} score (break the tie randomly if there are multiple {'maximum' if is_maximizing else 'minimum'} scores).")
                agent.write_to_log(f"Operation: called function GetScores to retrieve the scores for all actions at the current state and depth {current_depth + 1}.")
                agent.write_to_log(f"Result: The list of scores for all available actions presented as (action,score) pairs is: {[(move, score) for move, score in action_score_tuples]}.")
                agent.write_to_log(f"Exit: I am a {'maximizing' if is_maximizing else 'minimizing'} player therefore, I should choose action {best_move}.")
                self.board[best_move] = self.player
                agent.write_to_log(f"Board is now: {self.board}")
                print(f"Player {self.player}'s turn to move to position {best_move + 1}.")
                self.print_board()
                if self.over():
                    break
                self.switch_player()
                current_depth += 1
            else:
                agent.write_to_log("No valid moves left.")
                print("No valid moves left.")
                break
        if self.win('X'):
            print("Player X wins!")
            result = "player X wins!"
        elif self.win('O'):
            print("Player O wins!")
            result = "player O wins!"
        else:
            print("It's a tie!")
            result = "a tie!"
        agent.write_to_log("Result: Game is over.")
        agent.write_to_log(f"Over: Game is over and the result is {result} I should exit the program.")

        end_time = time.time()  
        duration = end_time - start_time  
        print(f"Duration of run is: {duration}")