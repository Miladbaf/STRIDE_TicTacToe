import random
import time
from pydantic import BaseModel, Field, NonNegativeInt, validator
from typing import List, Optional

class State(BaseModel):
    board: List[Optional[str]] = Field(..., description="A list representing the TicTacToe board, containing 'X', 'O', or None")
    player: str = Field(..., description="Current player, either 'X' or 'O'")
    actions: List[NonNegativeInt] = Field(..., description="List of available action indices")
    action_schema: List[tuple] = Field(..., description="Schema describing valid actions")
    textual_descript: str = Field(..., description="Textual description of the current game state")
    current_agent: str = Field(..., description="Agent currently controlling the game")
    is_maximizing: bool = Field(..., description="Flag indicating if the current player is maximizing score")
    observation: str = Field(..., description="Observation from the current game state")
    is_done: bool = Field(..., description="Flag indicating if the game has ended")

    @validator('board')
    def validate_board(cls, v):
        if len(v) != 9:
            raise ValueError('Board must have exactly 9 positions')
        return v

    def is_valid_action(self, action):
        return action in self.actions

class TicTacToe:
    def __init__(self, name='TicTacToe', initial_board=None, starting_player='X', is_maximizing=True, log_file=None, working_memory=None):
        """Initialize the TicTacToe game with optional custom settings.
        
        Args:
            name (str): Name of the game.
            initial_board (list): Initial state of the board.
            starting_player (str): Player who starts the game, 'X' or 'O'.
            is_maximizing (bool): Player is maximizing its score or not.
        """
        self.name = name
        self.current_depth = 0
        self.initial_board = initial_board[:] if initial_board is not None else [None] * 9
        self.initial_player = starting_player
        self.board = self.initial_board[:]
        self.player = starting_player
        self.is_maximizing = is_maximizing
        self.is_done = False

        self.log_file = log_file
        self.working_memory = working_memory

        self.opponent_marker = 'O' if starting_player == 'X' else 'X'

        if is_maximizing:
            self.maximizing_player = starting_player
        else:
            self.maximizing_player = self.opponent_func(starting_player)

        self.init_prompt = ("==== USER ====\n")
        self.init_prompt += "I will play a game of TicTacToe. The goal is to align three of your marks in a horizontal, vertical, or diagonal row."

        is_maximizing = (self.player == self.maximizing_player)
        action = "maximizing" if is_maximizing else "minimizing"
        max_depth = len(self.get_available_moves())
        max_depth_player = self.initial_player if max_depth % 2 == 1 else self.opponent_func(self.initial_player)
        is_max = (max_depth_player == self.maximizing_player)
        depth_maxmin = "maximizing" if is_max else "minimizing"
        self.init_prompt += f" Board has {max_depth} empty positions so I should go to depth {max_depth} and recursively score the game states until I reach depth 1. At depth 0 player is {self.initial_player} and it is a {action} player. So at depth {max_depth} player is {max_depth_player} and it is a {depth_maxmin} player."
    
    description_of_problem_class = """
    Tic-Tac-Toe is a classic two-player game where players take turns marking spaces in a 3x3 grid, aiming to place three of their marks in a horizontal, vertical, or diagonal row to win.

    Components:
    Players: Two players, usually denoted as Player X and Player O.
    Board: A 3x3 grid where each cell can be empty, marked with an X, or marked with an O.
    Marks: Each player has a unique mark (X or O) that they place on the board.

    Interaction protocol:
    - Players take turns starting with Player X.
    - On each turn, a player marks an empty cell on the grid with their mark (X or O).
    - The game continues until a player has three of their marks in a horizontal, vertical, or diagonal row, or all cells are filled resulting in a draw.

    Rules:
    1. Players alternate turns, with Player X always going first.
    2. A player can only mark an empty cell.
    3. The game ends when one player achieves a row of three marks horizontally, vertically, or diagonally, or when all cells are filled with no winner (a draw).

    Goals of the players:
    - Player X: Maximize the chances of placing three X's in a row before Player O does.
    - Player O: Maximize the chances of placing three O's in a row before Player X does.

    Winning Conditions:
    - A player wins if they place three of their marks in a horizontal, vertical, or diagonal row.
    - If all cells are filled without any player achieving three marks in a row, the game results in a draw.

    Game Setup:
    1. The game begins with an empty 3x3 grid.
    2. Players decide who will be Player X and who will be Player O.
    3. Player X makes the first move.

    Example Board:
    - Initial State:
    . . .
    . . .
    . . .

    - After Player X places an X in the center:
    . . .
    . X .
    . . .

    - After Player O places an O in the top-left corner:
    O . .
    . X .
    . . .

    - And so on, until either player X wins, or player O wins, or a tie happens.

    End-game Examples:
    1. Example board for player X winning by having 3 markers in the 3rd row:
    O . X
    . O O
    X X X

    2. Example board for player O winning by having 3 markers in the diogonal:
    X X O
    . O .
    O X .

    3. Example board for a tie (no player winning) where no empty spots are left:
    O X O
    O X X
    X O X

    Objective:
    Each player aims to either achieve a row of three of their marks or to block the opponent from doing so. Strategic planning and anticipation of the opponent's moves are crucial to winning the game.
    """
    
    def check_agents(self, agents):
        """Check if agents initiated correctly."""
        return len(list(agents.keys())) == 1 and list(agents.keys())[0] == "agent"
    
    def write_to_log(self, message):
        """Write a message to the log file."""
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")

    def reset(self):
        """Reset the game to the initial state."""
        self.board = self.initial_board[:]
        self.player = self.initial_player
        self.opponent_marker = 'O' if self.player == 'X' else 'X'
        self.is_done = False
        self.state = State(
            board=self.initial_board[:],
            player=self.initial_player,
            actions=self.get_available_moves(),
            action_schema=[("action", "integer", "the action chosen by the player, which should be in {}.".format(self.get_available_moves()))],
            textual_descript=self.get_textual_description(),
            current_agent=self.player,
            is_maximizing=self.is_maximizing,
            observation=self.get_textual_description(),
            is_done=False
        )
        
    def get_textual_description(self):
        return f"Now you are going to play a game of Tic-tac-toe. The current state of the board is {self.board}. It is Player {self.player}'s turn. Your objective is to place three of your marks in a horizontal, vertical, or diagonal row to win, while preventing your opponent from doing the same."

    def opponent_func(self, player):
        """Return the opponent's marker."""
        return 'O' if player == 'X' else 'X'

    def switch_player(self):
        """Switch the current player and opponent."""
        self.player, self.opponent_marker = self.opponent_marker, self.player
    
    def get_state_struct(self):
        """Constructs and returns the current state of the game as a Pydantic model."""
        available_moves = self.get_available_moves()
        state = State(
            board=self.board,
            player=self.player,
            actions=available_moves,
            action_schema=[("action", "integer", f"the action chosen by the player, which should be in {self.get_available_moves()}.")],
            textual_descript=self.get_textual_description(),
            current_agent=self.player,
            is_maximizing=self.is_maximizing,
            observation=self.get_textual_description(),
            is_done=self.is_done
        ) 
        return state 

    def print_board(self):
        """Print the current state of the board."""
        for i in range(3):
            print(' ' + ' | '.join([str(self.board[3*i+j] if self.board[3*i+j] is not None else ' ')
                                    for j in range(3)]))
            if i < 2:
                print('---+---+---')

    def win(self, player):
        """Check if the specified player has won the game."""
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for condition in win_conditions:
            if all(self.board[i] == player for i in condition):
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
        if self.win(self.player) or self.win(self.opponent_marker) or all(spot is not None for spot in self.board):
            self.is_done = True
            return True
        return False

    def get_available_moves(self):
        """Return a list of available moves on the board."""
        return [i for i in range(9) if self.board[i] is None]

    def save_state(self):
        """Save the current game state."""
        return (tuple(self.board), self.player, self.opponent_marker)

    def restore_state(self, state):
        """Restore the game to a previous state."""
        self.board, self.player, self.opponent_marker = list(state[0]), state[1], state[2]

    def step(self, action):
        """Executes a move in the TicTacToe game, updates the game state, and returns the new state."""
        if self.board[action] is not None:
            raise ValueError("Invalid move")
        self.board[action] = self.player
        self.switch_player()
        self.state = State(
            board=self.board,
            player=self.player,
            actions=self.get_available_moves(),
            action_schema=[("action", "integer", f"the action chosen by the player, which should be in {self.get_available_moves()}.")],
            textual_descript=self.get_textual_description(),
            current_agent=self.player,
            is_maximizing=self.is_maximizing,
            observation=self.get_textual_description(),
            is_done=self.over()
        )
        return self.state

    def play_with_bfs_minimax(self, agent):
        """Play a game using the BFS minimax strategy provided by the agent."""
        start_time = time.time()
        max_depth = len(self.get_available_moves())
        agent.calculate_scores(max_depth)
        current_depth = 0
        agent.write_to_log("\nNow let's start to play the TicTacToe game and choose action based on the calculated scores.")
        while not self.over():
            current_state = tuple(self.board)
            action_score_tuples = agent.get_action_score_tuples(current_state)
            agent.write_to_log(f"Question: This is depth {current_depth} of the TicTacToe game. What action should I choose based on the current state of the game?")
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