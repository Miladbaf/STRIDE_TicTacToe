from pydantic import BaseModel, Field
from typing import Dict, Tuple, List, Optional
import re

from envs.connectn.env import *

tool_names = ["CalculateScoresN", "GetScoresN"]

class InitializeLog(BaseModel):
    game: object = Field(..., description="The TicTacToe game instance")

    def execute(self, log_file):
        """Initialize the game log file with the starting log message."""
        is_maximizing = self.game.player == self.game.maximizing_player
        action = "maximizing" if is_maximizing else "minimizing"
        max_depth = len(self.game.get_available_moves())
        max_depth_player = self.game.initial_player if max_depth % 2 == 1 else self.game.opponent_func(self.game.initial_player)
        depth_maxmin = "maximizing" if max_depth_player == self.game.maximizing_player else "minimizing"
        
        with open(log_file, 'a') as f:
            f.write(
                f"==== USER ====\nI will play a game of TicTacToe. The goal is to align three of your marks in a horizontal, vertical, or diagonal row. I will use the Minimax algorithm to find the best move at each state of the game to play.\n"
                f"Before playing, I need to compute the scores corresponding to each depth via value iteration using the CalculateScores function. "
                f"Board has {max_depth} empty positions so I should go to depth {max_depth} and recursively score the game states until I reach depth 1. "
                f"At depth 0 player is {self.game.initial_player} and it is a {action} player. So at depth {max_depth} player is {max_depth_player} and it is a {depth_maxmin} player.\n"
            )

class ExploreStates(BaseModel):
    current_depth: int = Field(..., description="Current depth in the minimax recursion")
    max_depth: int = Field(..., description="Maximum depth in the minimax recursion")
    player: str = Field(..., description="The player ('X' or 'O') whose turn is currently being evaluated")
    alpha: float = Field(default=float('-inf'), description="Alpha value for alpha-beta pruning")
    beta: float = Field(default=float('inf'), description="Beta value for alpha-beta pruning")
    parent: Optional[Tuple] = Field(default=None, description="Parent state key")

    def execute(self, game, depth_states, working_memory):
        state_key = game.save_state()
        memo_key = (state_key, self.current_depth, self.player)
        memo = working_memory.memo 

        if memo_key in memo:
            return memo[memo_key]

        if self.current_depth == self.max_depth or game.over():
            score = game.score(self.player)
            depth_states[self.current_depth][state_key] = {'score': score, 'children': {}, 'player': game.player, 'parent': self.parent}
            memo[memo_key] = state_key
            return state_key

        if state_key not in depth_states[self.current_depth]:
            depth_states[self.current_depth][state_key] = {'children': {}, 'player': game.player, 'parent': self.parent}

        available_moves = game.get_available_moves()
        for move in available_moves:
            saved_state = game.save_state()
            game.board[move] = game.player
            game.switch_player()
            child_state_key = self.__class__(
                current_depth=self.current_depth + 1, max_depth=self.max_depth, player=self.player,
                alpha=self.alpha, beta=self.beta, parent=state_key
            ).execute(game=game, depth_states=depth_states, working_memory=working_memory)
            depth_states[self.current_depth][state_key]['children'][move] = {'state_key': child_state_key, 'move': move}

            if 'score' in depth_states[self.current_depth + 1][child_state_key]:
                child_score = depth_states[self.current_depth + 1][child_state_key]['score']
                if game.maximizing_player == self.player:
                    self.alpha = max(self.alpha, child_score)
                else:
                    self.beta = min(self.beta, child_score)

            game.restore_state(saved_state)
            if self.beta <= self.alpha:
                break

        memo[memo_key] = state_key
        return state_key

class CalculateScoresN(BaseModel):
    max_depth: int = Field(..., description="Maximum depth in the minimax recursion")

    def execute(self, game, working_memory):
        """Execute score calculation across depths."""
        if len(working_memory.depth_scores) == 10 or len(working_memory.depth_scores) == game.rows*game.columns + 1:
            return "Call function GetScoresN now! Starting from depth 0 and going down until reaching an end game situation! No need to call CalculateScores again."
        depth_states = {depth: {} for depth in range(self.max_depth + 1)}
        ExploreStates(current_depth=0, max_depth=self.max_depth, player=game.player).execute(game=game, depth_states=depth_states, working_memory=working_memory)

        logs = []
        for depth in range(self.max_depth - 1, -1, -1):
            current_player = game.player if depth % 2 == 0 else game.opponent_marker
            is_maximizing = (current_player == game.maximizing_player)
            action = "maximizing" if is_maximizing else "minimizing"
            log_entry = f"Thought: At Depth {depth + 1}, it is player {current_player}'s turn who is a {action} player."
            logs.append(log_entry)

            for state_key, state_val in depth_states[depth].items():
                if state_val['children']:
                    child_scores = [depth_states[depth + 1][child_info['state_key']]['score'] for move, child_info in state_val['children'].items()]
                    best_score = max(child_scores) if is_maximizing else min(child_scores)
                    depth_states[depth][state_key]['score'] = best_score

            logs.append(f"Operation: Called function CalculateScores with depth {depth + 1}, is_maximizing {is_maximizing}, and current player {current_player}.")
            logs.append(f"Result: Scores calculated at depth {depth + 1}.")

        for log in logs:
            game.write_to_log(log)

        game.write_to_log("Exit: I have reached depth 1. Scores for all depths have been computed.")
        working_memory.depth_scores = {depth: {k: {'score': v['score'], 'parent': v['parent'], 'children': v['children']} for k, v in depth_states[depth].items()} for depth in depth_states}
        
        return "Call function GetScoresN now! Starting from depth 0 and going down until reaching an end game situation! No need to call CalculateScores again."

class GetScoresN(BaseModel):
    depth: int = Field(..., description="Depth for which scores need to be retrieved")

    def execute(self, game, working_memory):
        state_key = tuple(game.board)
        action_score_tuples = GetActionScoreTuples(state=state_key, player=game.player, opponent_marker=game.opponent_marker).execute(working_memory=working_memory)
        valid_moves = game.get_available_moves()
        action_score_tuples = [(action, score) for action, score in action_score_tuples if action in valid_moves]
        action_score_tuples = sorted([(action, score) for action, score in action_score_tuples if action in valid_moves], key=lambda x: x[0])
        if action_score_tuples:
            act = f"Operation: Called function GetScoresN to retrieve the scores for all actions at the current state and depth {self.depth}."
            obs = (
                f"The list of scores for all available actions presented as (action,score) pairs is: {[(move, score) for move, score in action_score_tuples]}. I should absolutely choose an action and return a response with this format 'I am player 'O' and I am a minimizing player so I choose action 'n' which has the minimum score and set thought.exit to True.' After this you should call GetScoresN for the next depth and change the player from 'X' to 'O' or vice-versa. Pay attention to maximizing and minimizing players."
            )
            
            return act, obs 
        else:            
            return f"Operation: Called function GetScoresN to retrieve the scores for all actions at the current state and depth {self.depth}.", "Observation: No valid moves available."

class FindDepthForState(BaseModel):
    state_key: Tuple = Field(..., description="State key for which depth needs to be found")

    def execute(self, working_memory):
        for depth, states in working_memory.depth_scores.items():
            if self.state_key in states:
                return depth
        return None

class SaveState(BaseModel):
    def execute(self, game, working_memory):
        return (tuple(working_memory['board']), working_memory['player'], game.opponent_func(working_memory['player']))

class ParseAction(BaseModel):
    content: str = Field(..., description="Content string from which to parse the action")

    def execute(self, game):
        legal_actions = game.get_available_moves()
        match = re.search(r'(?<=action )\d+', self.content)
        if match:
            action = int(match.group())
            if action in legal_actions:
                return action   
            else:
                return 0
        else:
            return 0
    
class GetActionScoreTuples(BaseModel):
    state: Tuple = Field(..., description="The current state")
    player: str = Field(..., description="The current player")
    opponent_marker: str = Field(..., description="The opponent's marker")

    def execute(self, working_memory):
        state_key = (self.state, self.player, self.opponent_marker)
        op = FindDepthForState(state_key=state_key)
        depth_found = op.execute(working_memory=working_memory)
        if depth_found is None:
            print("State not found in the game tree.")
            return []
    
        children_info = working_memory.depth_scores[depth_found][state_key]['children']
        action_score_tuples = [(move, working_memory.depth_scores[depth_found + 1][child_info['state_key']]['score'])
                               for move, child_info in children_info.items()]

        return action_score_tuples