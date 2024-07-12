from rich.console import Console
from envs.env_helper import get_env_param
from agents.StriDe import StriDeAgent
from agents.Direct import DirectAgent
from agents.DirectCodeInterpreter import DirectCodeInterpreterAgent
import re
import numpy as np
from copy import deepcopy
import json
import random

direct_code_mdp = """
Let me check the contents of the working memory file to identify the correct keys for the transition matrix and the reward matrix.# Load the working memory file to check its content
with open(working_memory_file, "r") as file:
    working_memory_content = json.load(file)

# Display the keys stored in the working memory
working_memory_content.keys()

output >

dict_keys(['P', 'R'])

assistant > The working memory file contains keys 'P' and 'R' for the transition matrix and the reward matrix, respectively. Let's proceed with extracting these matrices and then compute the optimal policy for the finite-horizon tabular Markov Decision Process.import numpy as np

# Extracting the transition matrix P and reward matrix R from the working memory
transition_matrix_P = np.array(working_memory_content["P"])
reward_matrix_R = np.array(working_memory_content["R"])

# Displaying the shapes of transition matrix P and reward matrix R
transition_matrix_shape = transition_matrix_P.shape
reward_matrix_shape = reward_matrix_R.shape

transition_matrix_shape, reward_matrix_shape

output >

((10, 10, 10), (10, 10))

assistant > The transition matrix P has a shape of (10, 10, 10), and the reward matrix R has a shape of (10, 10).

Now, we will proceed with computing the optimal policy for the finite-horizon tabular Markov Decision Process with a horizon length of 5, number of states |S|=10, and number of actions |A|=10. Let's calculate the optimal policy based on the given transition matrix P and reward matrix R using Dynamic Programming methods.def calculate_optimal_policy(transition_matrix, reward_matrix, horizon_length):
    num_states, num_actions, _ = transition_matrix.shape
    optimal_policy = np.zeros((horizon_length, num_states), dtype=int)
    max_values = np.zeros((horizon_length, num_states))

    # Initialize the value function for the last time step
    max_values[-1] = reward_matrix.max(axis=1)

    # Perform dynamic programming to calculate the optimal policy
    for h in range(horizon_length - 2, -1, -1):
        for s in range(num_states):
            values = reward_matrix[s] + np.dot(transition_matrix[s], max_values[h + 1])
            optimal_policy[h, s] = np.argmax(values)
            max_values[h, s] = values[optimal_policy[h, s]]

    return optimal_policy

# Calculate the optimal policy for the given Markov Decision Process
optimal_policy = calculate_optimal_policy(transition_matrix_P, reward_matrix_R, 5)
optimal_policy

output >

array([[3, 7, 4, 7, 9, 8, 3, 5, 6, 5],
       [3, 7, 4, 7, 9, 8, 3, 5, 6, 5],
       [3, 7, 4, 7, 9, 8, 3, 5, 6, 5],
       [3, 7, 4, 8, 9, 8, 3, 5, 6, 5],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

assistant > The optimal policy for the finite-horizon tabular Markov Decision Process with a horizon length of 5 has been calculated. Here is the optimal action to take at each step for each state:

- At time step 0, optimal actions: [3, 7, 4, 7, 9, 8, 3, 5, 6, 5]
- At time step 1, optimal actions: [3, 7, 4, 7, 9, 8, 3, 5, 6, 5]
- At time step 2, optimal actions: [3, 7, 4, 7, 9, 8, 3, 5, 6, 5]
- At time step 3, optimal actions: [3, 7, 4, 8, 9, 8, 3, 5, 6, 5]
- At time step 4, optimal actions: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Each row corresponds to a time step (0 to 4), and each value in a row corresponds to the optimal action to take in the respective state at that time step.

This is time step 1, the current state is 8, and the available actions are [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
Question: Now which action the agent should take?
assistant > code_interpreter

# Define the current state and the available actions
current_state = 8
available_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Get the optimal action to take at time step 1 for the current state
optimal_action_index = optimal_policy[1, current_state]
optimal_action_index

output >

6

assistant > Based on the optimal policy calculated for time step 1 and the current state of 8, the action index that the agent should take is 6. Therefore, the agent should take action with index 6 from the available actions [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].

"""

direct_code_bargain_single = """
### Steps to Compute the SPE:
1. **Define the Problem**:
   - Players: Buyer (B) and Seller (S)
   - Values: Buyer's value = 1, Seller's value = 0
   - Discount Factors: \(\delta_b = 0.55\), \(\delta_s = 0.56\)
   - Deadline: \(T = 8\)

2. **Backward Induction Methodology**:
   - Starting at \(t = T\) and working backward to \(t = 1\).
   - At each \(t\), determine the optimal price \(p_t\) that the offering player should propose, which depends on the utility the other agent will get if the game continues to \(t+1\).

3. **Initial Conditions at \(T\) (when \(t = T\))**:
   - Since we assume Buyer makes the first offer, the last offering player is the seller as T=8 is even. Seller would offer \(p_T = 1\), which the buyer will accept because both accepting and rejecting lead to 0 utility.

4. **Iterative Calculation for Previous Time Steps**:
   - At each \(t < T\), calculate the offer \(p_t\) based on maximizing the proposer's utility while making sure the other agent gets a utility no less than the amount he would get if he waits for the next time step.

5. **Python Code Implementation**:
   - Implement this logic in a Python function to compute \(p_t\) and utilities recursively for each \(t\) from \(T\) down to 1.

Let's implement this process in Python.

def backward_onestep(agent, opponent_util_if_rej, time_step, buyer_discount, seller_discount):
    if agent == "buyer":
        # opponent is seller
        oppo_discount = seller_discount
    else:
        # opponent is buyer
        oppo_discount = buyer_discount
    # compute opponent's share of the surplus 1 based on his utility
    oppo_share_if_rej = opponent_util_if_rej / (oppo_discount**time_step)
    # compute agent's maximal share at current time step that can still incentive opponent to accept
    my_share_now = 1.0 - oppo_share_if_rej * oppo_discount
    # calculate the price corresponding to my_share_now
    if agent == "buyer":
        price = 1.0 - my_share_now
    else:
        price = my_share_now
    return price

    
def compute_spe(T, delta_b, delta_s):
    utilities_b = {T+1: 0}  # Buyer's utility if no agreement is reached by deadline
    utilities_s = {T+1: 0}  # Seller's utility if no agreement is reached by deadline
    prices = {}

    for t in range(T, 0, -1):
        if t % 2 == 1:  # Buyer's turn to offer
            price = backward_onestep("buyer", utilities_s[t+1], t, delta_b, delta_s)
            utilities_b[t] = (1 - price) * delta_b ** (t-1)
            utilities_s[t] = price * delta_s ** (t-1)
        else:  # Seller's turn to offer
            price = backward_onestep("seller", utilities_b[t+1], t, delta_b, delta_s)
            utilities_s[t] = price * delta_s ** (t-1)
            utilities_b[t] = (1 - price) * delta_b ** (t-1)

        prices[t] = price

    return prices, utilities_b, utilities_s

# Recompute the SPE with the corrected order of players
prices, utilities_b, utilities_s = compute_spe(T, delta_b, delta_s)
prices

The SPE prices are
{8: 1.0,
  7: 0.56,
  6: 0.758,
  5: 0.42447999999999997,
  4: 0.683464,
  3: 0.38273983999999994,
  2: 0.660506912,
  1: 0.36988387072000006}
"""

direct_code_tic_tac_toe = """
### Steps to Compute the Best Move:
1. **Define the Problem**:
   - Players: X and O
   - Goal: Align three of their marks in a horizontal, vertical, or diagonal row.
   - Board: A 3x3 grid.

2. **Minimax Algorithm with Depth Handling**:
   - Use the minimax algorithm to calculate the best move by considering future potential moves by the opponent.
   - Start at the current game state and evaluate all possible moves recursively.

3. **Python Code Implementation**:
   - Implement this logic in a Python function that uses recursion and backtracking to evaluate the best move.

```python
def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    if winner or depth == 0:
        return score(winner)

    if is_maximizing:
        best_score = float('-inf')
        for move in get_available_moves(board):
            make_move(board, move, 'X')
            score = minimax(board, depth - 1, False)
            undo_move(board, move)
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for move in get_available_moves(board):
            make_move(board, move, 'O')
            score = minimax(board, depth - 1, True)
            undo_move(board, move)
            best_score = min(score, best_score)
        return best_score

# Calculate the best move for 'X' at the start of the game
initial_board = [None] * 9
best_move = None
best_score = float('-inf')
for move in get_available_moves(initial_board):
    make_move(initial_board, move, 'X')
    score = minimax(initial_board, 8, False)  # Start at depth 8
    undo_move(initial_board, move)
    if score > best_score:
        best_score = score
        best_move = move

best_move
"""

direct_code_connect_n = """
### Steps to Compute the Best Move:
1. **Define the Problem**:
   - Players: Two players alternately drop their markers into columns.
   - Goal: Align N markers in a row, horizontally, vertically, or diagonally.
   - Board: A grid typically larger than 3x3 (e.g., 6x7 for Connect4).

2. **Gravity-affected Gameplay**:
   - Players choose a column to drop their piece, and it falls to the lowest available position in that column.

3. **Minimax Algorithm Adaptation for ConnectN**:
   - Apply a minimax algorithm that handles gravity by filling the lowest empty space in a column.
   - Evaluate all potential future game states considering opponent moves.

3. **Python Code Implementation**:
```python
def minimax_connectn(board, depth, is_maximizing, n_in_row, player):
    winner = check_winner(board, n_in_row)
    if winner or depth == 0 or is_draw(board):
        return score(winner)

    if is_maximizing:
        best_score = float('-inf')
        for col in get_available_columns(board):
            row = get_lowest_empty_row(board, col)
            make_move(board, row, col, player)
            score = minimax_connectn(board, depth - 1, False, n_in_row, switch_player(player))
            undo_move(board, row, col)
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for col in get_available_columns(board):
            row = get_lowest_empty_row(board, col)
            make_move(board, row, col, player)
            score = minimax_connectn(board, depth - 1, True, n_in_row, switch_player(player))
            undo_move(board, row, col)
            best_score = min(score, best_score)
        return best_score

# Initialize the ConnectN board with a specific size and n_in_row
board_size = (6, 7)  # Example size for Connect4
connectn_board = [None] * (board_size[0] * board_size[1])
n_in_row = 4  # Example for Connect4
current_player = 'X'  # Assume 'X' starts the game
best_move = None
best_score = float('-inf')
for col in get_available_columns(connectn_board):
    row = get_lowest_empty_row(connectn_board, col)
    make_move(connectn_board, row, col, current_player)
    score = minimax_connectn(connectn_board, 8, False, n_in_row, switch_player(current_player))
    undo_move(connectn_board, row, col)
    if score > best_score:
        best_score = score
        best_move = col  # Note we return the column as the move

best_move
"""

class Logger(object):
    def __init__(self, log_file, verbose=True):
        # self.terminal = sys.stdout
        self.console = Console(record=True)
        # self.log = open(log_file, "w")
        self.log_file = log_file
        self.verbose = verbose

        self.write("All outputs written to %s" % log_file)
        return 

    def write(self, message, color=None):
        # self.log.write(message + "\n")
        self.console.save_html(self.log_file, clear=False)
        if(self.verbose): 
            if color is not None:
                self.console.print("[{}]".format(color)+message+"[/{}]".format(color))
            else:
                self.console.print(message)

def load_initial_instructions(path_to_instructions):
    """Load initial instructions from textual format to a python dict"""
    pattern = r"==== (SYSTEM|USER|ASSISTANT) ===="

    # Use re.split to split the string by the pattern
    with open(path_to_instructions) as f:
        content = f.read()
        content = re.split(pattern, content)
        content_ = []
        for c in content: 
            if(c != ""): content_.append(c)
        content = content_
        l = len(content)
        assert(l % 2 == 0)
        initial_instruction = []
        for i in range(0, l, 2):
            instruction = {"role": content[i].strip().lower().replace("====", "").replace(" ", "").strip(), 
                           "content": content[i+1].strip()
                           }
            initial_instruction.append(instruction)
    return initial_instruction

def create_env(env_name, player_turn, is_maximizing=True, rows=3, columns=3, n_in_row=3, random_param=False, mdp_known=False, logger=None):
    if env_name == "tabular_mdp":
        from envs.tabular_mdp.env import TabularMDP
        return TabularMDP(env_param=get_env_param(env_name, random_param), mdp_known=mdp_known)
    elif env_name == "bargain_alternate_singleissue":
        from envs.bargain_alternate_singleissue.env import BargainAlternateSingleIssue
        return BargainAlternateSingleIssue(env_param=get_env_param(env_name, random_param))
    elif env_name == "bargain_onesided_uncertainty":
        from envs.bargain_onesided_uncertainty.env import BargainOneSidedUncertainty
        return BargainOneSidedUncertainty(env_param=get_env_param(env_name, random_param))
    elif env_name == "TicTacToe":
        from envs.tictactoe.program_agent import AgentMemory
        working_memory = AgentMemory(
                    memo=dict(),
                    depth_scores=dict()
                )
        from envs.tictactoe.env import TicTacToe
        env_param=get_env_param(env_name, False, board=None, player_turn='X', is_maximizing=True)
        return TicTacToe(initial_board=env_param['game_board'], starting_player=env_param['player_turn'], is_maximizing=env_param['is_maximizing'], log_file=logger, working_memory=working_memory)
    elif env_name == "ConnectN":
        from envs.connectn.program_agent import AgentMemory
        working_memory = AgentMemory(
                    memo=dict(),
                    depth_scores=dict()
                )
        from envs.connectn.env import ConnectN
        env_param=get_env_param(env_name, False, None, player_turn, is_maximizing, rows, columns, n_in_row)
        return ConnectN('ConnectN', rows=env_param['rows'], columns=env_param['columns'], n_in_row=env_param['n_in_row'], starting_player=env_param['player_turn'], is_maximizing=env_param['is_maximizing'], log_file=logger, working_memory=working_memory)

def create_agents(env, logger, agent_type, agent_engine, rows=3, columns=3, n_in_row=3, mdp_known=True, game=None):
    if env.name == "tabular_mdp" and mdp_known:
        demo = load_initial_instructions("envs/tabular_mdp/prompts/vi_exmps.txt")
        from envs.tabular_mdp.tools import tool_names
        init_memory = {"P":env.P, 
                       "R":env.R, 
                       "nState":env.nState, 
                       "nAction":env.nAction, 
                       "epLen":env.epLen,
                       "V": np.zeros((env.epLen,env.nState)),
                       "Q": np.zeros((env.epLen,env.nState,env.nAction)),
                       }
        if agent_type == "stride":
            agent = StriDeAgent(problem_description=env.description_of_problem_class, demo=demo, tool_names=tool_names, init_memory=init_memory, logger=logger, engine=agent_engine)
            # agent = StriDeAgent(problem_description=env.description_of_problem_class, demo=[], tool_names=tool_names, init_memory=init_memory, logger=logger, engine=agent_engine)
        elif agent_type == "direct":
            agent = DirectAgent(problem_description=env.description_of_problem_class, logger=logger,engine=agent_engine)
        elif agent_type == "direct-code":
            # store P and R to working_memory.json
            with open("agents/working_memory.json", "w") as outfile: 
                json.dump({"P":env.P.tolist(), "R":env.R[:,:,0].tolist()}, outfile)
            agent = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=direct_code_mdp, read_file=True, logger=logger,engine=agent_engine)
            # agent = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=None, read_file=True,logger=logger,engine=agent_engine)
        else:
            raise ValueError("invalid agent_type {}".format(agent_type))
        return {"agent":agent}
    elif env.name == "tabular_mdp" and not mdp_known:
        if agent_type == "stride":
            demo = load_initial_instructions("envs/tabular_mdp/prompts/ucbvi_exmps.txt")
            R_init = np.zeros((env.nState, env.nAction, 2))
            P_init = np.ones((env.nState, env.nAction, env.nState)) * 1.0 / env.nState
            assert R_init.shape == env.R.shape
            assert P_init.shape == env.P.shape
            bonus_scale_factor = 0.1
            init_memory = {"P":P_init, "R":R_init, 
                            "nState":env.nState, "nAction":env.nAction, "epLen":env.epLen,
                            "V": np.zeros((env.epLen,env.nState)),
                            "Q": np.zeros((env.epLen,env.nState,env.nAction)),
                            "Nsa": np.ones((env.nState, env.nAction)),
                            "bonus_scale_factor": bonus_scale_factor,
                            "epNum":75,
                            }
            from envs.tabular_mdp.tools import tool_names_unknown
            agent = StriDeAgent(problem_description=env.description_of_problem_class, demo=demo, tool_names=tool_names_unknown, init_memory=init_memory, logger=logger, engine=agent_engine)
        elif agent_type == "direct":
            agent = DirectAgent(problem_description=env.description_of_problem_class, logger=logger,engine=agent_engine)
        elif agent_type == "direct-code":
            # store P and R to working_memory.json
            with open("agents/working_memory.json", "w") as outfile: 
                json.dump({"P":env.P.tolist(), "R":env.R[:,:,0].tolist()}, outfile)
            agent = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=direct_code_mdp, read_file=True, logger=logger,engine=agent_engine)
            # agent = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=None, read_file=True,logger=logger,engine=agent_engine)
        else:
            raise ValueError("invalid agent_type {}".format(agent_type))
        return {"agent":agent}
    elif env.name == "bargain_alternate_singleissue":
        if agent_type == "stride":
            buyer_demo = load_initial_instructions("envs/bargain_alternate_singleissue/prompts/buyer_exmps.txt")
            seller_demo = load_initial_instructions("envs/bargain_alternate_singleissue/prompts/seller_exmps.txt")
            from envs.bargain_alternate_singleissue.tools import tool_names
            working_memory = {"T":env.T, "delta_b":env.buyerDiscount, "delta_s":env.sellerDiscount,
                            "SPEPrice": {},
                            }
            buyer = StriDeAgent(problem_description=env.description_of_problem_class, demo=buyer_demo, tool_names=tool_names, init_memory=deepcopy(working_memory), llm_validator=False, logger=logger, engine=agent_engine)
            seller = StriDeAgent(problem_description=env.description_of_problem_class, demo=seller_demo, tool_names=tool_names, init_memory=deepcopy(working_memory), llm_validator=False, logger=logger, engine=agent_engine)
            agents = {"buyer":buyer, "seller":seller}
            return agents
        elif agent_type == "direct":
            buyer = DirectAgent(problem_description=env.description_of_problem_class, logger=logger,engine=agent_engine)
            seller = DirectAgent(problem_description=env.description_of_problem_class, logger=logger,engine=agent_engine)
            agents = {"buyer":buyer, "seller":seller}
            return agents
        elif agent_type == "direct-code":
            # buyer = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=None, logger=logger,engine=agent_engine)
            # seller = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=None, logger=logger,engine=agent_engine)
            buyer = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=direct_code_bargain_single, read_file=False, logger=logger,engine=agent_engine)
            seller = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=direct_code_bargain_single, read_file=False, logger=logger,engine=agent_engine)
            agents = {"buyer":buyer, "seller":seller}
            return agents
        elif agent_type == "stride-direct-code":
            # let these two bargain with each other
            num = random.random()
            if num >= 0.5:
                buyer_demo = load_initial_instructions("envs/bargain_alternate_singleissue/prompts/buyer_exmps.txt")
                from envs.bargain_alternate_singleissue.tools import tool_names
                working_memory = {"T":env.T, "delta_b":env.buyerDiscount, "delta_s":env.sellerDiscount,
                                "SPEPrice": {},
                                }
                buyer = StriDeAgent(problem_description=env.description_of_problem_class, demo=buyer_demo, tool_names=tool_names, init_memory=deepcopy(working_memory), llm_validator=False, logger=logger, engine=agent_engine)
                seller = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=direct_code_bargain_single, read_file=False, logger=logger,engine=agent_engine)
            else:
                seller_demo = load_initial_instructions("envs/bargain_alternate_singleissue/prompts/seller_exmps.txt")
                from envs.bargain_alternate_singleissue.tools import tool_names
                working_memory = {"T":env.T, "delta_b":env.buyerDiscount, "delta_s":env.sellerDiscount,
                                "SPEPrice": {},
                                }
                seller = StriDeAgent(problem_description=env.description_of_problem_class, demo=seller_demo, tool_names=tool_names, init_memory=deepcopy(working_memory), llm_validator=False, logger=logger, engine=agent_engine)
                buyer = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=direct_code_bargain_single, read_file=False, logger=logger,engine=agent_engine)
            agents = {"buyer":buyer, "seller":seller}
            return agents
    elif env.name == "bargain_onesided_uncertainty":
        if agent_type == "stride":
            buyer_demo = load_initial_instructions("envs/bargain_onesided_uncertainty/prompts/buyer_exmps.txt")
            seller_demo = load_initial_instructions("envs/bargain_onesided_uncertainty/prompts/seller_exmps.txt")
            from envs.bargain_onesided_uncertainty.tools import tool_names
            # compute the constant c for all time steps
            c = np.zeros(env.T+1)
            for i in reversed(range(1, env.T+1)):
                if i == env.T:
                    c[i] = 0.5
                else:
                    c[i] = (1-env.buyerDiscount+env.buyerDiscount*c[i+1])**2 / (2*(1-env.buyerDiscount+env.buyerDiscount*c[i+1])-env.sellerDiscount*c[i+1])

            working_memory = {"T":env.T, "delta_b":env.buyerDiscount, "delta_s":env.sellerDiscount,
                            "SEPrice": {}, "c":c, "b_value":env.buyerVal
                            }
            buyer = StriDeAgent(problem_description=env.description_of_problem_class, demo=buyer_demo, tool_names=tool_names, init_memory=deepcopy(working_memory), llm_validator=False, logger=logger, engine=agent_engine)
            working_memory = {"T":env.T, "delta_b":env.buyerDiscount, "delta_s":env.sellerDiscount,
                            "SEPrice": {}, "c":c
                            }
            seller = StriDeAgent(problem_description=env.description_of_problem_class, demo=seller_demo, tool_names=tool_names, init_memory=deepcopy(working_memory), llm_validator=False, logger=logger, engine=agent_engine)
            agents = {"buyer":buyer, "seller":seller}
            return agents
        elif agent_type == "direct":
            buyer = DirectAgent(problem_description=env.description_of_problem_class, logger=logger,engine=agent_engine)
            seller = DirectAgent(problem_description=env.description_of_problem_class, logger=logger,engine=agent_engine)
            agents = {"buyer":buyer, "seller":seller}
            return agents
    elif env.name == "TicTacToe":
        if agent_type == "stride":
            game_demo = load_initial_instructions("envs/tictactoe/prompts/tictactoe_exmps.txt")
            from envs.tictactoe.tools import tool_names
            from envs.tictactoe.program_agent import AgentMemory
            working_memory = AgentMemory(
                    memo=dict(),
                    depth_scores=dict()
                )
            x = StriDeAgent(demo=game_demo, tool_names=tool_names, problem_description=env.description_of_problem_class, init_memory=working_memory, logger=logger, engine=agent_engine, game=game)
            o = StriDeAgent(demo=game_demo, tool_names=tool_names, problem_description=env.description_of_problem_class, init_memory=working_memory, logger=logger, engine=agent_engine, game=game)
            return {"X": x, "O": o}
        elif agent_type == "direct-code":
            x = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=direct_code_tic_tac_toe, read_file=False, logger=logger,engine=agent_engine)
            o = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=direct_code_tic_tac_toe, read_file=False, logger=logger,engine=agent_engine)
            agents = {"X":x, "O":o}
            return agents
        elif agent_type == "direct":
            x = DirectAgent(problem_description=env.description_of_problem_class, logger=logger,engine=agent_engine)
            o = DirectAgent(problem_description=env.description_of_problem_class, logger=logger,engine=agent_engine)
            return {"X": x, "O": o}
        elif agent_type == "stride-direct":
            game_demo = load_initial_instructions("envs/tictactoe/prompts/tictactoe_exmps.txt")
            from envs.tictactoe.tools import tool_names
            from envs.tictactoe.program_agent import AgentMemory
            working_memory = AgentMemory(
                    memo=dict(),
                    depth_scores=dict()
                )
            x = StriDeAgent(demo=game_demo, tool_names=tool_names, problem_description=env.description_of_problem_class, init_memory=working_memory, logger=logger, engine=agent_engine, game=game)           
            o = DirectAgent(problem_description=env.description_of_problem_class, logger=logger,engine=agent_engine)
            return {"X": x, "O": o}
        elif agent_type == "stride-direct-code":
            game_demo = load_initial_instructions("envs/tictactoe/prompts/tictactoe_exmps.txt")            
            from envs.tictactoe.tools import tool_names
            from envs.tictactoe.program_agent import AgentMemory            
            working_memory = AgentMemory(
                    memo=dict(),                    
                    depth_scores=dict()
                )
            x = StriDeAgent(demo=game_demo, tool_names=tool_names, problem_description=env.description_of_problem_class, init_memory=working_memory, logger=logger, engine=agent_engine, game=game)            
            o = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=direct_code_tic_tac_toe, read_file=False, logger=logger,engine=agent_engine)
            return {"X": x, "O": o}
    elif env.name == "ConnectN":
        if agent_type == "stride":
            game_demo = load_initial_instructions(f"envs/connectn/prompts/connect{n_in_row}_exmps{rows}*{columns}.txt")
            from envs.connectn.tools import tool_names
            from envs.connectn.program_agent import AgentMemory
            working_memory = AgentMemory(
                    memo=dict(),
                    depth_scores=dict()
                )
            x = StriDeAgent(demo=game_demo, tool_names=tool_names, problem_description=env.description_of_problem_class, init_memory=working_memory, logger=logger, engine=agent_engine, game=game)
            o = StriDeAgent(demo=game_demo, tool_names=tool_names, problem_description=env.description_of_problem_class, init_memory=working_memory, logger=logger, engine=agent_engine, game=game)
            return {"X": x, "O": o}
        elif agent_type == "direct-code":
            x = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=direct_code_connect_n, read_file=False, logger=logger,engine=agent_engine)
            o = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=direct_code_connect_n, read_file=False, logger=logger,engine=agent_engine)
            agents = {"X":x, "O":o}
            return agents
        elif agent_type == "direct":
            x = DirectAgent(problem_description=env.description_of_problem_class, logger=logger,engine=agent_engine)
            o = DirectAgent(problem_description=env.description_of_problem_class, logger=logger,engine=agent_engine)
            return {"X": x, "O": o}
        elif agent_type == "stride-direct":
            game_demo = load_initial_instructions(f"envs/connectn/prompts/connect{n_in_row}_exmps{rows}*{columns}.txt")
            from envs.connectn.tools import tool_names
            from envs.connectn.program_agent import AgentMemory
            working_memory = AgentMemory(
                    memo=dict(),
                    depth_scores=dict()
                )
            x = StriDeAgent(demo=game_demo, tool_names=tool_names, problem_description=env.description_of_problem_class, init_memory=working_memory, logger=logger, engine=agent_engine, game=game)            
            o = DirectAgent(problem_description=env.description_of_problem_class, logger=logger,engine=agent_engine)
            return {"X": x, "O": o}
        elif agent_type == "stride-direct-code":
            game_demo = load_initial_instructions(f"envs/connectn/prompts/connect{n_in_row}_exmps{rows}*{columns}.txt")            
            from envs.connectn.tools import tool_names
            from envs.connectn.program_agent import AgentMemory            
            working_memory = AgentMemory(
                    memo=dict(),                    
                    depth_scores=dict()
                )
            x = StriDeAgent(demo=game_demo, tool_names=tool_names, problem_description=env.description_of_problem_class, init_memory=working_memory, logger=logger, engine=agent_engine, game=game)            
            o = DirectCodeInterpreterAgent(problem_description=env.description_of_problem_class, demo=direct_code_tic_tac_toe, read_file=False, logger=logger,engine=agent_engine)
            return {"X": x, "O": o}    