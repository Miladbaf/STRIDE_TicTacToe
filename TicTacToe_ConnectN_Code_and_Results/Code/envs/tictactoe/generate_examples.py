import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(root_dir)

from datetime import datetime
import argparse

from env import TicTacToe
from program_agent import MinimaxAgent, AgentMemory
from utils import Logger
from envs.env_helper import get_env_param

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def play_through(game, agents, logger, n_episodes, exmps_file=None):
    game.check_agents(agents)

    for ep in range(n_episodes):
        game.reset()
        for agent in agents.values():
            agent.reset()
        while not game.over():
            current_agent = agents[game.player]
            game.play_with_bfs_minimax(current_agent)
        logger.write(f"The {ep+1} episode has ended.\n", color="red")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default="TabularMDP", help="[TabularMDP, GridWorldMDP, BargainingCompleteInfoSingle, TicTacToe, ConnectN]")
    parser.add_argument('--n_episodes', type=int, default=1, help='number of episodes')
    parser.add_argument('--output_path', type=str, default="./outputs/", help='path to save the output')
    parser.add_argument('--verbose', type=int, default=1, help="0: not logger.write, 1: logger.write")
    parser.add_argument('--board', type=str, help="Initial board state for TicTacToe, a comma-separated list like 'X,None,None,O,None,None,X,None,None'")
    parser.add_argument('--player_turn', type=str, help="Specify whose turn it is in the game ('X' or 'O')")
    parser.add_argument('--is_maximizing', type=str2bool, help="Specify if the current move is maximizing or not")
    args = parser.parse_args()

    output_path = "./outputs/"
    os.makedirs(output_path, exist_ok=True)
    now = datetime.now()
    time_string = now.strftime('%Y%m%d%H%M%S')
    logger = Logger(output_path + args.game + "-" + time_string + ".html", args.verbose)

    board = [None if x.strip() == 'None' else x.strip() for x in args.board.split(',')] if args.board else None

    ### initialize game and agents ###
    exmps_file = "prompts/tictactoe_exmps.txt"
    game_param = get_env_param(env_name="TicTacToe", board=board, player_turn=args.player_turn, is_maximizing=args.is_maximizing)
    game = TicTacToe(initial_board=game_param['game_board'], starting_player=game_param['player_turn'], is_maximizing=game_param['is_maximizing'], log_file=exmps_file)

    working_memory = AgentMemory(
        memo=dict(),
        depth_scores=dict()
    )

    if args.player_turn == 'X':
        agent_x = MinimaxAgent(game, working_memory, 'X', args.is_maximizing, log_file=exmps_file)
        agent_o = MinimaxAgent(game, working_memory, 'O', not args.is_maximizing, log_file=exmps_file)
    else:
        agent_x = MinimaxAgent(game, working_memory, 'X', not args.is_maximizing, log_file=exmps_file)
        agent_o = MinimaxAgent(game, working_memory, 'O', args.is_maximizing, log_file=exmps_file)
    agents = {'X': agent_x, 'O': agent_o}

    ### start play ###
    play_through(game=game, agents=agents, logger=logger, n_episodes=args.n_episodes, exmps_file=exmps_file)