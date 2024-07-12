from datetime import datetime
import os
import sys
import argparse
import matplotlib.pyplot as plt
from rich.console import Console
from utils import Logger, create_env, create_agents
import numpy as np

class Logger(object):
    def __init__(self, log_file, verbose=True):
        self.console = Console(record=True)
        self.log_file = log_file
        self.verbose = verbose

        self.write("All outputs written to %s" % log_file)
        return 

    def write(self, message, color=None):
        self.console.save_html(self.log_file, clear=False)
        if(self.verbose): 
            if color is not None:
                self.console.print("[{}]".format(color)+message+"[/{}]".format(color))
            else:
                self.console.print(message)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def play_through(env, agents, logger):
    n_episodes = 1
    for role in agents:
        agents[role].reset()
        # agents[role].messages.append({"role":"assistant", "content":"[Beginnining of the game]"})
        instance_description = env.get_description(role)
        agents[role].get_instance_info(instance_description)
        logger.write("To {}:".format(role))
        logger.write(instance_description)

    for ep in range(1, n_episodes+1):
        # repeatively play the same instance of the env (e.g., MDP with unknown env model)
        env.reset()
        logger.write("episode {}/{} ...".format(ep, n_episodes), color = "red")

        if env.name == "tabular_mdp":
            # agents["agent"].reason("How do I compute the optimal Q values for this MDP instance using value iteration?")
            agents["agent"].reason("Now compute the optimal policy, that is, the optimal action at each step and each state.")
        elif env.name == "bargain_alternate_singleissue":
            agents["buyer"].reason("Now compute the subgame perfect equilibrium (SPE) step by step.")
            agents["seller"].reason("Now compute the subgame perfect equilibrium (SPE) step by step.")

        metric_ls = []
        state = env.state # initial state
        while not env.is_done: # in case game never ends due to failure in checking terminal condition
            logger.write(state.textual_descript, color = "red")
            cur_agent = agents[state.cur_agent]
            action = cur_agent.move(state)

            # compute some performance metric
            metric = get_result(env, agents, state, action, logger)
            metric_ls.append(metric)

            logger.write("{}: {}".format(state.cur_agent, action), color = "red")
            logger.write("metric: {}".format(metric), color = "red")
            state, reward = env.step(action)
            cur_agent.update(state, reward)

        logger.write("This episode has ended!", color="red")
        logger.write("Performance metric: {}".format(metric_ls))
    return metric_ls

def play_through_connect(game, agents, logger, n_episodes, agent_type):
    game.check_agents(agents) 
    
    for ep in range(n_episodes):
        game.reset()
        logger.write("episode {}/{}...".format(ep, n_episodes-1), color = "green")
        
        for agent in agents.values():
            agent.reset()

        if env.name == "TicTacToe":
            if agent_type == "direct-code":
                while not game.over():
                    current_agent = agents[game.player]
                    action = current_agent.move_connect(game.get_state_struct())
                    if action is not None:
                        if game.board[action] == None:
                            game.board[action] = game.player
                            game.switch_player()
            else:
                while not game.over():
                    current_agent = agents[game.player]
                    action = current_agent.move(game.get_state_struct())
                    if action is not None:
                        if game.board[action] == None:
                            game.board[action] = game.player
                            game.switch_player()
        elif env.name == "ConnectN":
            if agent_type == 'direct':
                while not game.over():
                    current_agent = agents[game.player]
                    action = current_agent.move(game.get_state_struct())
                    if action is not None:
                        if game.board[action] == None:
                            game.board[action] = game.player
                            game.switch_player()
            elif agent_type == "direct-code":
                while not game.over():
                    current_agent = agents[game.player]
                    action = current_agent.move_connect(game.get_state_struct())

                    if action is not None:
                        placed = False
                        for row in reversed(range(game.rows)):
                            index = row * game.columns + action
                            if index < (game.rows * game.columns - 1):
                                if game.board[index] is None:
                                    game.board[index] = game.player
                                    placed = True
                                    break
                        if not placed:
                            print("Column is full or index is out of range!")
                        else:
                            game.switch_player()
            else:
                while not game.over():
                    current_agent = agents[game.player]
                    action = current_agent.move(game.get_state_struct())
                    if action is not None:
                        if game.board[action] == None:
                            game.board[action] = game.player
                            game.switch_player()

        if game.win('X'):
            result = "player X wins!"
        elif game.win('O'):
            result = "player O wins!"
        else:
            result = "a tie!"
            
        logger.write(f"Board is: {game.board}")
        logger.write(f"Over: Game is over and the result is {result} I should exit the program.")

        # with open("results.txt", "a") as results_file:
        #     results_file.write(f"Episode {ep+1}: result: {result}\n")

        logger.write(f"The episode {ep+1} has ended.\n", color="red")

def get_result(env, agents, state, action, logger):
    if env.name == "tabular_mdp":
        # compute the regret: V[h,s,optimal_action]-V[h,s,action]
        q_optimal, _ = env.compute_qVals_v1()
        q = q_optimal[state.time_step, state.mathematical_descript]
        logger.write("q_optimal for current step and state {}".format(q))
        optimal_actions = np.where(q==np.max(q))
        if action in optimal_actions:
            success = True
        else:
            success = False
        return success
    if env.name == "bargain_alternate_singleissue":
        # the current agent is proposing a price
        # let's see if this price is spe price
        if state.actions == [0.0, 1.0]:
            price, util = env.calculate_spe_price_utility(cur_time=state.time_step, cur_player=state.cur_agent, deadline=env.T, buyer_discount=env.buyerDiscount, seller_discount=env.sellerDiscount)
            # print("spe price {} and utility {}.".format(price, util))
            logger.write("spe price {}, {} proposed price {}".format(price, state.cur_agent, action))
            if abs(price-action) <= 1e-2:
                success = True
            else:
                success = False
        else:
            # the current agent is deciding to acc or rej
            if state.cur_agent == "buyer":
                discount = env.env_param["buyerDiscount"]
                value = 1.0
            else:
                discount = env.env_param["sellerDiscount"]
                value = 0.0
            # utility of acc
            price = state.mathematical_descript[-1]
            util_acc = abs(price-value) * discount**(state.time_step-1)
            _, util_rej = env.calculate_spe_price_utility(cur_time=state.time_step+1, cur_player=state.cur_agent, deadline=env.T, buyer_discount=env.buyerDiscount, seller_discount=env.sellerDiscount)
            logger.write("utility accept {}, utility reject {}, {} action {}".format(util_acc, util_rej, state.cur_agent, action))
            if util_acc >= util_rej - 0.01:
                if action == "accept":
                    success = True
                else:
                    success = False
            else:
                if action == "accept":
                    success = False
                else:
                    success = True
    if env.name == "bargain_onesided_uncertainty":
        # the current agent, seller, is proposing a price
        # let's see if this price is spe price
        if state.actions == [0.0, 1.0]:
            se_prices = env.get_se_prices()
            price = se_prices[state.time_step]
            # print("spe price {} and utility {}.".format(price, util))
            logger.write("spe price {}".format(price))
            if abs(price-action) <= 1e-2:
                success = True
            else:
                success = False
        else:
            # the current agent, buyer, is deciding to acc or rej
            # utility of acc
            discount = env.buyerDiscount
            price = state.mathematical_descript[-1]
            util_acc = (env.buyerVal-price) * discount**(state.time_step-1)
            if state.time_step == env.T:
                util_rej = 0.0
            else:
                se_prices = env.get_se_prices()
                se_price_next_time = se_prices[state.time_step+1]
                util_rej = (env.buyerVal-se_price_next_time) * discount**(state.time_step)
            logger.write("utility accept {}, utility reject {}, {} action {}".format(util_acc, util_rej, state.cur_agent, action))
            if util_acc >= util_rej - 0.01:
                if action == "accept":
                    success = True
                else:
                    success = False
            else:
                if action == "accept":
                    success = False
                else:
                    success = True
        return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="bargain_onesided_uncertainty", help="[tabular_mdp, bargain_alternate_singleissue, bargain_onesided_uncertainty, TicTacToe, ConnectN]")
    parser.add_argument('--mdp_known', type=bool, default=False)
    parser.add_argument('--agent_type', type=str, default="stride-direct-code", help="[direct, direct-code, stride, stride-direct-code, stride-direct]")
    parser.add_argument('--agent_engine', type=str, default="gpt-3.5-turbo", help="[gpt-3.5-turbo, gpt-4o, gpt-4-turbo-2024-04-09]")
    parser.add_argument('--random_param', type=bool, default=True)
    parser.add_argument('--n_exps', type=int, default=10, help='number of times to play in the environment')
    parser.add_argument('--output_path', type=str, default="./outputs/", help='path to save the output')
    parser.add_argument('--verbose', type=int, default=1, help="0: not logger.write, 1: logger.write")
    parser.add_argument('--rows', type=int, help="Specify the number of rows in game board for Connect-N")
    parser.add_argument('--columns', type=int, help="Specify the number of columns in game board for Connect-N")
    parser.add_argument('--n_in_row', type=int, help="Specify how many markers should be in a rom for a player to win in Connect-N")
    parser.add_argument('--player_turn', type=str, help="Specify whose turn it is in the game ('X' or 'O')")
    parser.add_argument('--is_maximizing', type=str2bool, help="Specify if the current move is maximizing or not")
    parser.add_argument('--board', type=str, help="Initial board state for TicTacToe, a comma-separated list like 'X,None,None,O,None,None,X,None,None'")
    args = parser.parse_args()

    output_path = "./outputs/" + args.env + "/"
    os.makedirs(output_path, exist_ok=True) 
    now = datetime.now()
    time_string = now.strftime('%Y%m%d%H%M%S')
    logger = Logger(output_path + args.env + "-" + time_string + ".html", args.verbose)

    result_list = []
    if args.env == "tabular_mdp":
        for exp in range(1, args.n_exps+1):
            logger.write("experiment {}/{} ...".format(exp, args.n_exps), color = "red")
            
            # initialize the environment and agents
            env = create_env(args.env, args.random_param)
            agents = create_agents(env, logger, args.agent_type, args.agent_engine)
            if not env.check_agents(agents): # check if all agents required by the env are specified
                raise ValueError("illegal agents for env {}".format(args.env))
            
            # start playing
            logger.write("Start to play {}".format(env.name), color = "red")
            result = play_through(env=env, agents=agents, logger=logger)
            result_list.append(result)

        total_success = 0.0
        for res in result_list:
            for r in res:
                if r:
                    total_success += 1.0
        logger.write("success rate is {}={}/{}".format(total_success/(args.n_exps*env.epLen), total_success, args.n_exps*env.epLen))
    elif args.env == "bargain_alternate_singleissue" or args.env == "bargain_onesided_uncertainty":
        for exp in range(1, args.n_exps+1):
            logger.write("experiment {}/{} ...".format(exp, args.n_exps), color = "red")
            
            # initialize the environment and agents
            env = create_env(args.env, args.random_param)
            agents = create_agents(env, logger, args.agent_type, args.agent_engine)
            if not env.check_agents(agents): # check if all agents required by the env are specified
                raise ValueError("illegal agents for env {}".format(args.env))
            
            # start playing
            logger.write("Start to play {}".format(env.name), color = "red")
            result = play_through(env=env, agents=agents, logger=logger)
            result_list.append(result)
            
        total_success = 0.0
        total_num = 0.0
        for res in result_list:
            total_num += len(res)
            for r in res:
                if r:
                    total_success += 1.0
        logger.write("success rate is {}={}/{}".format(total_success/total_num, total_success, total_num))
    elif args.env == "TicTacToe":
        env = create_env(env_name=args.env, player_turn=args.player_turn,is_maximizing=args.is_maximizing, logger=logger)
        agents = create_agents(env=env, logger=logger, agent_type=args.agent_type, agent_engine=args.agent_engine, game=env)
        play_through_connect(game=env, agents=agents, logger=logger, n_episodes=args.n_exps, agent_type=args.agent_type)
    elif args.env == "ConnectN":
        env = create_env(env_name=args.env, player_turn=args.player_turn, is_maximizing=args.is_maximizing, rows=args.rows, columns=args.columns, n_in_row=args.n_in_row, logger=logger)
        agents = create_agents(env=env, logger=logger, agent_type=args.agent_type, agent_engine=args.agent_engine, rows=args.rows, columns=args.columns, n_in_row=args.n_in_row, game=env)
        play_through_connect(game=env, agents=agents, logger=logger, n_episodes=args.n_exps, agent_type=args.agent_type)