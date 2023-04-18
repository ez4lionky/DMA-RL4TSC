import random
import gym
import torch
from tqdm import tqdm
from envs.environment import TLCEnv
from world import World
from generator import LaneVehicleGenerator, IntersectionVehicleGenerator
from agent import DQNAgent
from metric import TravelTimeMetric, ThroughputMetric, SpeedScoreMetric, MaxWaitingTimeMetric, WaitingCountMeric
import argparse
import os
import numpy as np
import logging
from datetime import datetime
import utils as u

# os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
torch.autograd.set_detect_anomaly(True)

# set CUDA and seed
seed = 1
# random.seed(seed)
# np.random.seed(seed)
n_training_threads = 1
if torch.cuda.is_available():
    print("choose to use gpu...")
    device = torch.device("cuda")
    torch.set_num_threads(n_training_threads)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    print("choose to use cpu...")
    device = torch.device("cpu")
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(n_training_threads)

# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--config_file', type=str, default='./envs/jinan_3_4/config.json', help='path of config file')
# parser.add_argument('--config_file', type=str, default='./envs/hangzhou_4_4/config.json', help='path of config file')
parser.add_argument('--thread', type=int, default=8, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument("--save_rate", type=int, default=20,
                    help="save model once every time this many episodes are completed")
parser.add_argument('--save_dir', type=str, default="model/dqn_ma", help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/dqn", help='directory in which logs should be saved')
parser.add_argument('--network', type=str, default="CNN", choices=("FC", "CNN", "RNN", "Full"),
                    help='Network type used in DQN, fully-connected network with queue_size or CNN with map')
parser.add_argument('--model_average', type=str, default="None", choices=("Federated", "Distributed", "None"),
                    help='path to the file with information about the model')
parser.add_argument('--parameters', type=str, default="agent/configs_new_dqn/config.yaml",
                    help='path to the file with informations about the model')
parser.add_argument('--debug', type=u.str2bool, const=True, nargs='?',
                    default=False, help='When in the debug mode, it will not record logs')
parser.add_argument("--other_params",
                    nargs='*',
                    default=[],
                    type=str)

args = parser.parse_args()
logger = u.get_logger(args)

# Config File
parameters = u.get_info_file(args.parameters)
episodes = parameters['episodes']
learning_start = parameters['learning_start']
update_model_freq = parameters['update_model_freq']
update_target_model_freq = parameters["update_target_model_freq"]
parameters['log_path'] = args.log_dir
action_interval = parameters['action_interval']
yellow_phase_time = parameters['yellow_phase_time']
parameters['network'] = args.network

# start wandb
# u.wand_init(args, parameters, "TLC - Results", "new_dqn_ma", "new_dqn_ma", debug=args.debug)
u.log_args_and_parameters(logger, args, parameters)
# create world
world = World(args.config_file, args.thread, action_interval, yellow_phase_time)

# create agents
agents = []
for i in world.intersections:
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(DQNAgent(
        action_space,
        [LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None, scale=.025),
         IntersectionVehicleGenerator(parameters, world, i, targets=["current_phase", "vehicle_map"])],
        # IntersectionVehicleGenerator(parameters, world, i, targets=["vehicle_map"])],
        LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=True, average="all", negative=True),
        # PressureRewardGenerator(world, i, scale=0.005, negative=True),
        i.id,
        parameters,
        world,
        device
    ))
    # if args.load_model:
    #     agents[0].load_model(args.save_dir)
agent_nums = len(agents)

# Create metric
metric = [TravelTimeMetric(world), ThroughputMetric(world), SpeedScoreMetric(world),
          MaxWaitingTimeMetric(world), WaitingCountMeric(world)]

# create env
env = TLCEnv(world, agents, metric)
edges_list = u.edges_index_list(agent_nums, world)


# train dqn_agent
def train(args, env):
    epochs_replay = parameters["epochs_replay"]
    epochs_initial_replay = parameters["epochs_initial_replay"]
    first_replay = True

    total_num = 0
    edge_features = None
    for e in range(episodes):
        last_obs_list = env.reset()

        if e % args.save_rate == args.save_rate - 1:
            env.eng.set_save_replay(True)
            env.eng.set_replay_file("replay_%s.txt" % e)
        else:
            env.eng.set_save_replay(False)
        episodes_rewards = [0 for _ in agents]
        episodes_decision_num = 0
        i = 0

        while i < args.steps:  # 3600 steps
            if i % action_interval == 0:
                actions = []
                att_weights_all = []
                last_obs = [item[0] for item in last_obs_list]
                last_obs_phase = [item[1][0] for item in last_obs_list]
                last_obs_map = [item[1][1] for item in last_obs_list]
                for agent_id, agent in enumerate(agents):
                    if total_num > agent.learning_start:
                        action, att_weights = agent.get_action(last_obs[agent_id], last_obs_phase[agent_id],
                                                               last_obs_map[agent_id], edges_list, edge_features)
                    else:
                        action, att_weights = agent.sample(last_obs[agent_id])
                    actions.append(action)
                    att_weights_all.append(att_weights[0])

                rewards_list = []
                for _ in range(action_interval):
                    obs_list, rewards, dones, _ = env.step(actions)
                    obs = [item[0] for item in obs_list]
                    obs_phase = [item[1][0] for item in obs_list]
                    obs_map = [item[1][1] for item in obs_list]
                    rewards_list.append(rewards)
                    i += 1
                rewards = np.mean(rewards_list, axis=0)

                for agent_id, agent in enumerate(agents):
                    agent.remember(last_obs[agent_id], last_obs_phase[agent_id], last_obs_map[agent_id],
                                   actions[agent_id], rewards[agent_id], obs[agent_id], obs_phase[agent_id],
                                   obs_map[agent_id], edges_list, edge_features, edge_features)
                    episodes_rewards[agent_id] += rewards[agent_id]

                episodes_decision_num += 1
                total_num += 1
                last_obs_list = obs_list

                if total_num > learning_start and total_num % update_model_freq == update_model_freq - 1:
                    data_list = []
                    for agent in agents:
                        data_list.append(agent.fetch_data())

                    if first_replay:
                        for epoch in tqdm(range(epochs_initial_replay)):
                            for agent_id, agent in enumerate(agents):
                                agent.new_replay(data_list[agent_id], adjust_epsilon=False)
                            if args.model_average != 'None':
                                weight_matrix = u.average_weights_matrix(args, len(agents), world, device, pick_rate=1)
                                u.average_models(agents, weight_matrix)
                        for agent in agents:
                            agent.update_target_network()
                        first_replay = False
                    else:
                        for epoch in range(epochs_replay):
                            for agent_id, agent in enumerate(agents):
                                agent.new_replay(data_list[agent_id])
                            if args.model_average != 'None':
                                weight_matrix = u.average_weights_matrix(args, len(agents), world, device, pick_rate=1)
                                u.average_models(agents, weight_matrix)

                        torch.cuda.synchronize()

                if total_num > learning_start and total_num % update_target_model_freq == update_target_model_freq - 1:
                    for agent_id, agent in enumerate(agents):
                        agent.update_target_network()

            # if all(dones):
            #     break

        ## lr scheduler
        if total_num > learning_start:
            for agent in agents:
                agent.learning_rate_decay()
        logger.info(f'episode:{e}, lr:{agents[0].optimizer.param_groups[0]["lr"]}')

        if e % args.save_rate == args.save_rate - 1:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            for agent in agents:
                agent.save_model(args.save_dir)

        eval_dict = {}

        logger.info(f"episode:{e}/{episodes - 1}, steps:{i}")
        # eval_dict["episode"] = e
        # eval_dict["steps"] = i
        eval_dict["learning rate"] = agents[0].get_lr()

        for agent_id, agent in enumerate(agents):
            logger.info("\tagent:{}, mean_episode_reward:{}".format(agent_id,
                                                                    np.mean(episodes_rewards[
                                                                                agent_id]) / episodes_decision_num))

        for metric in env.metric:
            logger.info(f"\t{metric.name}: {metric.eval()}")
            eval_dict[metric.name] = metric.eval()

        eval_dict["epsilon"] = agents[0].epsilon
        eval_dict["mean_episode_reward"] = np.mean(episodes_rewards) / episodes_decision_num

        # if not args.debug:
        #     u.wand_log(eval_dict)

    # for agent in agents:
    # agents[0].save_model(args.save_dir)


if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    train(args, env)
