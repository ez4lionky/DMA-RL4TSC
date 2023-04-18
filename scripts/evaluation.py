import gym
import torch
from envs.environment import TLCEnv
from world import World
from generator import LaneVehicleGenerator, IntersectionVehicleGenerator
from agent import DQNAgent
from metric import TravelTimeMetric, ThroughputMetric, SpeedScoreMetric, MaxWaitingTimeMetric, WaitingCountMeric
import argparse
import os.path as osp
import json
import numpy as np
import utils as u
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import pickle


def plot_spatial_temporal(ts_gen_flow, road_net, steps=3600):
    def divide_data(width=10):
        new_idxs, new_nums = [], []
        for idx in range(0, steps, width):
            new_idxs.append(idx + width / 2)
            cur_num = sum(flow_nums[idx:idx + width])
            new_nums.append(cur_num)
        return new_idxs, new_nums, width

    def color_map(values):
        sm = ScalarMappable(norm=Normalize(vmin=min(values), vmax=max(values)),
                            cmap=sns.cubehelix_palette(as_cmap=True))
        colors = [sm.to_rgba(v) for v in values]
        return colors

    def bar_plot(title):
        fig, axes = plt.subplots()
        fig.set_size_inches(18.5, 10.5)
        plt.tight_layout(pad=2)
        axes.bar(new_idxs, new_nums, width)
        axes.set_title(title)
        return

    sns.set_style("whitegrid")
    label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    vis_graph = road_net.vis_graph
    node_pos = road_net.node_positions
    edge_pos = road_net.edge_positions
    ts, flow_nums = np.arange(steps), np.zeros(steps)
    for k in ts_gen_flow.keys():
        flow_nums[k] = ts_gen_flow[k]

    new_idxs, new_nums, width = divide_data(10)
    bar_plot(f'The number of generated flows for each {width} step')
    new_idxs, new_nums, width = divide_data(100)
    bar_plot(f'The number of generated flows for each {width} step')

    # spatial distribution (intersection and approach) of generated flow num
    fig, axes = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    plt.tight_layout(pad=2)
    node_sizes = [(vis_graph.nodes[n]['gen_flow_num'] // 100 + 1) * 1000 for n in vis_graph.nodes]
    nx.draw(vis_graph, node_pos, ax=axes, font_size=8, connectionstyle='arc3, rad = 0.1', node_size=node_sizes)
    labels = {n: f"{n}_{vis_graph.nodes[n]['gen_flow_num']}" for n in vis_graph.nodes}
    nx.draw_networkx_labels(vis_graph, node_pos, labels, font_size=12, bbox=label_options)
    axes.set_title('Generated flow number for each intersection')
    plt.show()

    # passed flow number distribution
    fig, axes = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    plt.tight_layout(pad=2)
    node_sizes = [np.sqrt(vis_graph.nodes[n]['pass_flow_num'] // 100) * 1000 for n in vis_graph.nodes]
    node_colors = color_map([vis_graph.nodes[n]['pass_flow_num'] for n in vis_graph.nodes])

    nx.draw(vis_graph, node_pos, ax=axes, font_size=8, connectionstyle='arc3, rad = 0.1',
            node_size=node_sizes, node_color=node_colors)
    labels = {n: f"{n}_{vis_graph.nodes[n]['pass_flow_num']}" for n in vis_graph.nodes}
    nx.draw_networkx_labels(vis_graph, node_pos, labels, font_size=12, bbox=label_options)
    labels = {e: f"{vis_graph.edges[e]['edge_id']}_{vis_graph.edges[e]['pass_flow_num']}" for e in vis_graph.edges}
    edge_colors = color_map([vis_graph.edges[e]['pass_flow_num'] for e in vis_graph.edges])
    for i, (k, v) in enumerate(labels.items()):
        label_option = label_options.copy()
        label_option.update({"fc": edge_colors[i]})
        axes.text(edge_pos[k][0], edge_pos[k][1], v, size=8, color='k', bbox=label_option)
    axes.set_title('Passed flow number for each intersection and approach')
    plt.show()
    return


def calc_flow_statistics(road_net, config, steps=3600):
    flow_file = osp.join(config['dir'], config['flowFile'])
    flows = json.load(open(flow_file, 'r'))
    vis_graph = road_net.vis_graph
    edge_dict = road_net.net_edge_dict
    ts_gen_flow = defaultdict(int)

    for flow in flows:
        ts_gen_flow[flow['startTime']] += 1
        vehicle_data = flow['vehicle']
        route = flow['route']
        sum_length = 0
        for rid, r in enumerate(route):
            edge_info = edge_dict[r]
            input_node, output_node = edge_info['input_node'][13:], edge_info['output_node'][13:]
            if rid == 0:
                vis_graph.nodes[input_node]['gen_flow_num'] += 1
                vis_graph.edges[input_node, output_node]['gen_flow_num'] += 1
            vis_graph.nodes[output_node]['pass_flow_num'] += 1
            vis_graph.edges[input_node, output_node]['pass_flow_num'] += 1
            length = road_net.net_edge_dict[r]['length']
            sum_length += length
        max_speed = vehicle_data['maxSpeed']
        max_pos_acc = vehicle_data['maxPosAcc']
        t1 = max_speed / max_pos_acc
        s1 = 1/2 * max_pos_acc * t1 ** 2
        t2 = (sum_length - s1) / max_speed
        flow['min_cost_time'] = t1 + t2
        assert max_speed == world.max_speed

    mean_min_time = np.mean([f['min_cost_time'] for f in flows])
    # plot_spatial_temporal(ts_gen_flow, road_net, steps)
    return flows, mean_min_time


def update_analysis_data(num, data, in_env, in_agents, last_actions, rewards):
    w = in_env.world
    # lane_vehicles = w.eng.get_lane_vehicle_count()
    lane_vehicles = w.eng.get_lane_waiting_vehicle_count()
    for ii in range(len(in_agents)):
        cur_in_lanes = in_agents[ii].ob_generator[1].all_in_lanes
        # v_num = sum([len(lane_vehicles[ll]) for ll in cur_in_lanes])
        v_num = sum([lane_vehicles[ll] for ll in cur_in_lanes])
        inter_name = in_agents[ii].iid
        data[inter_name][num]['sum_queue_size'] = v_num
        data[inter_name][num]['lane_vehicles'] = lane_vehicles
        data[inter_name][num]['last_action'] = None if last_actions is None else last_actions[ii]
        data[inter_name][num]['rewards'] = rewards[ii]
    return data


class FixedTimeAgent:
    def __init__(self, intersection):
        # self.action_space = action_space
        # self.ob_generator = ob_generator
        # self.reward_generator = reward_generator
        self.intersection = intersection
        self.iid = intersection.id
        return

    def get_action(self):
        inter = self.intersection
        phase = inter.current_phase
        next_phase = (phase + 1) % len(inter.phases)
        return next_phase


# parse args
parser = argparse.ArgumentParser(description='Run Example')
parser.add_argument('--config_file', type=str, default='./envs/jinan_3_4/config.json', help='path of config file')
# parser.add_argument('--config_file', type=str, default='./envs/synthetic_4_4/config.json', help='path of config file')
# parser.add_argument('--config_file', type=str, default='./envs/hangzhou_4_4/config.json', help='path of config file')
parser.add_argument('--thread', type=int, default=8, help='number of threads')
parser.add_argument('--steps', type=int, default=3600, help='number of steps')
parser.add_argument('--load_model', action="store_true", default=True)
parser.add_argument('--save_dir', type=str, default="model/dqn_ma_gat",
                    help='directory in which model should be saved')
parser.add_argument('--log_dir', type=str, default="log/evaluate", help='directory in which logs should be saved')
parser.add_argument('--network', type=str, default="Full", choices=("FC", "CNN", "Full"),
                    help='Network type used in DQN, fully-connected network with queue_size or CNN with map')
parser.add_argument('--parameters', type=str, default="agent/configs_new_dqn/config.yaml",
                    help='path to the file with informations about the model')
parser.add_argument('--debug', type=u.str2bool, const=True, nargs='?',
                    default=False, help='When in the debug mode, it will not record logs')

args = parser.parse_args()
logger = u.get_logger(args)

# Config File
parameters = u.get_info_file(args.parameters)
parameters['log_path'] = args.log_dir
action_interval = parameters['action_interval']
yellow_phase_time = parameters['yellow_phase_time']
parameters['network'] = args.network
parameters['epsilon_initial'] = 0.0
parameters['epsilon_min'] = 0.0

# create world
yellow_phase_time = 3
world = World(args.config_file, args.thread, action_interval, yellow_phase_time)
calc_flow_statistics(world.road_graph, world.cityflow_config)

# create agents
device = torch.device("cuda:0")
baseline_agents = []
agents = []
for ind, i in enumerate(world.intersections):
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(DQNAgent(
        action_space,
        [LaneVehicleGenerator(world, i, ["lane_count"], in_only=True, average=None, scale=.025),
         IntersectionVehicleGenerator(parameters, world, i, targets=["current_phase", "vehicle_map"])],
        LaneVehicleGenerator(world, i, ["lane_waiting_count", "speed_score"],
                             in_only=True, average="all", negative=True),
        i.id,
        parameters,
        world,
        device
    ))
    baseline_agents.append(FixedTimeAgent(i))
    if args.load_model:
        agents[ind].load_model(args.save_dir)
agent_nums = len(agents)

# Create metric
metric = [TravelTimeMetric(world), ThroughputMetric(world), SpeedScoreMetric(world),
          WaitingCountMeric(world), MaxWaitingTimeMetric(world)]
# create env
env = TLCEnv(world, agents, metric)
edges_list = u.edges_index_list(agent_nums, world)


def test(args, agents, env):
    last_obs = env.reset()
    env.eng.set_save_replay(True)
    env.eng.set_replay_file("replay_evaluation.txt")
    episodes_rewards = [0 for _ in agents]
    episodes_decision_num = 0
    i = 0
    history_buffer = [[] for _ in range(agent_nums)]

    action_num = 0
    action_nums = args.steps // action_interval
    inter_data = {}
    for agent_id in range(agent_nums):
        i_name = env.world.intersection_ids[agent_id]
        inter_data[i_name] = [{} for _ in range(action_nums)]

    actions = None
    for _ in range(action_interval):
        # vehicles = env.world.eng.get_vehicles()
        # if vehicles:
        #     for v in vehicles:
        #         print(v)
        #         print(env.world.eng.get_vehicle_info(v))
        obs_list, rewards, dones, _ = env.step(actions)
        # obs_map = [item[1] for item in obs_list]
        obs = [item[0] for item in obs_list]
        obs_phase = [item[1][0] for item in obs_list]
        obs_map = [item[1][1] for item in obs_list]

        obs = u.aggregate_obs_screen(obs, world)  # (5, 16), [12 + 4 one hot]
        for j in range(agent_nums):
            history_buffer[j].append(obs[j])  # (12, 5, 16)
        i += 1
    inter_data = update_analysis_data(action_num, inter_data, env, agents, actions, rewards)

    last_obs_map = np.array(obs_map)
    last_hist_obs = np.array(history_buffer).transpose(0, 2, 1, 3)  # (12, 10, 5, 16)->(12, 5, 10, 16)
    last_obs_phase = np.array(obs_phase)
    last_edge_features = u.get_edge_features_sa(last_obs, world)
    while i < args.steps:
        print(i)
        if i % action_interval == 0:
            action_num += 1

            actions = []
            att_weights_all = []
            for agent_id, agent in enumerate(agents):
                # if i > args.steps:
                if i < args.steps:
                # if i > 3000:
                    action, att_weights = agent.get_action(last_hist_obs[agent_id], last_obs_phase[agent_id],
                                                           last_obs_map[agent_id], edges_list[agent_id],
                                                           last_edge_features[agent_id])
                    actions.append(action)
                    att_weights_all.append(att_weights[0])
                else:
                    action = baseline_agents[agent_id].get_action()
                    actions.append(action)
            rewards_list = []
            history_buffer = [[] for _ in range(agent_nums)]
            for _ in range(action_interval):
                # print(f'step {i}')
                # vehicles = env.world.eng.get_vehicles()
                # if vehicles:
                #     for v in vehicles:
                #         print(env.world.eng.get_vehicle_info(v))
                obs_list, rewards, dones, _ = env.step(actions)
                # obs_map = [item[1] for item in obs_list]
                obs = [item[0] for item in obs_list]
                obs_phase = [item[1][0] for item in obs_list]
                obs_map = [item[1][1] for item in obs_list]

                rewards_list.append(rewards)
                obs = u.aggregate_obs_screen(obs, world)
                for j in range(agent_nums):
                    history_buffer[j].append(obs[j])
                i += 1

            rewards = np.mean(rewards_list, axis=0)
            inter_data = update_analysis_data(action_num, inter_data, env, agents, actions, rewards)
            cur_edge_features = u.get_edge_features_sa(last_obs, world)
            last_obs = obs.copy()
            last_hist_obs = np.array(history_buffer).transpose(0, 2, 1, 3).copy()  # (12, 10, 5, 16)->(12, 5, 10, 16)
            last_obs_phase = np.array(obs_phase).copy()
            last_obs_map = np.array(obs_map).copy()
            last_edge_features = cur_edge_features.copy()

            for agent_id, agent in enumerate(agents):
                episodes_rewards[agent_id] += rewards[agent_id]
                episodes_decision_num += 1

    for agent_id, agent in enumerate(agents):
        logger.info("\tagent:{}, mean_episode_reward:{}".format(agent_id,
                                                                episodes_rewards[agent_id] / episodes_decision_num))
    pickle.dump(inter_data, file=open(f"original_inter_data.pkl", "wb"))
    for m in env.metric:
        logger.info(f"\t{m.name}: {m.eval()}")
    plt.figure(figsize=(18.5, 10.5))
    custom_palette = sns.color_palette("viridis", agent_nums)
    all_rewards = []
    for iid, k in enumerate(inter_data.keys()):
        sum_qs = [inter_data[k][step]['sum_queue_size'] for step in range(action_nums)]
        rewards = [inter_data[k][step]['rewards'] for step in range(action_nums)]
        all_rewards.append(rewards)
        plt.plot(range(action_nums), sum_qs, color=custom_palette[iid], label=k)

    # plt.title('Flow distribution of Fixed-time agent')
    plt.tight_layout(pad=2)
    plt.title('The queue size of our method')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.xlabel('Action number')
    plt.ylabel('The number of vehicles')
    # plt.savefig(f"flow.png", bbox_inches='tight', transparent=True)
    plt.show()
    
    all_rewards = np.array(all_rewards)
    inter_rewards = np.sum(all_rewards, axis=-1)
    worst_inter_idxs = np.argsort(inter_rewards)[:3]
    for wii in worst_inter_idxs:
        print(agents[wii].iid)
    fig, axes = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    plt.tight_layout(pad=2)
    road_net = env.world.road_graph
    vis_graph = road_net.vis_graph
    node_pos = road_net.node_positions
    labels = {n: f"{n}_{vis_graph.nodes[n]['node_id']}" for n in vis_graph.nodes}
    node_sizes = []
    for n in vis_graph.nodes:
        node_id = vis_graph.nodes[n]['node_id']
        if node_id in world.id2idx.keys():
            idx = world.id2idx[node_id]
            labels[n] = f"{n}_{inter_rewards[idx]:.2f}"
            node_sizes.append(-inter_rewards[idx] * 25)
        else:
            labels[n] = f"{n}_0"
            node_sizes.append(1000)
    nx.draw(vis_graph, node_pos, ax=axes, font_size=8, connectionstyle='arc3, rad = 0.1', node_size=node_sizes)
    label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    nx.draw_networkx_labels(vis_graph, node_pos, labels, font_size=12, bbox=label_options)
    plt.show()

    bad_case_idxs = np.unravel_index(np.argsort(all_rewards, axis=None), all_rewards.shape)
    print(bad_case_idxs)
    return


if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    test(args, agents, env)
