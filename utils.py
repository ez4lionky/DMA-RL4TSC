import torch
import collections
import copy
from typing import List
import numpy as np
import pandas as pd
import csv
import os
import json
import logging
from datetime import datetime
from pathlib import Path
import yaml
import random
import wandb

# from example.example0.run_new_dqn_ma import get_neighbors


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')


def get_logger(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    file_name_time = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    file_name = f"{args.log_dir}/{file_name_time}"

    if not args.debug:
        fh = logging.FileHandler(file_name + '.log')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    return logger


def log_args_and_parameters(logger, args, parameters):
    logger.info("env: ")
    logger.info(args.config_file)
    logger.info("other params: ")
    logger.info(args.other_params)
    logger.info("args: ")
    logger.info(args)
    logger.info("parameters: ")
    logger.info(parameters)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# model parameters averaging of federated or distributed learning
def average_weights_matrix(args, n_agents, world, device, pick_rate=0.8):
    weight_matrix = None
    if args.model_average == 'Federated':
        perm = np.vstack([np.random.permutation(n_agents) for _ in range(n_agents)])
        average_num = int(n_agents * pick_rate)
        # weight_matrix = np.ones((n_agents, n_agents), dtype=np.float32) / n_agents
        weight_matrix = np.zeros((n_agents, n_agents), dtype=np.float32)
        # weight_matrix[]
        for i in range(n_agents):
            weight_matrix[i][perm[i][:average_num]] = np.ones(average_num) / average_num
        weight_matrix = torch.from_numpy(weight_matrix).to(device)
    elif args.model_average == 'Distributed':
        weight_matrix = np.ones((n_agents, n_agents), dtype=np.float32) / n_agents
        graph = world.graph
        for i in range(n_agents):
            cur_neighbors = list(graph.neighbors(i))
            n_neighbors = len(cur_neighbors)
            for j in range(n_agents):
                if j in cur_neighbors:
                    weight_matrix[i][j] = 1 / (n_neighbors * 2)
                elif i == j:
                    weight_matrix[i][j] = 1 / 2
                else:
                    weight_matrix[i][j] = 0
        weight_matrix = torch.from_numpy(weight_matrix).to(device)
    return weight_matrix


@torch.no_grad()
def average_models(agents, weight_matrix):
    worker_state_dict = [x.model.state_dict() for x in agents]
    weight_keys = list(filter(lambda x: 'target' not in x, worker_state_dict[0].keys()))
    # weight_keys = list(worker_state_dict[0].keys())
    result_models = [collections.OrderedDict() for _ in agents]

    for key in weight_keys:
        cur_weights = torch.cat([worker_state_dict[i][key][None, ...] for i in range(len(agents))])
        weight_shape = cur_weights.shape[1:]
        if any(key.__contains__(k) for k in ['running_mean', 'running_var', 'num_batches_tracked']):
            # result_models =
            # print(cur_weights)
            result_weights = cur_weights.reshape(-1, weight_shape.numel())
        else:
            result_weights = torch.matmul(weight_matrix, cur_weights.reshape(-1, weight_shape.numel()))
        result_weights = result_weights.reshape(-1, *weight_shape)
        for i in range(len(agents)):
            result_models[i][key] = result_weights[i]

    # update fed weights to each model
    for i, agent in enumerate(agents):
        model = agent.model
        cur_sd = model.state_dict()
        cur_sd.update(result_models[i])
        model.load_state_dict(cur_sd)
    return


def yaml_read(yaml_file):
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


# wandb
def wand_init(args, parameters, project_name='my-test-project', name=None, group=None, debug=False):
    wb_keys_file = os.path.expanduser('~/wandb_keys.yaml')
    # user = 'chch9907'
    user = 'lionky'
    if Path(wb_keys_file).exists():
        wb_cfg = yaml_read(wb_keys_file)[user]
        os.environ['WANDB_API_KEY'] = wb_cfg['WANDB_API_KEY']
        os.environ['WANDB_BASE_URL'] = wb_cfg['WANDB_BASE_URL']
        os.environ['WANDB_START_METHOD'] = 'thread'
    if not debug:
        run = wandb.init(project=project_name, group=group)
        run.name = f"{name}-{run.name.split('-')[-1]}"
        args_dict = json.loads(json.dumps(args, default=lambda o: o.__dict__))
        cfg_dict = dict(args=args_dict, params=parameters)
        wandb.config.update(cfg_dict)
        wandb.define_metric("Average Travel Time", summary="min")


def wand_log(dict):
    wandb.log(dict)


def get_info_file(file_name):
    try:
        data = yaml_read(file_name)

    except Exception as e:
        print("Error while opening config file")
        print(e)
        exit()

    return data


def append_new_line_states(file_name, lista):
    """Append given list as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    path = f'{file_name}_state.csv'
    list_to_save = []
    list_to_save.append(lista[0])
    list_to_save.append(lista[1])
    list_to_save.append(lista[2][0])
    list_to_save.append(lista[3][0])
    list_to_save.append(lista[4])
    list_to_save.append(lista[5])

    list_to_save = np.array(
        list_to_save, dtype="object").reshape(-1, len(list_to_save))
    df = pd.DataFrame(list_to_save, columns=[
                      'episode', 'i', 'actual_state', 'next_state', 'Mphase', 'Iphase'])
    df.to_csv(path, sep=';', mode='a', index=False,
              header=not os.path.exists(path))

    # with open(file_name, "a+") as file_object:
    # file_object.write(list_to_append)
    # file_object.write("\n")
    # write = csv.writer(file_object, delimiter = ';')
    # print(list_to_append)
    # write.writerows(map(lambda x: [x], list_to_append))
    # file_object.close()


def append_new_line(file_name, lista):
    """Append given list as a new line at the end of file"""
    # Open the file in append & read mode ('a+')

    list_to_save = []
    list_to_save.append(lista[0][0])
    list_to_save.append(lista[0][1])
    list_to_save.append(lista[1])
    list_to_save.append(lista[2])
    list_to_save.append(lista[3][0])
    list_to_save.append(lista[3][1])
    list_to_save.append(lista[4])
    list_to_save.append(lista[5])

    list_to_save = np.array(
        list_to_save, dtype="object").reshape(-1, len(list_to_save))
    df = pd.DataFrame(list_to_save,
                      columns=['actual_state', 'actual_phase', 'action', 'reward', 'next_state', 'next_phase',
                               'episode', 'i'])
    df = df[['episode', 'i', 'actual_state', 'actual_phase',
             'action', 'reward', 'next_state', 'next_phase']]
    df.to_csv(f'{file_name}.csv', sep=';', mode='a', index=False,
              header=not os.path.exists(f'{file_name}.csv'))


def get_neighbors(agent_id, world):
    return list(world.graph.neighbors(agent_id))


def aggregate_obs(obs, world):
    """aggregate obs without adding position onehot:state = queue size(12)
       padding zero vectors if no neighbor intersections at that position.
    """
    agg_obs_list = []
    for agent_id, ob in enumerate(obs):
        current_ob = [np.expand_dims(ob, axis=0)]
        nbrs_list = get_neighbors(agent_id, world)
        nbrs_pos_list = judge_neighbor_dir_jinan(agent_id, nbrs_list)
        j = 0
        for pos in range(4):
            if pos in nbrs_pos_list:
                current_ob.append(np.expand_dims(obs[nbrs_list[j]], axis=0))
                j += 1
            else:
                current_ob.append(np.expand_dims(np.zeros_like(ob), axis=0))
        nbrs_ob = np.concatenate(current_ob, axis=0)
        agg_obs_list.append(nbrs_ob)
        # print("nbrs_ob:", nbrs_ob.shape)
    return agg_obs_list


def _one_hot_encoding(target, nums):
    return [0 if i != target else 1 for i in range(nums)]


def aggregate_obs_sa(obs, world):
    """state = queue size(12) + position onehot(4)
    """
    agg_obs_list = []
    # last_shape = None
    for agent_id, ob in enumerate(obs):
        onehot_dir_i = np.array(_one_hot_encoding(-1, 4))
        onehot_ob = np.concatenate((ob, onehot_dir_i))
        current_ob = [np.expand_dims(onehot_ob, axis=0)]
        nbrs_list = get_neighbors(agent_id, world)
        nbrs_pos_list = judge_neighbor_dir_jinan(agent_id, nbrs_list)
        j = 0
        for pos in range(4):
            onehot_dir_j = np.array(_one_hot_encoding(pos, 4))
            if pos in nbrs_pos_list:
                idx = nbrs_pos_list.index(pos)
                onehot_nbrs_ob = np.concatenate(
                    (obs[nbrs_list[idx]], onehot_dir_j))
                current_ob.append(np.expand_dims(onehot_nbrs_ob, axis=0))
                j += 1
            else:
                onehot_nbrs_ob = np.concatenate(
                    (np.zeros_like(ob), onehot_dir_j))
                current_ob.append(np.expand_dims(onehot_nbrs_ob, axis=0))
        nbrs_ob = np.concatenate(current_ob, axis=0)
        # if last_shape is None:
        #     last_shape = nbrs_ob.shape
        # else:
        #     assert last_shape == nbrs_ob.shape, f"{last_shape}, {nbrs_ob.shape}"
        agg_obs_list.append(nbrs_ob)
    return agg_obs_list


# todo screen and sa meaning difference?
def aggregate_obs_screen(obs, world):
    """ for each agent (intersection), use neighboring nodes queue_size (12-dim) plus position one-hot encoding (4-dim)
        the virtual nodes (actually not exist) will use all zero data (i.e. 16 dim)
        including itself, [5, 16] shape features in the final
    """
    agg_obs_list = []
    for agent_id, ob in enumerate(obs):
        onehot_li = np.array(_one_hot_encoding(-1, 4))
        onehot_ob = np.concatenate((ob, onehot_li))
        current_ob = [np.expand_dims(onehot_ob, axis=0)]

        nbrs_list = get_neighbors(agent_id, world)
        nbrs_pos_list = judge_neighbor_dir_jinan(agent_id, nbrs_list)
        # 0 up 1 right 2 down 3 left
        j = 0
        for i in range(4):
            dir_j = np.array(_one_hot_encoding(i, 4))
            if i not in nbrs_pos_list:
                onehot_nbrs_ob = np.concatenate((np.zeros_like(ob), dir_j))
                current_ob.append(np.expand_dims(onehot_nbrs_ob, axis=0))
                continue
        # for i, nbrs_id in enumerate(nbrs_list):
            nbrs_ob = copy.deepcopy(obs[nbrs_list[j]])
            weights_dict = idx2weight[nbrs_pos_list[j]]
            for idx in range(len(nbrs_ob)):
                if idx in weights_dict.keys():
                    nbrs_ob[idx] *= weights_dict[idx]
                else:
                    nbrs_ob[idx] = 0
            onehot_nbrs_ob = np.concatenate((nbrs_ob, dir_j))
            current_ob.append(np.expand_dims(onehot_nbrs_ob, axis=0))
            j += 1
        nbrs_ob = np.concatenate(current_ob, axis=0)
        agg_obs_list.append(nbrs_ob)
        # print("nbrs_ob:", nbrs_ob.shape)
    return agg_obs_list


def aggregate_obs_new(obs, world):
    agg_obs_list = []
    for agent_id, ob in enumerate(obs):
        # onehot_li = np.array(_one_hot_encoding(-1, 4))
        # onehot_ob = np.concatenate((ob, onehot_li))
        current_ob = [np.expand_dims(ob, axis=0)]

        nbrs_list = get_neighbors(agent_id, world)
        nbrs_pos_list = judge_neighbor_dir_jinan(agent_id, nbrs_list)
        j = 0
        for i in range(4):
            dir_j = np.array(_one_hot_encoding(i, 4))
            if i not in nbrs_pos_list:
                onehot_nbrs_ob = np.concatenate((np.zeros(3), np.array(ob[3*i:3*(i+1)]), dir_j, np.zeros(2))) # , dir_j
                current_ob.append(np.expand_dims(onehot_nbrs_ob, axis=0))
                continue
            nbrs_ob = obs[nbrs_list[j]]
            weights_dict = idx2weight[nbrs_pos_list[j]]
            screened_ob = []
            for k, v in weights_dict.items():
                screened_ob.append(v * nbrs_ob[k])  #  * weights_dict[idx]
            onehot_nbrs_ob = np.concatenate((np.array(screened_ob), np.array(ob[3*i:3*(i+1)]), dir_j, np.zeros(2)))  # dir_j,
            current_ob.append(np.expand_dims(onehot_nbrs_ob, axis=0))
            j += 1
        nbrs_ob = np.concatenate(current_ob, axis=0)
        agg_obs_list.append(nbrs_ob)
        # print("nbrs_ob:", nbrs_ob.shape)
    return agg_obs_list

right_w = 1
# straight_w = 1 / 4
# left_w = 1 / 8
straight_w = 1
left_w = 1

# 0: up: {r:9, s:1, l:5}
# 1: right: {r:0, s:4, l:8}
# 2: down: {r:3, s:7, l:11}
# 3: left:{r:6, s:10, l:2}
idx2weight = {
    0: {9: right_w, 1: straight_w, 5: left_w},
    1: {0: right_w, 4: straight_w, 8: left_w},
    2: {3: right_w, 7: straight_w, 11: left_w},
    3: {6: right_w, 10: straight_w, 2: left_w}
}

# new
# idx2weight = {
#     0: {2: right_w, 4: straight_w, 8: left_w},
#     1: {3: right_w, 7: straight_w, 9: left_w},
#     2: {6: right_w, 11: straight_w, 1: left_w},
#     3: {10: right_w, 0: straight_w, 5: left_w}
# }


def aggregate_rewards(rewards, world, att_weights_all=None):
    agg_reward_list = []
    for agent_id, r in enumerate(rewards):
        nbrs_list = get_neighbors(agent_id, world)
        if att_weights_all is None:
            # nbrs_r = np.mean([rewards[nbrs_id] for nbrs_id in nbrs_list])
            # agg_r = (r + nbrs_r) / 2
            agg_r = (3/4) * r + (1/16) * \
                np.sum([rewards[nbrs_id] for nbrs_id in nbrs_list])  # !
        else:
            att_weights = att_weights_all[agent_id]  # [e10, e20, e30, e00]
            # print("att_weights:", att_weights)
            # nbrs_att_weights = att_weights[:-1] / np.sum(att_weights[:-1])
            nbrs_r = np.sum([rewards[nbrs_id] * att_weights[j]
                            for j, nbrs_id in enumerate(nbrs_list)])
            agg_r = nbrs_r + r * att_weights[-1]

        agg_reward_list.append(agg_r)
    return agg_reward_list


def edges_index_list(agent_nums, world):
    edges_list = []
    for agent_id in range(agent_nums):
        n_neighbours = len(get_neighbors(agent_id, world))
        # edges = np.array([[i, 0] for i in range(1, n_neighbours+1)],  #!
        #                     dtype=np.int64).T  # from source to target
        edges = np.array([[pos + 1, 0] for pos in range(4)],  # ! nbrs_pos_list
                         dtype=np.int64).T
        edges_list.append(edges)
    return edges_list


def edges_index_list_sa(agent_nums, world):
    edges_list = []
    for agent_id in range(agent_nums):
        nbrs_list = get_neighbors(agent_id, world)
        nbrs_pos_list = judge_neighbor_dir_jinan(agent_id, nbrs_list)
        edges = np.array([[pos + 1, 0] for pos in range(4)],  # ! nbrs_pos_list
                         dtype=np.int64).T  # from source to target
        edges_list.append(edges)
    return edges_list


def judge_neighbor_dir_jinan(agent_id, neighbor_list):
    # clockwise judgement
    nbrs_pos_list = []
    for id in neighbor_list:
        if id == agent_id + 3:  # up
            nbrs_pos_list.append(0)
        if id == agent_id + 1:  # right
            nbrs_pos_list.append(1)
        if id == agent_id - 3:  # down
            nbrs_pos_list.append(2)
        if id == agent_id - 1:  # left
            nbrs_pos_list.append(3)
    return nbrs_pos_list


def judge_neighbor_dir_hangzhou(agent_id, neighbor_list):
    # clockwise judgement
    nbrs_pos_list = []
    for id in neighbor_list:
        if id == agent_id + 4:  # up
            nbrs_pos_list.append(0)
        if id == agent_id + 1:  # right
            nbrs_pos_list.append(1)
        if id == agent_id - 4:  # down
            nbrs_pos_list.append(2)
        if id == agent_id - 1:  # left
            nbrs_pos_list.append(3)
    return nbrs_pos_list


def get_edge_features(obs, world):
    """ extract 3 useful lanes of each neighbor intersection as edge features
    """
    # d_ij, v_ij, l_i, l_j
    # d_ij, v_ij, l_j, p_j
    # eij: j in {up, right, down, left}
    all_edge_features = []
    phase_num = 8
    max_neighbors_num = 4

    def _one_hot_encoding(target, nums):
        return [0 if i != target else 1 for i in range(nums)]

    for agent_id, ob in enumerate(obs):
        # obs_map:(2, 12, 60)
        # target_pos_map = ob[0, :, :]  # get position map
        # target_pos_map = ob  # get phase  (1, 12)

        # ob: (1, 12) phase
        # target_intersect = world.intersections[agent_id]
        # clockwise order: [up, right, down, left]
        neighbor_list = get_neighbors(agent_id, world)
        neighbor_intersects = [world.intersections[j] for j in neighbor_list]
        # nbrs_pos_list = _judge_neighbor_dir_hangzhou(neighbor_list)
        nbrs_pos_list = judge_neighbor_dir_jinan(agent_id, neighbor_list)

        edge_features = []
        # pos_i = target_intersect.pos
        # l_i = target_intersect.current_phase
        # onehot_l_i = _one_hot_encoding(l_i, phase_num)

        # onehot_dir_i = _one_hot_encoding(max_intersect_num, max_intersect_num + 1)
        target_e_ij = [0, 0, 0] # + onehot_l_i + onehot_dir_i
        edge_features.append(target_e_ij)

        j = 0
        for i in range(4):
            onehot_dir_j = _one_hot_encoding(i, max_neighbors_num)
            # edge_features.append(onehot_dir_j)
            if i not in nbrs_pos_list:
                edge_features.append([0, 0, 0])
                continue
            nbrs_inter = neighbor_intersects[j]
        # for i, nbrs_inter in enumerate(neighbor_intersects):
            # l_j = nbrs_inter.current_phase
            # onehot_l_j = _one_hot_encoding(l_j, phase_num)

            # pos_j = nbrs_inter.pos
            # d_ij = abs(pos_j[0] - pos_i[0]) + abs(pos_j[1] - pos_i[1])  # 600 or 800
            # v_ij = target_pos_map[nbrs_pos_list[i]*3 : (nbrs_pos_list[i] + 1)*3]

            incoming_cars_num = []
            nbrs_ob = obs[neighbor_list[j]]

            # nbrs_ob = nbrs_ob[0]  #! nbrs position map

            weights_dict = idx2weight[nbrs_pos_list[j]]
            for k, v in weights_dict.items():
                incoming_cars_num.append(nbrs_ob[k] * v)
                # incoming_cars_num.append(np.sum(nbrs_ob[k]) * v)  #!
            # print("incoming_cars_num:", incoming_cars_num)
            # onehot_dir_j = _one_hot_encoding(i, max_neighbors_num)

            # length = 1 + 3 + 8 + 8 + 4 = 24
            # e_ij = [d_ij] + incoming_cars_num + onehot_l_i + onehot_l_j + onehot_dir_j
            e_ij = incoming_cars_num
            edge_features.append(e_ij)
            j += 1
        all_edge_features.append(np.array(edge_features, dtype=float))
        # all_edge_features.append(None)
    return all_edge_features


def get_edge_features_sa(obs, world):
    """position one hot: e.g., [1, 0, 0, 0] 
    """
    # d_ij, v_ij, l_i, l_j
    # d_ij, v_ij, l_j, p_j
    # eij: j in {up, right, down, left}
    all_edge_features = []
    phase_num = 8
    max_neighbors_num = 4

    def _one_hot_encoding(target, nums):
        return [0 if i != target else 1 for i in range(nums)]

    for agent_id, ob in enumerate(obs):
        # obs_map:(2, 12, 60)
        # target_pos_map = ob[0, :, :]  # get position map
        # target_pos_map = ob  # get phase  (1, 12)

        # ob: (1, 12) phase
        # target_intersect = world.intersections[agent_id]
        # neighbor_list = get_neighbors(agent_id, world) # clockwise order: [up, right, down, left]
        # neighbor_intersects = [world.intersections[j] for j in neighbor_list]
        # nbrs_pos_list = judge_neighbor_dir_jinan(agent_id, neighbor_list)

        edge_features = []
        # pos_i = target_intersect.pos
        # l_i = target_intersect.current_phase
        # onehot_l_i = _one_hot_encoding(l_i, phase_num)
        # onehot_dir_i = _one_hot_encoding(max_intersect_num, max_intersect_num + 1)
        # target_e_ij = [0, 0] + onehot_l_i + onehot_dir_i
        # edge_features.append(target_e_ij)

        # j = 0
        for i in range(4):
            onehot_dir_j = _one_hot_encoding(i, max_neighbors_num)
            edge_features.append(onehot_dir_j)
            # if i not in nbrs_pos_list:
            #     edge_features.append([0, 0, 0])
            #     continue
            # nbrs_inter = neighbor_intersects[j]
        # for i, nbrs_inter in enumerate(neighbor_intersects):
            # l_j = nbrs_inter.current_phase
            # onehot_l_j = _one_hot_encoding(l_j, phase_num)

            # pos_j = nbrs_inter.pos
            # d_ij = abs(pos_j[0] - pos_i[0]) + abs(pos_j[1] - pos_i[1])  # 600 or 800
            # v_ij = target_pos_map[nbrs_pos_list[i]*3 : (nbrs_pos_list[i] + 1)*3]

            # incoming_cars_num = []
            # nbrs_ob = obs[neighbor_list[j]]

            # nbrs_ob = nbrs_ob[0]  #! nbrs position map

            # weights_dict = idx2weight[nbrs_pos_list[j]]
            # for k, v in weights_dict.items():
            #     incoming_cars_num.append(nbrs_ob[k] * v)
            #     # incoming_cars_num.append(np.sum(nbrs_ob[k]) * v)  #!
            # # print("incoming_cars_num:", incoming_cars_num)
            # onehot_dir_j = _one_hot_encoding(nbrs_pos_list[j], max_neighbors_num)

            # length = 1 + 3 + 8 + 8 + 4 = 24
            # e_ij = [d_ij] + incoming_cars_num + onehot_l_i + onehot_l_j + onehot_dir_j
            # e_ij = incoming_cars_num
            # edge_features.append(e_ij)
            # j += 1
        all_edge_features.append(np.array(edge_features, dtype=float))
        # all_edge_features.append(None)
    return all_edge_features


def remove_right_turn(obs: List[np.ndarray]):
    new_obs = []
    right_turn_index = [0, 3, 6, 9]
    for ob in obs:
        ob[right_turn_index] = 0
        new_obs.append(ob)
    return new_obs
