import argparse
import shutil
from collections import deque

import config
from lenient_DQNAgent import LHDQNAgent
from anon_env import AnonEnv
import time
import pandas as pd
import os
import copy
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# --road_net 3_4 --volume jinan --suffix real
# --road_net 3_4 --volume synthetic --suffix turn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--road_net", type=str, default='4_4')  # '1_2') # which road net you are going to run
    parser.add_argument("--volume", type=str, default='mydata')  # '300'
    parser.add_argument("--suffix", type=str, default="500")  # 0.3

    global hangzhou_archive
    hangzhou_archive = False
    global TOP_K_ADJACENCY
    TOP_K_ADJACENCY = 5
    global TOP_K_ADJACENCY_LANE
    TOP_K_ADJACENCY_LANE = 5
    global NUM_ROUNDS
    NUM_ROUNDS = 100
    global EARLY_STOP
    EARLY_STOP = False
    global NEIGHBOR
    # TAKE CARE
    NEIGHBOR = True
    global SAVEREPLAY  # if you want to relay your simulation, set it to be True
    SAVEREPLAY = True
    global ADJACENCY_BY_CONNECTION_OR_GEO
    # TAKE CARE
    ADJACENCY_BY_CONNECTION_OR_GEO = False

    # modify:TOP_K_ADJACENCY in line 154
    global PRETRAIN
    PRETRAIN = False
    parser.add_argument("--cnt", type=int, default=3600)  # 3600

    global ANON_PHASE_REPRE
    ANON_PHASE_REPRE = {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0]
    }
    # print('ANON_PHASE_REPRE:', ANON_PHASE_REPRE)

    return parser.parse_args()


def calc_flow_statistics(road_net, flow_file):
    flows = json.load(open(flow_file, 'r'))
    for flow in flows:
        vehicle_data = flow['vehicle']
        route = flow['route']
        sum_length = 0
        for r in route:
            length = road_net.net_edge_dict[r]['length']
            sum_length += length
        max_speed = vehicle_data['maxSpeed']
        max_pos_acc = vehicle_data['maxPosAcc']
        t1 = max_speed / max_pos_acc
        s1 = 1/2 * max_pos_acc * t1 ** 2
        t2 = (sum_length - s1) / max_speed
        flow['min_cost_time'] = t1 + t2

    mean_min_time = np.mean([f['min_cost_time'] for f in flows])
    return flows, mean_min_time


def _init_env(road_net, suffix, volume):
    # main(args.memo, args.env, args.road_net, args.gui, args.volume, args.ratio, args.mod, args.cnt, args.gen)
    # Jinan_3_4
    NUM_COL = int(road_net.split('_')[0])
    NUM_ROW = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL

    ENVIRONMENT = "anon"

    traffic_file = "{0}_{1}_{2}_{3}.json".format(ENVIRONMENT, road_net, volume, suffix)

    global PRETRAIN
    global NUM_ROUNDS
    global EARLY_STOP
    dic_exp_conf_extra = {
        "TRAFFIC_FILE": [traffic_file],  # here: change to multi_traffic
        "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
        "NUM_ROUNDS": NUM_ROUNDS,
        "MODEL_POOL": False,
        "NUM_BEST_MODEL": 3,
        "PRETRAIN": PRETRAIN,  #
        "AGGREGATE": False,
        "DEBUG": False,
        "EARLY_STOP": EARLY_STOP,
    }

    global TOP_K_ADJACENCY
    global TOP_K_ADJACENCY_LANE
    global NEIGHBOR
    global SAVEREPLAY
    global ADJACENCY_BY_CONNECTION_OR_GEO
    global ANON_PHASE_REPRE
    dic_traffic_env_conf_extra = {
        "USE_LANE_ADJACENCY": True,
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,
        "ACTION_PATTERN": "set",
        "MEASURE_TIME": 10,
        # "IF_GUI": gui,
        "DEBUG": False,
        "TOP_K_ADJACENCY": TOP_K_ADJACENCY,
        "ADJACENCY_BY_CONNECTION_OR_GEO": ADJACENCY_BY_CONNECTION_OR_GEO,
        "TOP_K_ADJACENCY_LANE": TOP_K_ADJACENCY_LANE,
        "SIMULATOR_TYPE": ENVIRONMENT,
        "BINARY_PHASE_EXPANSION": True,
        "FAST_COMPUTE": True,

        "NEIGHBOR": NEIGHBOR,

        "SAVEREPLAY": SAVEREPLAY,
        "NUM_ROW": NUM_ROW,
        "NUM_COL": NUM_COL,

        "TRAFFIC_FILE": traffic_file,
        "VOLUME": volume,
        "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

        "phase_expansion": {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0],
            5: [1, 1, 0, 0, 0, 0, 0, 0],
            6: [0, 0, 1, 1, 0, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 1, 1],
            8: [0, 0, 0, 0, 1, 1, 0, 0]
        },

        "phase_expansion_4_lane": {
            1: [1, 1, 0, 0],
            2: [0, 0, 1, 1],
        },

        "LIST_STATE_FEATURE": [
            "cur_phase",
            "lane_num_vehicle",
            "lane_num_vehicle_been_stopped_thres1",
        ],

        "DIC_FEATURE_DIM": dict(
            D_LANE_QUEUE_LENGTH=(4,),
            D_LANE_NUM_VEHICLE=(4,),

            D_COMING_VEHICLE=(12,),
            D_LEAVING_VEHICLE=(12,),

            D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
            D_CUR_PHASE=(1,),
            D_NEXT_PHASE=(1,),
            D_TIME_THIS_PHASE=(1,),
            D_TERMINAL=(1,),
            D_LANE_SUM_WAITING_TIME=(4,),
            D_VEHICLE_POSITION_IMG=(4, 60,),
            D_VEHICLE_SPEED_IMG=(4, 60,),
            D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

            D_PRESSURE=(1,),

            D_ADJACENCY_MATRIX=(2,),

            D_ADJACENCY_MATRIX_LANE=(6,), ),

        "DIC_REWARD_INFO": {
            "flickering": 0,  # -5,#
            "sum_lane_queue_length": 0,
            "sum_lane_wait_time": 0,
            "sum_lane_num_vehicle_left": 0,  # -1,#
            "sum_duration_vehicle_left": 0,
            "sum_num_vehicle_been_stopped_thres01": 0,
            "sum_num_vehicle_been_stopped_thres1": -0.25,
            "pressure": 0
        },

        "LANE_NUM": {
            "LEFT": 1,
            "RIGHT": 1,
            "STRAIGHT": 1
        },

        "PHASE": {
            "anon": ANON_PHASE_REPRE,
        }
    }

    ## ==================== multi_phase ====================
    global hangzhou_archive
    if hangzhou_archive:
        template = 'Archive+2'
    elif volume == "mydata":
        template = "mydata"
    elif volume == "synthetic":
        template = "Synthetic"
    elif volume == 'jinan':
        template = "Jinan"
    elif volume == 'hangzhou':
        template = 'Hangzhou'
    elif volume == 'newyork':
        template = 'NewYork'
    elif volume == 'chacha':
        template = 'Chacha'
    elif volume == 'dynamic_attention':
        template = 'dynamic_attention'
    elif dic_traffic_env_conf_extra["LANE_NUM"] == config._LS:
        template = "template_ls"
    elif dic_traffic_env_conf_extra["LANE_NUM"] == config._S:
        template = "template_s"
    elif dic_traffic_env_conf_extra["LANE_NUM"] == config._LSR:
        template = "template_lsr"
    else:
        raise ValueError

    if dic_traffic_env_conf_extra['NEIGHBOR']:
        list_feature = dic_traffic_env_conf_extra["LIST_STATE_FEATURE"].copy()
        list_feature.remove("lane_num_vehicle_been_stopped_thres1")
        for feature in list_feature:
            for i in range(4):
                dic_traffic_env_conf_extra["LIST_STATE_FEATURE"].append(feature + "_" + str(i))

    dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]

    if dic_traffic_env_conf_extra['BINARY_PHASE_EXPANSION']:
        dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE'] = (8,)
        if dic_traffic_env_conf_extra['NEIGHBOR']:
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (8,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (8,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (8,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (8,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)
        else:

            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (1,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (1,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (1,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (1,)
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)

    print(traffic_file)
    prefix_intersections = str(road_net)
    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join("model", traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S',
                                                                                  time.localtime(time.time()))),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S',
                                                                                             time.localtime(
                                                                                                 time.time()))),
        "PATH_TO_DATA": os.path.join("data", template, prefix_intersections),
    }
    deploy_dic_exp_conf = merge(config.DIC_EXP_CONF, dic_exp_conf_extra)
    deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
    # TODO add agent_conf for different agents

    deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

    path_check(deploy_dic_path)
    copy_conf_file(deploy_dic_path, deploy_dic_exp_conf, deploy_dic_traffic_env_conf)
    copy_anon_file(deploy_dic_path, deploy_dic_exp_conf)

    path_to_log = os.path.join(deploy_dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)

    env = AnonEnv(
        path_to_log=path_to_log,
        path_to_work_directory=deploy_dic_path["PATH_TO_WORK_DIRECTORY"],
        dic_traffic_env_conf=deploy_dic_traffic_env_conf)

    return env, deploy_dic_exp_conf, deploy_dic_path


def copy_conf_file(dic_path, dic_exp_conf, dic_traffic_env_conf):
    # write conf files

    path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"),
              indent=4)
    json.dump(dic_traffic_env_conf,
              open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)


def path_check(deploy_dic_path):
    for path in deploy_dic_path.values():
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)


def copy_anon_file(dic_path, dic_exp_conf):
    # hard code !!!

    path = dic_path["PATH_TO_WORK_DIRECTORY"]
    # copy sumo files

    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_exp_conf["TRAFFIC_FILE"][0]),
                os.path.join(path, dic_exp_conf["TRAFFIC_FILE"][0]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_exp_conf["ROADNET_FILE"]),
                os.path.join(path, dic_exp_conf["ROADNET_FILE"]))


def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result


def tra_state(state):
    cur_phase = state['cur_phase']
    lane_num_vehicle = state['lane_num_vehicle']
    s = np.concatenate((cur_phase, lane_num_vehicle))
    return s


def state2state(state, current_step):
    cur_phase = state['cur_phase']
    lane_num_vehicle = state['lane_num_vehicle']
    step = float(current_step / 360)
    return np.concatenate(([step], cur_phase, lane_num_vehicle))


def neighbor_state(state):
    cur_phase = state['cur_phase']
    lane_num_vehicle = state['lane_num_vehicle']
    nei_state = np.concatenate((cur_phase, lane_num_vehicle))
    for i in range(4):
        nei_cur_phase = state["cur_phase_{0}".format(i)]
        nei_lane_num_vehicle = state["lane_num_vehicle_{0}".format(i)]
        nei_state = np.concatenate((nei_state, nei_cur_phase, nei_lane_num_vehicle))
    return nei_state


def evaluate():
    args = parse_args()
    test_env, test_dic_exp_conf, test_deploy_dic_path = _init_env(args.road_net, args.suffix, args.volume)
    # test_deploy_dic_path['PATH_TO_WORK_DIRECTORY'] =
    # test_deploy_dic_path['PATH_TO_MODEL'] =
    agent_num = test_env.dic_traffic_env_conf["NUM_INTERSECTIONS"]
    # test_deploy_dic_path['PATH_TO_MODEL'] = 'model/anon_3_4_jinan_real.json_09_02_20_43_43'
    test_deploy_dic_path['PATH_TO_MODEL'] = 'model/anon_3_4_jinan_real.json_09_09_13_40_49'

    flow_file_path = os.path.join(test_env.path_to_work_directory, test_env.dic_traffic_env_conf["TRAFFIC_FILE"])
    flows, min_travel_time = calc_flow_statistics(test_env.roadnet, flow_file_path)
    agents = []
    for i in range(agent_num):
        agent = LHDQNAgent(13, 4, [200, 200], 'LenientAgent' + str(i), epsilon_start=0,
                           logdir=test_deploy_dic_path['PATH_TO_WORK_DIRECTORY'] + '/logs' + str(i),
                           savedir=test_deploy_dic_path['PATH_TO_MODEL'] + '/save' + str(i),
                           batch_size=1000)
        agent.load_model()
        agents.append(agent)
    episode_len = int(test_dic_exp_conf["RUN_COUNTS"] / test_env.dic_traffic_env_conf["MIN_ACTION_TIME"])

    ep_agent_reward = []
    ep_reward = []

    agent_reward = np.zeros(agent_num)
    inter_data = {}
    for i in range(agent_num):
        i_name = test_env.list_intersection[i].inter_name
        inter_data[i_name] = np.zeros(episode_len)
    for iii in range(1):
        state = test_env.reset()
        ep_len = 0
        final_reward = 0
        while ep_len < episode_len:
            action = []
            for i in range(agent_num):
                # s = state2state(state[i], ep_len)
                s = tra_state(state[i])
                action.append(agents[i].test_choose(s))
                sum_queue_size = sum(state[i]['lane_num_vehicle'])
                inter_name = test_env.list_intersection[i].inter_name
                inter_data[inter_name][ep_len] = sum_queue_size

            next_state_, reward_, done_, _ = test_env.step(action)
            state = next_state_
            final_reward += np.mean(reward_)
            for i in range(agent_num):
                agent_reward[i] += reward_[i]
            ep_len += 1
        ep_agent_reward.append(agent_reward)
        ep_reward.append(final_reward)
        print('cur average travel time: ', test_env.info.items())
        print('min average travel time: ', min_travel_time)
        # TODO: why len(test_env.vehicle_set) is not equal to len(flows)
        # assert len(flows) == len(test_env.vehicle_set)

        plt.figure(figsize=(18, 12))
        custom_palette = sns.color_palette("viridis", agent_num)
        for i, k in enumerate(inter_data.keys()):
            episode_queue_size = inter_data[k]
            plt.plot(range(episode_len), episode_queue_size, color=custom_palette[i], label=k)
        plt.title('Flow distribution')
        plt.legend(bbox_to_anchor=(1, 1))
        plt.show()
    return np.mean(ep_agent_reward, axis=0), np.mean(ep_reward)


if __name__ == '__main__':
    evaluate()
