import os
import argparse
import torch
from world import World
from agent.dqn_gat import TLC_DQN
import utils as u
import time


if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    torch.autograd.set_detect_anomaly(True)

    # set CUDA and seed
    seed = 1
    # random.seed(seed)
    # np.random.seed(seed)
    n_training_threads = 1
    if torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
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
    parser.add_argument('--thread', type=int, default=8, help='number of threads')
    parser.add_argument('--parameters', type=str, default="agent/configs_new_dqn/config.yaml",
                        help='path to the file with informations about the model')
    parser.add_argument('--debug', type=u.str2bool, const=True, nargs='?',
                        default=False, help='When in the debug mode, it will not record logs')
    args = parser.parse_args()

    # Config File
    parameters = u.get_info_file(args.parameters)
    action_interval = parameters['action_interval']
    yellow_phase_time = parameters['yellow_phase_time']

    world = World(args.config_file, args.thread, action_interval, yellow_phase_time)
    agent_nums = len(world.intersections)
    edges_list = u.edges_index_list_sa(agent_nums, world)
    model = TLC_DQN(32, 12, 8, [2, 12, 40], 8, device)
    batch_size = 32

    for i in range(1000):
        obs = torch.randn(batch_size, 5, 10, 16).numpy()  # bs * neighbor_nodes * temporal length * (12 + 4)
        obs_phase = torch.nn.functional.one_hot(torch.tensor(0), 8).unsqueeze(0).repeat(batch_size, 1).numpy()
        obs_map = torch.randn(batch_size, 2, 12, 48).numpy()  # B, C, H, W
        edges_idx = torch.tensor(edges_list[0], dtype=torch.long).unsqueeze(0).repeat(batch_size, 1, 1).numpy()
        # batch_size, 2, 4
        edges_feature = torch.randn(batch_size, 4, 4).numpy()

        torch.cuda.synchronize()
        start = time.time()
        model(obs, obs_phase, obs_map, edges_idx, edges_feature)
        torch.cuda.synchronize()
        end = time.time()
        print(f'Total cost time: {end - start}')
