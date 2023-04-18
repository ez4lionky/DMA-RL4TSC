import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from . import RLAgent
import random
import numpy as np
from collections import deque
from agent.SamplerAlgorithms import PrioritizedMemory
import os
import pickle
from agent.dqn import DQNNet, DQNMapNet, DQNSeqNet
from agent.dqn_gat import TLC_DQN
import utils as u


class DQNAgent(RLAgent):
    model_constructor = {'FC': DQNNet, 'CNN': DQNMapNet, 'RNN': DQNSeqNet, 'Full': TLC_DQN}

    def __init__(self, action_space, ob_generator, reward_generator, iid, parameters, world, device):
        super().__init__(action_space, ob_generator, reward_generator)

        self.iid = iid
        self.parameters = parameters
        network = parameters['network']
        ob_length = ob_generator[0].ob_length
        phase_shape = ob_generator[1].phase_shape
        map_shape = ob_generator[1].map_shape
        hidden_size = parameters['hidden_nodes']

        self.world = world
        self.world.subscribe("pressure")

        self.gamma = self.parameters["gamma"]  # discount rate
        self.epsilon = self.parameters["epsilon_initial"]  # exploration rate
        self.epsilon_min = self.parameters["epsilon_min"]
        self.epsilon_decay = self.parameters["epsilon_decay"]

        self.use_prioritized = self.parameters["use_prioritized"]
        self.batch_size = self.parameters['batch_size']
        if not self.use_prioritized:
            self.memory = deque(maxlen=self.parameters["buffer_size"])
        else:
            self.memory = PrioritizedMemory(capacity=self.parameters["buffer_size"])

        self.learning_start = self.parameters["learning_start"]
        self.update_model_freq = self.parameters["update_model_freq"]
        self.update_target_model_freq = self.parameters["update_target_model_freq"]
        self.epochs_replay = self.parameters["epochs_replay"]
        self.epochs_initial_replay = self.parameters["epochs_initial_replay"]

        self.first_replay = True
        self.device = device

        # for the federated and distributed training, initialize all models with same random seed
        u.setup_seed(1)
        Model = self.model_constructor[network]
        self.model = Model(hidden_size, ob_length, phase_shape, map_shape, action_space.n, device).to(device)
        self.target_model = Model(hidden_size, ob_length, phase_shape, map_shape, action_space.n, device).to(device)
        self.target_model.eval()
        self.update_target_network()
        self.lr = self.parameters["lr"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.5)
        self.steps = 0

    def learning_rate_decay(self):
        self.scheduler.step()

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def sample(self, ob):
        n_nodes = ob.shape[0]
        default_weights = [np.array([1 / n_nodes for _ in range(n_nodes)])]
        return self.action_space.sample(), default_weights

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def remember(self, ob, phase, obs_map, action, reward, next_ob, next_phase, next_obs_map,
                 edges_idx, edge_features, next_edge_features):
        data = ob, phase, obs_map, action, reward, next_ob, next_phase, next_obs_map, \
               edges_idx, edge_features, next_edge_features
        self.memory.append(data)

    def get_action(self, ob, phase, obs_map, edges_idx, edge_features):
        # print("obs_map:", obs_map.shape)  # bs, 2, 12, 60
        if np.random.rand() <= self.epsilon:
            return self.sample(ob)
        ob = np.expand_dims(ob, 0)
        phase = np.expand_dims(phase, 0)
        edges_idx = np.expand_dims(edges_idx, 0)
        if edge_features is not None:
            edge_features = np.expand_dims(edge_features, 0)
        obs_map = np.expand_dims(obs_map, 0)
        self.model.eval()
        act_values, att_weights = self.model(ob, phase, obs_map, edges_idx, edge_features)
        # _, att_weights = self.target_model(ob, edges_idx, edge_features)
        act_values = act_values.detach().cpu().numpy()
        action = np.argmax(act_values[0])  # 0~7
        return action, att_weights

    def model_update(self, obs, phase, obs_map, actions, rewards, next_obs, next_phase, next_obs_map,
                     edges_idx, edge_features, next_edge_features, weights=None):
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float).unsqueeze(1)
        # print("actions:", actions.shape)
        ## Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. 
        state_action_values, _ = self.model(obs, phase, obs_map, edges_idx, edge_features)
        # print("state_action_values:", state_action_values.shape)
        state_action_values = state_action_values.gather(1, actions)

        ## Compute V(s_{t+1}) for all next states.
        # with torch.no_grad():
        next_state_values, _ = self.target_model(next_obs, next_phase, next_obs_map, edges_idx, next_edge_features)
        # print("next_state_values:", next_state_values.shape)
        next_state_values = next_state_values.detach().max(1)[0].unsqueeze(1)

        ## Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + rewards
        abs_errors = torch.abs(expected_state_action_values - state_action_values)
        abs_errors = abs_errors.mean(axis=-1)
        # print("expected_state_action_values:", expected_state_action_values)

        ## Compute loss
        # criterion = nn.SmoothL1Loss()
        criterion = nn.MSELoss()
        if weights is not None:
            weights = torch.tensor(weights, device=self.device, dtype=torch.float)
            loss = (weights * (state_action_values - expected_state_action_values) ** 2).mean()
        else:
            loss = criterion(state_action_values, expected_state_action_values)

        ## Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return abs_errors

    def fetch_data(self):
        if self.use_prioritized:
            b_idx, minibatch, b_weights = self.memory.sample(self.batch_size)
        else:
            minibatch = random.sample(self.memory, self.batch_size)
            b_idx, b_weights = None, None
        data = [np.stack(x) for x in np.array(minibatch, dtype=object).T]
        # obs, obs_map, actions, rewards, next_obs, next_obs_map, edges_idx, edge_features, next_edge_features, tree_idx
        data.append(b_idx)
        data.append(b_weights)
        return data

    def new_replay(self, data, adjust_epsilon=True):
        obs, phase, obs_map, actions, rewards, next_obs, next_phase, next_obs_map, edges_idx, \
        edge_features, next_edge_features, tree_idx, weights = data
        self.model.train()
        abs_errors = self.model_update(obs, phase, obs_map, actions, rewards, next_obs, next_phase, next_obs_map,
                                       edges_idx, edge_features, next_edge_features, weights)
        if self.use_prioritized:
            self.memory.batch_update(tree_idx, abs_errors)
        if adjust_epsilon and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return

    def replay(self):
        data = self.fetch_data()
        obs, phase, obs_map, actions, rewards, next_obs, next_phase, next_obs_map, edges_idx, \
        edge_features, next_edge_features, tree_idx, weights = data
        self.model.train()
        if self.first_replay:
            for i in tqdm(range(self.epochs_initial_replay)):
                # minibatch = self.sampler_algorithm.get_sample(self.memory)
                abs_errors = self.model_update(obs, phase, obs_map, actions, rewards, next_obs, next_phase, next_obs_map,
                                               edges_idx, edge_features, next_edge_features, weights)
                if self.use_prioritized:
                    self.memory.batch_update(tree_idx, abs_errors)
            self.update_target_network()
            self.first_replay = False
        else:
            # minibatch = self.sampler_algorithm.get_sample(self.memory)
            abs_errors = self.model_update(obs, phase, obs_map, actions, rewards, next_obs, next_phase, next_obs_map,
                                           edges_idx, edge_features, next_edge_features)
            if self.use_prioritized:
                self.memory.batch_update(tree_idx, abs_errors)

        # self.steps += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def load_model(self, dir="model/dqn"):
        name = "dqn_agent_{}.pickle".format(self.iid)
        model_name = os.path.join(dir, name)
        # self.model.load_weights(model_name)
        self.model = pickle.load(open(f"{model_name}", "rb"))

    def save_model(self, dir="model/dqn"):
        name = "dqn_agent_{}.pickle".format(self.iid)
        model_name = os.path.join(dir, name)

        pickle.dump(self.model, file=open(f"{model_name}", "wb"))
