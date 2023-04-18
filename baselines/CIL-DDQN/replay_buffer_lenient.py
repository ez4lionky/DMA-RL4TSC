import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.storage = 0
        self.maxsize = size
        self.next_idx = 0
        self.obs_t, self.action, self.reward, self.obs_tp1, self.done, self.important = [], [], [], [], [], []

    def __len__(self):
        return self.storage

    def show(self):
        for item in range(self.storage):
            print(self.obs_t[item], self.action[item], self.reward[item],
                  self.obs_tp1[item], self.done[item], self.important[item])

        # print(np.sum(self._storage == (np.array([6]), 1, 1.0, np.array([1]), 1.0)))

    def add(self, obs_t, action, reward, obs_tp1, done):
        # data = (obs_t, action, reward, obs_tp1, done, 1.0)
        if self.next_idx >= self.storage:
            self.storage += 1
            self.obs_t.append(obs_t)
            self.action.append(action)
            self.reward.append(reward)
            self.obs_tp1.append(obs_tp1)
            self.done.append(done)
            self.important.append(1.0)
        else:
            self.obs_t[self.next_idx] = obs_t
            self.action[self.next_idx] = action
            self.reward[self.next_idx] = reward
            self.obs_tp1[self.next_idx] = obs_tp1
            self.done[self.next_idx] = done
            self.important[self.next_idx] = 1.0
        self.next_idx = (self.next_idx + 1) % self.maxsize

    def update_important(self, decrease_rate):
        for i in range(self.storage):
            self.important[i] *= decrease_rate

    def encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, importants = [], [], [], [], [], []
        for item in idxes:
            # data = self.storage[i]
            obs_t, action, reward, obs_tp1, done, important = self.obs_t[item], self.action[item], self.reward[item], \
                                                              self.obs_tp1[item], self.done[item], self.important[item]
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            importants.append(important)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(
            importants)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, self.storage - 1) for _ in range(batch_size)]
        return self.encode_sample(idxes)

    def erase(self):
        self.storage = 0
        self.next_idx = 0
