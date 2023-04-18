import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import random
from schedules import Sechedules
from replay_buffer_lenient import ReplayBuffer


class LHDQNAgent(object):

    def __init__(self, state_n, action_n, hidden_layers: list, scope_name: str,
                 learning_rate=1e-3, her=0.5, sess=None, discount=0.9, replay_memory_size=500000,
                 batch_size=32, begin_train=5000, lenient_start=0.5, lenient_end=0.01, lenient_decay_step=800000,
                 epsilon_start=0.8, epsilon_end=0, epsilon_decay_step=360000,
                 targetnet_update_freq=1000, seed=1, logdir='logs',
                 savedir='save', auto_save=True, save_freq=10000,
                 use_tau=False, tau=0.001):

        self.states_n = state_n
        self.actions_n = action_n
        self._hidden_layers = hidden_layers
        self._scope_name = scope_name
        self.lr = learning_rate
        self.her = her
        self._target_net_update_freq = targetnet_update_freq
        self._current_time_step = 0
        self.train_time = 0
        self.train_batch_size = batch_size
        self._begin_train = begin_train
        self._gamma = discount

        self.train_freq = 20

        self._use_tau = use_tau
        self._tau = tau

        self._auto_save = auto_save
        self.savedir = savedir
        self.save_freq = save_freq

        self.qnet_optimizer = tf.train.AdamOptimizer(self.lr)
        self.replay_buffer = ReplayBuffer(replay_memory_size)

        # self._seed(seed)

        # leniency part
        self.lenient_schedule = Sechedules(schedule_timesteps=lenient_decay_step, final_p=lenient_end,
                                           initial_p=lenient_start)
        self.epsilon_schedule = Sechedules(schedule_timesteps=epsilon_decay_step, final_p=epsilon_end,
                                           initial_p=epsilon_start)

        self.leniency = lenient_start
        self.epsilon = epsilon_start

        # self.ts_greedy_coeff = ts_greedy_coeff  # 0.25 0.5 1.0
        with tf.Graph().as_default():
            self._build_graph()
            self._merged_summary = tf.summary.merge_all()

            if sess is None:
                self.sess = tf.Session()
            else:
                self.sess = sess
            self.sess.run(tf.global_variables_initializer())

            self._saver = tf.train.Saver()

            self._summary_writer = tf.summary.FileWriter(logdir=logdir)
            self._summary_writer.add_graph(tf.get_default_graph())

    def show_memory(self):
        print(self.replay_buffer.show())

    def _q_network(self, state, hidden_layers, outputs, scope_name, trainable):
        with tf.variable_scope(scope_name):
            out = state
            for ly in hidden_layers:
                out = layers.fully_connected(out, ly, activation_fn=tf.nn.relu, trainable=trainable)
            out = layers.fully_connected(out, outputs, activation_fn=None, trainable=trainable)
        return out

    def _build_graph(self):
        self._state = tf.placeholder(dtype=tf.float32, shape=(None, self.states_n), name='state_input')
        with tf.variable_scope(self._scope_name):
            self._q_values = self._q_network(self._state, self._hidden_layers, self.actions_n, 'q_network', True)
            self._target_q_values = self._q_network(self._state, self._hidden_layers, self.actions_n,
                                                    'target_q_network', False)

        with tf.variable_scope('q_network_update'):
            self._actions_onehot = tf.placeholder(dtype=tf.float32, shape=(None, self.actions_n),
                                                  name='actions_onehot_input')
            self._td_targets = tf.placeholder(dtype=tf.float32, shape=(None,), name='td_targets')
            self._q_values_pred = tf.reduce_sum(self._q_values * self._actions_onehot, axis=1)

            # lenient
            self.importants = tf.placeholder(dtype=tf.float32, shape=(None,), name='important')
            # deltas = self._q_values_pred - self._td_targets

            deltas = self._td_targets - self._q_values_pred

            leniencies = tf.ones_like(self._td_targets) * self.leniency
            real_deltas = tf.where(tf.greater(deltas, tf.constant(0.0)), deltas * self.importants,
                                   deltas * (1.0 - leniencies) * self.importants)

            # real_deltas = tf.where(tf.greater(deltas, tf.zeros_like(self._td_targets)), deltas, deltas * self._leniencies)
            # real_deltas = deltas
            self._error = tf.abs(real_deltas)
            quadratic_part = tf.clip_by_value(self._error, 0.0, 1.0)
            linear_part = self._error - quadratic_part
            self._loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
            # self._loss = tf.reduce_mean(tf.square(real_deltas))

            qnet_gradients = self.qnet_optimizer.compute_gradients(self._loss, tf.trainable_variables())
            for i, (grad, var) in enumerate(qnet_gradients):
                if grad is not None:
                    qnet_gradients[i] = (tf.clip_by_norm(grad, 1), var)
            self.train_op = self.qnet_optimizer.apply_gradients(qnet_gradients)

            tf.summary.scalar('loss', self._loss)

            with tf.name_scope('target_network_update'):
                q_network_params = [t for t in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                 scope=self._scope_name + '/q_network')
                                    if t.name.startswith(self._scope_name + '/q_network/')]
                target_q_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                            scope=self._scope_name + '/target_q_network')

                self.target_update_ops = []
                for var, var_target in zip(sorted(q_network_params, key=lambda v: v.name),
                                           sorted(target_q_network_params, key=lambda v: v.name)):
                    # self.target_update_ops.append(var_target.assign(var))

                    # soft target update
                    self.target_update_ops.append(var_target.assign(tf.multiply(var_target, 1 - self._tau) +
                                                                    tf.multiply(var, self._tau)))
                self.target_update_ops = tf.group(*self.target_update_ops)

    def choose_action(self, state):
        # if epsilon is None:
        epsilon_used = self.epsilon
        # else:
        #     epsilon_used = 0.0

        if np.random.random() < epsilon_used:
            return np.random.randint(0, self.actions_n)
        else:
            q_values = self.sess.run(self._q_values, feed_dict={self._state: state[None]})

            return np.argmax(q_values[0])

    '''
        def choose_action_my(self, state, epsilon=True):
        # true 探索； false 利用
        #cur_phase = state['cur_phase']
        #lane_num_vehicle = state['lane_num_vehicle']
        #state = np.concatenate((cur_phase, lane_num_vehicle))
        if epsilon:
            return np.random.randint(0, self.actions_n)
        else:
            q_values = self.sess.run(self._q_values, feed_dict={self._state: state[None]})
            return np.argmax(q_values[0])
    '''

    def test_choose(self, state):
        q_values = self.sess.run(self._q_values, feed_dict={self._state: state[None]})

        return np.argmax(q_values[0])

    def check_network_output(self, state):
        q_values = self.sess.run(self._q_values, feed_dict={self._state: state})
        print(q_values[0])

    def store(self, state, action, reward, next_state, terminate):
        self.replay_buffer.add(state, action, reward, next_state, terminate)

    def erase(self):
        self.replay_buffer.erase()

    def get_max_target_Q_s_a(self, next_states):
        next_state_q_values = self.sess.run(self._q_values, feed_dict={self._state: next_states})
        next_state_target_q_values = self.sess.run(self._target_q_values, feed_dict={self._state: next_states})

        next_select_actions = np.argmax(next_state_q_values, axis=1)
        bt_sz = len(next_states)
        next_select_actions_onehot = np.zeros((bt_sz, self.actions_n))
        for i in range(bt_sz):
            next_select_actions_onehot[i, next_select_actions[i]] = 1.

        next_state_max_q_values = np.sum(next_state_target_q_values * next_select_actions_onehot, axis=1)
        return next_state_max_q_values

    def train(self):

        self._current_time_step += 1

        if self._current_time_step == 1:
            # print('Training starts.')
            self.sess.run(self.target_update_ops)

        if self._current_time_step > self._begin_train and self._current_time_step % self.train_freq == 0:
            self.train_time += self.train_freq
            self.epsilon = self.epsilon_schedule.update_linear(self.train_time)
            self.leniency = self.lenient_schedule.update_linear(self.train_time)

            states, actions, rewards, next_states, terminates, importants = self.replay_buffer.sample(
                batch_size=self.train_batch_size)
            # states, actions, rewards, next_states, terminates, importants = self.replay_buffer.encode_sample(index)
            actions_onehot = np.zeros((self.train_batch_size, self.actions_n))
            for i in range(self.train_batch_size):
                actions_onehot[i, actions[i]] = 1.
            next_state_q_values = self.sess.run(self._q_values, feed_dict={self._state: next_states})
            next_state_target_q_values = self.sess.run(self._target_q_values, feed_dict={self._state: next_states})
            next_select_actions = np.argmax(next_state_q_values, axis=1)
            next_select_actions_onehot = np.zeros((self.train_batch_size, self.actions_n))
            for i in range(self.train_batch_size):
                next_select_actions_onehot[i, next_select_actions[i]] = 1.

            next_state_max_q_values = np.sum(next_state_target_q_values * next_select_actions_onehot, axis=1)
            # next_state_max_q_values = self.get_max_target_Q_s_a(next_states)

            td_targets = rewards + self._gamma * next_state_max_q_values * (1 - terminates)

            _, str_ = self.sess.run([self.train_op, self._merged_summary],
                                    feed_dict={self._state: states, self._actions_onehot: actions_onehot,
                                               self._td_targets: td_targets, self.importants: importants})

            self._summary_writer.add_summary(str_, self._current_time_step)

            # update target_net
            if self._use_tau:
                self.sess.run(self.target_update_ops)
            else:
                if self._current_time_step % self._target_net_update_freq == 0:
                    self.sess.run(self.target_update_ops)

        # save model
        if self._auto_save:
            if self._current_time_step % self.save_freq == 0:
                # TODO save the model with highest performance
                self._saver.save(sess=self.sess, save_path=self.savedir + '/my-model',
                                 global_step=self._current_time_step)

    def train_without_replaybuffer(self, states, actions, target_values):

        self._current_time_step += 1

        if self._current_time_step == 1:
            # print('Training starts.')
            self.sess.run(self.target_update_ops)

        bt_sz = len(states)
        actions_onehot = np.zeros((bt_sz, self.actions_n))
        for i in range(bt_sz):
            actions_onehot[i, actions[i]] = 1.

        _, str_ = self.sess.run([self.train_op, self._merged_summary], feed_dict={self._state: states,
                                                                                  self._actions_onehot: actions_onehot,
                                                                                  self._td_targets: target_values
                                                                                  })

        self._summary_writer.add_summary(str_, self._current_time_step)

        # update target_net
        if self._use_tau:
            self.sess.run(self.target_update_ops)
        else:
            if self._current_time_step % self._target_net_update_freq == 0:
                self.sess.run(self.target_update_ops)

        # save model
        if self._auto_save:
            if self._current_time_step % self.save_freq == 0:
                # TODO save the model with highest performance
                self._saver.save(sess=self.sess, save_path=self.savedir + '/my-model',
                                 global_step=self._current_time_step)

    def save_model(self):
        self._saver.save(sess=self.sess, save_path=self.savedir + '/my-model',
                         global_step=self._current_time_step)

    def load_model(self):
        self._saver.restore(self.sess, tf.train.latest_checkpoint(self.savedir))

    def _seed(self, lucky_number):
        tf.set_random_seed(lucky_number)
        np.random.seed(lucky_number)
        random.seed(lucky_number)
