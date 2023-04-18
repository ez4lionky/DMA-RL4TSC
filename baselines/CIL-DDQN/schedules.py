import tensorflow as tf
#import tensorflow.contrib as tc
import numpy as np
import math
flags = tf.app.flags

class Optimistic_C51():
    def __init__(self, atoms=51, vmax=15, vmin=-6, op_pa=2, op_pb=0.002):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.atoms = atoms
        # self.start_train = False

        self.v_max = vmax
        self.v_min = vmin
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1.)
        z = np.tile(np.asarray([self.v_min + i * self.delta_z for i in range(self.atoms)]).astype(np.float32), (1, 1))
        self.z = tf.transpose(tf.convert_to_tensor(z))
        #self.z = np.tile(np.asarray([self.v_min + i * self.delta_z for i in range(self.atoms)]).astype(np.float32),(batchsize, 1))  # shape (BATCH_SIZE,atoms)
        #self.optimistic = 1 / (1 + op_pa * math.exp(0))
        #self.batchsize = batchsize
        self.op_pa = op_pa
        self.op_pb = op_pb

    def optimistic_qvalue(self, optimistic, c51_output):
        matrix = np.zeros([self.atoms, self.atoms], dtype=np.float32)
        for i in range(self.atoms):
            for j in range(i):
                matrix[i, j] = 1
        matrix = tf.convert_to_tensor(matrix)
        # matrix = [[1,0 ,...,0],
        #           [1,1, ...,0],
        #           ...,
        #           [1,1,...,1]]
        optimistic_q_1 = tf.matmul(c51_output, matrix)
        m_optimistic = optimistic * tf.ones_like(optimistic_q_1)
        optimistic_q_2 = tf.where(tf.greater(optimistic_q_1, m_optimistic), tf.zeros_like(optimistic_q_1),
                                  c51_output)
        return tf.matmul(tf.nn.softmax(optimistic_q_2, axis=1), self.z), tf.nn.softmax(optimistic_q_2, axis=1)

    def non_optimistic_qvalue(self, c51_output):
        return tf.matmul(tf.nn.softmax(c51_output, axis=1), self.z)

    def c51_target_value(self, c51_target_output, rewards, dones, gamma):
        #z = np.tile(self.z, (bt_sz, 1))  # shape (BATCH_SIZE,atoms)
        bt_sz = len(rewards)
        z = np.tile(np.asarray([self.v_min + i * self.delta_z for i in range(self.atoms)]).astype(np.float32), (bt_sz, 1))

        Tz = np.minimum(self.v_max, np.maximum(self.v_min,
                                               rewards[:, np.newaxis] + gamma * z * (1.0 - dones[:, np.newaxis])))
        b = (Tz - self.v_min) / self.delta_z
        u = np.ceil(b + 1e-3).astype(int)
        l = u - 1

        p = c51_target_output
        m_batch = np.zeros((bt_sz, self.atoms))
        A = p * (u - b)
        B = p * (b - l)
        for i in range(bt_sz):
            for j in range(self.atoms):
                m_batch[i, l[i, j]] += A[i, j]
                m_batch[i, u[i, j]] += B[i, j]
        #cross_entropy_loss = -tf.reduce_sum(m_batch * tf.log(c51_output))
        target_q = m_batch

        # tf.reduce_sum(self.q_value_output, axis=1)
        return target_q


    def update_optimistic(self, step):
        return 1 / (1 + self.op_pa * math.exp(- self.op_pb * step))

class Sechedules():
    def __init__(self, schedule_timesteps=50000, final_p=0.01, initial_p=1.0, op_pa=20, op_pb=0.0002):
        """Linear interpolation between initial_p and final_p over
                schedule_timesteps. After this many timesteps pass final_p is
                returned.
                Parameters
                ----------
                schedule_timesteps: int
                    Number of timesteps for which to linearly anneal initial_p
                    to final_p
                initial_p: float
                    initial output value
                final_p: float
                    final output value
                """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.op_pa = op_pa
        self.op_pb = op_pb

    def update_linear(self, step):
        """See Schedule.value"""
        fraction = min(float(step) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

    def update_optimistic(self, step):
        return 1 - 1 / (1 + self.op_pa * math.exp(- self.op_pb * step))


