#coding=utf-8
import tensorflow as tf
import numpy as np

def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

def lstm_ortho_initializer(scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        size_x = shape[0]
        size_h = shape[1]/4 # assumes lstm.
        t = np.zeros(shape)
        t[:, :size_h] = orthogonal([size_x, size_h])*scale
        t[:, size_h:size_h*2] = orthogonal([size_x, size_h])*scale
        t[:, size_h*2:size_h*3] = orthogonal([size_x, size_h])*scale
        t[:, size_h*3:] = orthogonal([size_x, size_h])*scale
        return tf.constant(t, dtype)
  return _initializer

# Layer normalization
def layer_norm(x, num_units, scope="layer_norm", reuse=False, gamma_start=1.0, epsilon = 1e-3, use_bias=True):
    axes = [1]
    mean = tf.reduce_mean(x, axes, keep_dims = True)
    x_shifted = x - mean
    var = tf.reduce_mean(tf.square(x_shifted), axes, keep_dims = True)
    inv_std = tf.rsqrt(var + epsilon)
    with tf.variable_scope(scope):
        if reuse == True:
            tf.get_variable_score().reuse_variables()
        gamma = tf.get_variable('ln_gamma', [num_units], initializer=tf.constant_initializer(gamma_start))
        if use_bias:
            beta = tf.get_variable('ln_beta', [num_units], initializer=tf.constant_initializer(0.0))
    output = gamma*(x_shifted)*inv_std
    if use_bias:
        output = output + beta
    return output

class LSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, RL, forget_bias=1.0, use_layer_norm=False,
    use_recurrent_dropout=False, dropout_keep_prob=1.0):
        self.num_units = num_units
        self.RL = RL
        self.forget_bias = forget_bias
        self.use_layer_norm = use_layer_norm
        self.use_recurrent_dropout = use_recurrent_dropout
        self.dropout_keep_prob = dropout_keep_prob
        self.cur_time =0

    def get_newch(self, state, x, cur_time, RL):
        self.C_state = tf.concat([self.C_state[1:], tf.expand_dims(state[0],0)], 0)
        self.H_state = tf.concat([self.H_state[1:], tf.expand_dims(state[1],0)], 0)
        RL_input = tf.concat([x, tf.reduce_mean(self.H_state, 0)], 1)
        actions = self.RL.choose_action(RL_input, cur_time)
        coord = tf.concat([tf.cast(actions, tf.int32), tf.expand_dims(tf.range(tf.shape(actions)[0]),1)],1)

        self.c_state = tf.gather_nd(self.C_state, coord)
        self.h_state = tf.gather_nd(self.H_state, coord)

        return self.c_state, self.h_state

    def __call__(self, x, state, scope = None):
        with tf.variable_scope(scope or type(self).__name__):
            batch_size = x.get_shape().as_list()[0]
            x_size = x.get_shape().as_list()[1]

            cur_time = self.cur_time
            self.cur_time += 1

            if cur_time == 0:
                self.RL.actions = []
                self.RL.act_prob=[]
                c, h = state
                self.RL.choose_action(tf.concat([x,h],1), cur_time)
                self.C_state = tf.reshape(tf.tile(tf.reshape(tf.zeros_like(c, dtype=tf.float32),[-1]),[self.RL.n_actions]), [self.RL.n_actions, batch_size, self.num_units])
                self.H_state = tf.reshape(tf.tile(tf.reshape(tf.zeros_like(h, dtype=tf.float32),[-1]),[self.RL.n_actions]), [self.RL.n_actions, batch_size, self.num_units])
            else:
                c, h = self.get_newch(state, x, cur_time, self.RL)

            h_size = self.num_units
            w_init=None # uniform

            h_init=lstm_ortho_initializer()

            W_xh = tf.get_variable("W_xh", [x_size, 4 * self.num_units], initializer=w_init)
            W_hh = tf.get_variable("W_hh", [self.num_units, 4 * self.num_units], initializer = h_init)
            W_full = tf.concat([W_xh, W_hh], 0)

            bias = tf.get_variable("bias", [4 * self.num_units], initializer = tf.Constant(0.0))

            concat = tf.concat([x, h], 1)
            concat = tf.matmul(concat, W_full) + bias

            i, j, f, o = tf.split(concat, 4, 1)

            if self.use_recurrent_dropout:
                g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
            else:
                g = tf.tanh(j)

            new_c = c * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) * g
            if self.use_layer_norm:
                new_h = tf.tanh(layer_norm(new_c, self.num_units, 'ln_c')) * tf.sigmoid(o)
            else:
                new_h = tf.tanh(new_c) * tf.sigmoid(o)
        tf.contrib.rnn.LSTMStateTuple(new_c, new_h)