#coding=utf-8
import tensorflow as tf
import numpy as np

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