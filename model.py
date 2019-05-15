#coding=utf-8

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
from cell import LSTMCell
from gradient import PolicyGradient

def inference_graph(word_vocab_size=10000,  # configuration of medium
                    batch_size=20,
                    num_rnn_layers=2,
                    rnn_size=650,
                    num_unroll_steps=35,
                    n_actions=5,
                    dropout=0.0,
                    lamda=0.5
                    ):
    input_word = tf.placeholder(tf.int32, shape = [batch_size, num_unroll_steps], name = "input")
    with tf.variable_scope("Embedding"):


if __name__ == "__main__":
    with tf.Session() as sess:
        with tf.variable_scope("Model"):
