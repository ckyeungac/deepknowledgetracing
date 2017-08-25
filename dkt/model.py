import numpy as np
import pandas as pd
import tensorflow as tf


def length(sequence):
    """
    This function return the sequence length of each x in the batch.
    :param sequence: the batch sequence of shape [batch_size, num_steps, feature_size]
    :return length: A tensor of shape [batch_size]
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    seq_length = tf.reduce_sum(used, 1)
    seq_length = tf.cast(seq_length, tf.int32)
    return seq_length

# reference:
# https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py
# https://github.com/davidoj/deepknowledgetracingTF/blob/master/model.py
class BasicModel(object):
    def __init__(self, num_problems, **kwargs):
        # dataset-dependent attributes
        self.num_problems = num_problems

        # network configuration
        self.hidden_layer_structure = kwargs.get('hidden_layer_structure', (200,))
        self.batch_size = kwargs.get('batch_size', 32)
        self.rnn_cell = kwargs.get('rnn_cell', tf.contrib.rnn.LSTMCell)
        self.learning_rate = kwargs.get('learning_rate', 0.01)

    def build_graph(self):
        hidden_layer_structure = self.hidden_layer_structure
        num_problems = self.num_problems

        # placeholder
        X = self._X = tf.placeholder(tf.float32, [None, None, 2*num_problems], name='X')
        y_seq = self._y_seq = tf.placeholder(tf.float32, [None, None, num_problems], name='y_seq')
        y_corr = self._y_corr = tf.placeholder(tf.float32, [None, None, num_problems], name='y_corr')
        keep_prob = self._keep_prob = tf.placeholder(tf.float32)

        # Hidden Layer Construction
        hidden_layers_outputs = self._hidden_layers_outputs = []
        hidden_layers_states = self._hidden_layers_state = []
        hidden_layer_input = X
        for i, layer_state_size in enumerate(hidden_layer_structure):
            variable_scope_name = "hidden_layer_{}".format(i)
            with tf.variable_scope(variable_scope_name, reuse=tf.get_variable_scope().reuse):
                cell = self.rnn_cell(num_units=layer_state_size)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
                outputs, state = tf.nn.dynamic_rnn(
                    cell,
                    hidden_layer_input,
                    dtype=tf.float32,
                    sequence_length=length(X)
                )
            hidden_layers_outputs.append(outputs)
            hidden_layers_states.append(state)
            hidden_layer_input = outputs

        last_layer_size = hidden_layer_structure[-1]
        last_layer_outputs = hidden_layers_outputs[-1]
        last_layer_state = hidden_layers_states[-1]

        # Output Layer Construction
        with tf.variable_scope("output_layer", reuse=tf.get_variable_scope().reuse):
            W_yh = tf.get_variable("weights", shape=[last_layer_size, num_problems],
                                   initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(num_problems)))
            b_yh = tf.get_variable("biases", shape=[num_problems, ],
                                   initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(num_problems)))

            # Flatten the last layer output
            outputs_flat = tf.reshape(last_layer_outputs, shape=[-1, last_layer_size])
            logits_flat = tf.matmul(outputs_flat, W_yh) + b_yh
            preds_flat = self._preds_flat = tf.sigmoid(logits_flat)
            y_seq_flat = tf.cast(tf.reshape(y_seq, [-1, num_problems]), dtype=tf.float32)
            y_corr_flat = tf.cast(tf.reshape(y_corr, [-1, num_problems]), dtype=tf.float32)

        # Filter out the target indices as follow:
        # Get the indices where y_seq_flat are not equal to 0, where the indices
        # implies that a student has answered the question in the time step and
        # thereby exclude those time step that the student hasn't answered.
        target_indices = tf.where(tf.not_equal(y_seq_flat, 0))

        target_logits = self._target_logits = tf.gather_nd(logits_flat, target_indices)
        target_preds = self._target_preds = tf.gather_nd(preds_flat, target_indices)  # needed to return AUC
        target_labels = self._target_labels = tf.gather_nd(y_corr_flat, target_indices)

        total_loss = self._total_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=target_logits,
                                                                                labels=target_labels)

        loss = self._loss = tf.reduce_mean(total_loss)
        optimizer = self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # Network successfully defined, reuse the variables later on
        tf.get_variable_scope().reuse_variables()

    @property
    def X(self):
        return self._X

    @property
    def y_seq(self):
        return self._y_seq

    @property
    def y_corr(self):
        return self._y_corr

    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def hidden_layers_outputs(self):
        return self._hidden_layers_outputs

    @property
    def hidden_layers_state(self):
        return self._hidden_layers_state

    @property
    def preds_flat(self):
        return self._preds_flat

    @property
    def target_logits(self):
        return self._target_logits

    @property
    def target_preds(self):
        return self._target_preds

    @property
    def target_labels(self):
        return self._target_labels

    @property
    def total_loss(self):
        return self._total_loss

    @property
    def loss(self):
        return self._loss

    @property
    def optimizer(self):
        return self._optimizer


class GaussianNoiseInputModel(BasicModel):
    def __init__(self, num_problems, **kwargs):
        super(GaussianNoiseInputModel, self).__init__(num_problems, **kwargs)

    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    def build_graph(self):
        hidden_layer_structure = self.hidden_layer_structure
        num_problems = self.num_problems

        # placeholder
        X = self._X = tf.placeholder(tf.float32, [None, None, 2*num_problems], name='X')
        y_seq = self._y_seq = tf.placeholder(tf.float32, [None, None, num_problems], name='y_seq')
        y_corr = self._y_corr = tf.placeholder(tf.float32, [None, None, num_problems], name='y_corr')
        keep_prob = self._keep_prob = tf.placeholder(tf.float32)
        gaussian_std = self._gaussian_std = tf.placeholder(tf.float32)

        X_noised = tf.reshape(self.gaussian_noise_layer(X, std=gaussian_std), shape=tf.shape(X))

        # Hidden Layer Construction
        hidden_layers_outputs = self._hidden_layers_outputs = []
        hidden_layers_states = self._hidden_layers_state = []
        hidden_layer_input = X_noised
        for i, layer_state_size in enumerate(hidden_layer_structure):
            variable_scope_name = "hidden_layer_{}".format(i)
            with tf.variable_scope(variable_scope_name, reuse=tf.get_variable_scope().reuse):
                cell = self.rnn_cell(num_units=layer_state_size)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
                outputs, state = tf.nn.dynamic_rnn(
                    cell,
                    hidden_layer_input,
                    dtype=tf.float32,
                    sequence_length=length(X)
                )
            hidden_layers_outputs.append(outputs)
            hidden_layers_states.append(state)
            hidden_layer_input = outputs

        last_layer_size = hidden_layer_structure[-1]
        last_layer_outputs = hidden_layers_outputs[-1]
        last_layer_state = hidden_layers_states[-1]

        # Output Layer Construction
        with tf.variable_scope("output_layer", reuse=tf.get_variable_scope().reuse):
            W_yh = tf.get_variable("weights", shape=[last_layer_size, num_problems],
                                   initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(num_problems)))
            b_yh = tf.get_variable("biases", shape=[num_problems, ],
                                   initializer=tf.random_normal_initializer(stddev=1.0 / np.sqrt(num_problems)))

            # Flatten the last layer output
            outputs_flat = tf.reshape(last_layer_outputs, shape=[-1, last_layer_size])
            logits_flat = tf.matmul(outputs_flat, W_yh) + b_yh
            preds_flat = self._preds_flat = tf.sigmoid(logits_flat)
            y_seq_flat = tf.cast(tf.reshape(y_seq, [-1, num_problems]), dtype=tf.float32)
            y_corr_flat = tf.cast(tf.reshape(y_corr, [-1, num_problems]), dtype=tf.float32)

        # Filter out the target indices as follow:
        # Get the indices where y_seq_flat are not equal to 0, where the indices
        # implies that a student has answered the question in the time step and
        # thereby exclude those time step that the student hasn't answered.
        target_indices = tf.where(tf.not_equal(y_seq_flat, 0))

        target_logits = self._target_logits = tf.gather_nd(logits_flat, target_indices)
        target_preds = self._target_preds = tf.gather_nd(preds_flat, target_indices)  # needed to return AUC
        target_labels = self._target_labels = tf.gather_nd(y_corr_flat, target_indices)

        total_loss = self._total_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=target_logits,
                                                                                labels=target_labels)

        loss = self._loss = tf.reduce_mean(total_loss)
        optimizer = self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # Network successfully defined, reuse the variables later on
        tf.get_variable_scope().reuse_variables()

    @property
    def gaussian_std(self):
        return self._gaussian_std