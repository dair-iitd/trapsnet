import pickle as pkl
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf

from gat.models import GAT


class PolicyEstimator():
    """
    Policy Function approximator. Given a observation, returns probabilities
    over all possible actions.

    Args:
      num_outputs: Size of the action space.
      reuse: If true, an existing shared network will be re-used.
      trainable: If true we add train ops to the network.
        Actor threads that don't update their local models and don't need
        train ops would set this to false.
    """

    def __init__(self,
                 num_inputs_list,
                 N,
                 num_gcn_hidden,
                 num_action_dim,
                 num_decoder_dim,
                 num_outputs_list,
                 fluent_feature_dims,
                 nonfluent_feature_dims,
                 activation="lrelu",
                 learning_rate=5e-5,
                 reuse=False,
                 trainable=True):
        self.num_inputs_list = num_inputs_list
        self.fluent_feature_dims = fluent_feature_dims
        self.nonfluent_feature_dims = nonfluent_feature_dims
        self.feature_dims = fluent_feature_dims + nonfluent_feature_dims
        self.input_size_list = [(int(num_inputs / self.fluent_feature_dims),
                                 self.feature_dims)
                                for num_inputs in self.num_inputs_list]
        self.num_gcn_hidden = num_gcn_hidden
        self.num_action_dim = num_action_dim
        self.num_decoder_dim = num_decoder_dim
        self.num_outputs_list = [i[0] for i in self.input_size_list]
        self.num_supports = 1
        self.activation = activation
        if activation == "relu":
            self.activation_fn = tf.nn.relu
        if activation == "lrelu":
            self.activation_fn = tf.nn.leaky_relu
        if activation == "elu":
            self.activation_fn = tf.nn.elu

        self.N = N
        self.learning_rate = learning_rate

        self.lambda_tr = 1.0
        self.lambda_entropy = 1.0
        self.lambda_grad = 0.1

        # Placeholders for our input

        self.states_list = [
            tf.placeholder(
                shape=[None, num_inputs],
                dtype=tf.uint8,
                name="X_{}".format(i))
            for i, num_inputs in enumerate(self.num_inputs_list)
        ]

        self.instance = tf.placeholder(
            shape=[None], dtype=tf.int32, name="instance")

        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.env_num = tf.placeholder(tf.int32, name="env_num")
        tf.summary.scalar("env_num", self.env_num)

        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions = tf.placeholder(
            shape=[None], dtype=tf.int32, name="actions")

        self.inputs = tf.placeholder(
            dtype=tf.float32,
            shape=(None, None, self.feature_dims),
            name="inputs")
        self.adj_biases_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=(None, None, None),
            name="adj_biases_placeholder")
        self.is_train = tf.placeholder(
            dtype=tf.bool, shape=(), name="is_train")

        # Build network
        with tf.variable_scope("policy_net", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("gat1", reuse=tf.AUTO_REUSE):
                gat1 = GAT.inference(
                    self.inputs,
                    self.num_gcn_hidden,
                    self.input_size_list[0][0],
                    self.is_train,
                    0.0,
                    0.0,
                    bias_mat=self.adj_biases_placeholder,
                    hid_units=[8],
                    n_heads=[8, 1],
                    residual=False,
                    activation=self.activation_fn)

                self.gcn_hidden = gat1

            self.action_embedding1 = tf.layers.dense(
                inputs=self.gcn_hidden,
                units=self.num_action_dim,
                activation=self.activation_fn,
                name="action_embedding1")

            self.action_embedding1_flat = tf.reshape(self.action_embedding1, [
                self.batch_size,
                tf.cast(
                    tf.gather(self.input_size_list, self.env_num)[0],
                    tf.int32), self.num_action_dim
            ])
            self.graph_embedding = tf.reduce_max(
                self.action_embedding1_flat, axis=1, name="graph_embedding")

            self.graph_embedding_repeat = tf.reshape(
                tf.tile(self.graph_embedding, [
                    1,
                    tf.cast(
                        tf.gather(self.input_size_list, self.env_num)[0],
                        tf.int32)
                ]), tf.shape(self.action_embedding1))

            self.node_state_embedding_concat = tf.concat(
                values=[self.action_embedding1, self.graph_embedding_repeat],
                axis=2,
                name="node_state_embedding")

            self.action_embedding2 = tf.layers.dense(
                inputs=self.node_state_embedding_concat,
                units=self.num_decoder_dim,
                activation=self.activation_fn,
                name="action_embedding2")

            self.action_embedding3 = tf.layers.dense(
                inputs=self.action_embedding2,
                units=1,
                activation=self.activation_fn,
                name="action_embedding3")

            self.decoder_hidden_list = [None] * self.N
            self.probs_list = [None] * self.N
            self.predictions_list = [None] * self.N
            self.entropy_list = [None] * self.N
            self.entropy_mean_list = [None] * self.N
            self.picked_action_probs_list = [None] * self.N
            self.losses_list = [None] * self.N
            self.loss_list = [None] * self.N
            self.final_loss_list = [None] * self.N
            self.grads_and_vars_list = [None] * self.N
            self.train_op_list = [None] * self.N

            for i in range(self.N):

                self.decoder_hidden_list[i] = tf.reshape(
                    self.action_embedding3, [
                        self.batch_size,
                        tf.gather(self.input_size_list, self.env_num)[0]
                    ])

                self.probs_list[i] = tf.nn.softmax(
                    self.decoder_hidden_list[i]) + 1e-8

                self.predictions_list[i] = {
                    "logits": self.decoder_hidden_list[i],
                    "probs": self.probs_list[i]
                }

                # We add entropy to the loss to encourage exploration
                self.entropy_list[i] = - \
                    tf.reduce_sum(self.probs_list[i] * tf.log(self.probs_list[i]),
                                  1, name="entropy_{}".format(i))
                self.entropy_mean_list[i] = tf.reduce_mean(
                    self.entropy_list[i], name="entropy_mean_{}".format(i))

                # Get the predictions for the chosen actions only
                gather_indices = tf.range(self.batch_size) * \
                    tf.shape(self.probs_list[i])[1] + self.actions
                self.picked_action_probs_list[i] = tf.gather(
                    tf.reshape(self.probs_list[i], [-1]), gather_indices)

                self.losses_list[i] = -(
                    tf.log(self.picked_action_probs_list[i]) * self.targets +
                    self.lambda_entropy * self.entropy_list[i])
                self.loss_list[i] = tf.reduce_sum(
                    self.losses_list[i], name="loss_{}".format(i))

                self.final_loss_list[i] = self.loss_list[i]
                # tf.summary.scalar(self.loss_list[i].op.name, self.loss_list[i])
                # tf.summary.scalar(self.transition_loss_list[i].op.name,
                #                   self.transition_loss_list[i])
                # tf.summary.scalar(self.final_loss_list[i].op.name,
                #                   self.final_loss_list[i])
                # tf.summary.scalar(self.entropy_mean_list[i].op.name,
                #                   self.entropy_mean_list[i])
                # tf.summary.histogram(self.entropy_list[i].op.name,
                #                      self.entropy_list[i])

                if trainable:
                    # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    self.optimizer = tf.train.RMSPropOptimizer(
                        self.learning_rate, 0.99, 0.0, 1e-6)
                    self.grads_and_vars_list[
                        i] = self.optimizer.compute_gradients(
                            self.final_loss_list[i])
                    self.grads_and_vars_list[i] = [[
                        grad, var
                    ] for grad, var in self.grads_and_vars_list[i]
                                                   if grad is not None]
                    self.train_op_list[i] = self.optimizer.apply_gradients(
                        self.grads_and_vars_list[i],
                        global_step=tf.contrib.framework.get_global_step())
        # Merge summaries from this network and the shared network (but not the value net)
        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [
            s for s in summary_ops
            if "policy_net" in s.name or "shared" in s.name
        ]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)


class ValueEstimator():
    """
    Value Function approximator. Returns a value estimator for a batch of observations.

    Args:
      reuse: If true, an existing shared network will be re-used.
      trainable: If true we add train ops to the network.
        Actor threads that don't update their local models and don't need
        train ops would set this to false.
    """

    def __init__(self,
                 num_inputs_list,
                 N,
                 num_gcn_hidden,
                 num_action_dim,
                 num_decoder_dim,
                 fluent_feature_dims,
                 nonfluent_feature_dims,
                 activation="elu",
                 learning_rate=5e-5,
                 reuse=False,
                 trainable=True):
        self.num_inputs_list = num_inputs_list
        self.fluent_feature_dims = fluent_feature_dims
        self.nonfluent_feature_dims = nonfluent_feature_dims
        self.feature_dims = fluent_feature_dims + nonfluent_feature_dims
        self.input_size_list = [(int(num_inputs / self.fluent_feature_dims),
                                 self.feature_dims)
                                for num_inputs in self.num_inputs_list]
        self.num_gcn_hidden = num_gcn_hidden
        self.num_action_dim = num_action_dim
        self.num_decoder_dim = num_decoder_dim
        self.num_outputs = 1
        self.activation = activation

        if activation == "relu":
            self.activation_fn = tf.nn.relu
        if activation == "lrelu":
            self.activation_fn = tf.nn.leaky_relu
        if activation == "elu":
            self.activation_fn = tf.nn.elu

        self.N = N
        self.learning_rate = learning_rate

        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.env_num = tf.placeholder(tf.int32, name="env_num")
        # Build network
        # TODO: add support

        self.inputs = tf.placeholder(
            dtype=tf.float32,
            shape=(None, None, self.feature_dims),
            name="inputs")
        self.adj_biases_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=(None, None, None),
            name="adj_biases_placeholder")
        self.is_train = tf.placeholder(
            dtype=tf.bool, shape=(), name="is_train")

        with tf.variable_scope("value_net"):
            with tf.variable_scope("gat1", reuse=tf.AUTO_REUSE):
                gat1 = GAT.inference(
                    self.inputs,
                    self.num_gcn_hidden,
                    self.input_size_list[0][0],
                    self.is_train,
                    0.0,
                    0.0,
                    bias_mat=self.adj_biases_placeholder,
                    hid_units=[8],
                    n_heads=[8, 1],
                    residual=False,
                    activation=self.activation_fn)

                self.gcn_hidden = gat1

            self.embedding1 = tf.layers.dense(
                inputs=self.gcn_hidden,
                units=self.num_action_dim,
                activation=self.activation_fn,
                name="embedding1")

            self.embedding1_flat = tf.reshape(self.embedding1, [
                self.batch_size,
                tf.cast(
                    tf.gather(self.input_size_list, self.env_num)[0],
                    tf.int32), self.num_action_dim
            ])
            self.graph_embedding = tf.reduce_max(
                self.embedding1_flat, axis=1, name="graph_embedding")

            self.graph_embedding_repeat = tf.reshape(
                tf.tile(self.graph_embedding, [
                    1,
                    tf.cast(
                        tf.gather(self.input_size_list, self.env_num)[0],
                        tf.int32)
                ]), tf.shape(self.embedding1))

            self.node_state_embedding_concat = tf.concat(
                values=[self.embedding1, self.graph_embedding_repeat],
                axis=2,
                name="node_state_embedding")

            self.embedding2 = tf.layers.dense(
                inputs=self.node_state_embedding_concat,
                units=self.num_decoder_dim,
                activation=self.activation_fn,
                name="embedding2")

            self.embedding3 = tf.layers.dense(
                inputs=self.embedding2,
                units=1,
                activation=self.activation_fn,
                name="embedding3")

            # Common summaries
            prefix = tf.get_variable_scope().name
            # tf.contrib.layers.summarize_activation(self.gcn_hidden1)
            # tf.summary.scalar("{}/reward_max".format(prefix),
            #                   tf.reduce_max(self.targets))
            # tf.summary.scalar("{}/reward_min".format(prefix),
            #                   tf.reduce_min(self.targets))
            tf.summary.scalar("{}/reward_mean".format(prefix),
                              tf.reduce_mean(self.targets))
            # tf.summary.histogram("{}/reward_targets".format(prefix),
            #                      self.targets)

            self.hidden_list = [None] * self.N
            self.logits_list = [None] * self.N
            self.predictions_list = [None] * self.N
            self.losses_list = [None] * self.N
            self.loss_list = [None] * self.N
            self.grads_and_vars_list = [None] * self.N
            self.train_op_list = [None] * self.N

            for i in range(self.N):

                self.hidden_list[i] = tf.reshape(
                    self.embedding3,
                    [self.batch_size, self.input_size_list[i][0]])

                self.logits_list[i] = tf.reduce_sum(
                    self.hidden_list[i], axis=1, name="logits_{}".format(i))

                self.losses_list[i] = tf.squared_difference(
                    self.logits_list[i], self.targets)
                self.loss_list[i] = tf.reduce_sum(
                    self.losses_list[i], name="loss_{}".format(i))

                self.predictions_list[i] = {"logits": self.logits_list[i]}

                # Summaries
                # tf.summary.scalar(self.loss_list[i].name, self.loss_list[i])
                # tf.summary.scalar("{}/max_value_{}".format(prefix, i),
                #                   tf.reduce_max(self.logits_list[i]))
                # tf.summary.scalar("{}/min_value_{}".format(prefix, i),
                #                   tf.reduce_min(self.logits_list[i]))
                # tf.summary.scalar("{}/mean_value_{}".format(prefix, i),
                #                   tf.reduce_mean(self.logits_list[i]))
                # tf.summary.histogram("{}/values_{}".format(prefix, i),
                #                      self.logits_list[i])

                if trainable:
                    # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    self.optimizer = tf.train.RMSPropOptimizer(
                        self.learning_rate, 0.99, 0.0, 1e-6)
                    self.grads_and_vars_list[
                        i] = self.optimizer.compute_gradients(
                            self.loss_list[i])
                    self.grads_and_vars_list[i] = [[
                        grad, var
                    ] for grad, var in self.grads_and_vars_list[i]
                                                   if grad is not None]
                    self.train_op_list[i] = self.optimizer.apply_gradients(
                        self.grads_and_vars_list[i],
                        global_step=tf.contrib.framework.get_global_step())

        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [
            s for s in summary_ops
            if "value_net" in s.name or "shared" in s.name
        ]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)
