import pickle as pkl
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf

from gcn.utils import *
from gcn.layers import GraphConvolution
from gcn.models import GCN, MLP


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
                 num_attention_dim,
                 num_rl_hidden,
                 num_hidden_transition,
                 num_action_embed,
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
        self.input_size_list = [(num_inputs / self.fluent_feature_dims,
                                 self.feature_dims)
                                for num_inputs in self.num_inputs_list]
        self.num_gcn_hidden = num_gcn_hidden
        self.num_attention_dim = num_attention_dim
        self.num_rl_hidden = num_rl_hidden
        self.num_hidden_transition = num_hidden_transition
        self.num_action_embed = num_action_embed
        self.num_outputs_list = num_outputs_list
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
        self.inputs = tf.sparse_placeholder(
            tf.float32, shape=[None, self.feature_dims], name="inputs")
        self.placeholders_hidden = {
            'support': [tf.sparse_placeholder(tf.float32, name="support")],
            'dropout': tf.placeholder_with_default(
                0., shape=(), name="dropout"),
            # helper variable for sparse dropout
            'num_features_nonzero': tf.placeholder(tf.int32)
        }
        self.instance = tf.placeholder(
            shape=[None], dtype=tf.int32, name="instance")

        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.env_num = tf.placeholder(tf.int32, name="env_num")

        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions = tf.placeholder(
            shape=[None], dtype=tf.int32, name="actions")
        self.action_probs_list = [
            tf.placeholder(
                shape=[None, num_outputs],
                dtype=tf.float32,
                name="action_probs_{}".format(i))
            for i, num_outputs in enumerate(self.num_outputs_list)
        ]

        # Build network
        with tf.variable_scope("policy_net", reuse=tf.AUTO_REUSE):
            gconv1 = GraphConvolution(
                input_dim=self.feature_dims,
                output_dim=self.num_gcn_hidden,
                placeholders=self.placeholders_hidden,
                act=self.activation_fn,
                dropout=True,
                sparse_inputs=True,
                name='gconv1',
                logging=True)
            self.gcn_hidden = gconv1(self.inputs)

            self.attention_output = tf.layers.dense(
                inputs=self.gcn_hidden,
                units=self.num_attention_dim,
                use_bias=False,
                name="attention",
                reuse=tf.AUTO_REUSE)

            self.gcn_hidden_flat = tf.reshape(self.attention_output, [
                self.batch_size,
                tf.cast(
                    tf.gather(self.input_size_list, self.env_num)[0],
                    tf.int32), self.num_attention_dim
            ])
            self.gcn_max = tf.reduce_max(
                self.gcn_hidden_flat, axis=1, name="gcn_max")

            self.decoder_hidden_list = [None] * self.N
            self.decoder_transition_list = [None] * self.N
            self.probs_list = [None] * self.N
            self.predictions_list = [None] * self.N
            self.entropy_list = [None] * self.N
            self.entropy_mean_list = [None] * self.N
            self.picked_action_probs_list = [None] * self.N
            self.losses_list = [None] * self.N
            self.loss_list = [None] * self.N
            self.transition_loss_list = [None] * self.N
            self.final_loss_list = [None] * self.N
            self.grads_and_vars_list = [None] * self.N
            self.train_op_list = [None] * self.N

            self.rl_hidden = tf.layers.dense(
                inputs=self.gcn_max,
                units=self.num_rl_hidden,
                activation=self.activation_fn,
                name="rl_hidden")

            self.logits = tf.layers.dense(
                inputs=self.rl_hidden,
                units=self.num_action_embed,
                activation=self.activation_fn,
                name="logits_hidden")

            # Transition model
            self.current_state_embeding_flat = self.gcn_max[1:]
            self.next_state_embeding_flat = self.gcn_max[:-1]

            self.current_states_list = [
                states[1:] for states in self.states_list
            ]

            self.transition_states_concat = tf.concat(
                [
                    self.current_state_embeding_flat,
                    self.next_state_embeding_flat
                ],
                axis=1)

            self.transition_hidden1 = tf.layers.dense(
                inputs=self.transition_states_concat,
                units=self.num_hidden_transition,
                activation=self.activation_fn,
                name="transition_hidden1")

            self.transition_hidden2 = tf.layers.dense(
                inputs=self.transition_hidden1,
                units=self.num_action_embed,
                activation=self.activation_fn,
                name="transition_hidden2")

            self.state_action_concat_list = [
                tf.concat(
                    [self.logits, tf.cast(states, tf.float32)], axis=1)
                for states in self.states_list
            ]
            self.state_action_concat_transition_list = [
                tf.concat(
                    [
                        self.transition_hidden2,
                        tf.cast(current_states, tf.float32)
                    ],
                    axis=1) for current_states in self.current_states_list
            ]

            # state classifier
            self.classifier_logits = tf.layers.dense(
                inputs=self.logits,
                units=self.N,
                activation=self.activation_fn,
                name="classifier_layer")

            self.classification_prob = tf.nn.softmax(
                self.classifier_logits) + 1e-8

            # instance classification loss
            self.instance_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.instance, logits=self.classifier_logits),
                name="instance_loss")
            # tf.summary.scalar("instance_loss", self.instance_loss)

            #transition classifier
            self.transition_classifier_logits = tf.layers.dense(
                inputs=self.transition_hidden2,
                units=self.N,
                activation=self.activation_fn,
                name="transition_classifier_layer")

            self.transition_classification_prob = tf.nn.softmax(
                self.transition_classifier_logits) + 1e-8

            # instance classification loss
            self.transition_classification_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.instance[:-1],
                    logits=self.transition_classifier_logits),
                name="transition_classification_loss")

            for i in range(self.N):

                self.decoder_hidden_list[i] = tf.layers.dense(
                    inputs=self.state_action_concat_list[i],
                    units=self.num_outputs_list[i],
                    activation=self.activation_fn,
                    reuse=tf.AUTO_REUSE,
                    name="output_{}".format(i))

                self.decoder_transition_list[i] = tf.layers.dense(
                    inputs=self.state_action_concat_transition_list[i],
                    units=self.num_outputs_list[i],
                    activation=self.activation_fn,
                    reuse=tf.AUTO_REUSE,
                    name="output_{}".format(i))

                self.probs_list[i] = tf.nn.softmax(
                    self.decoder_hidden_list[i]) + 1e-8
                # tf.contrib.layers.summarize_activation(
                #     self.decoder_hidden_list[i])

                self.predictions_list[i] = {
                    "logits": self.decoder_hidden_list[i],
                    "probs": self.probs_list[i]
                }

                self.transition_probs = tf.nn.softmax(
                    self.decoder_transition_list[i]) + 1e-8
                # tf.contrib.layers.summarize_activation(
                #     self.decoder_transition_list[i])

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
                self.transition_loss_list[i] = tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.action_probs_list[i],
                        logits=self.decoder_transition_list[i]),
                    name="transition_loss_{}".format(i))
                self.final_loss_list[i] = self.loss_list[i] + \
                    self.lambda_tr * self.transition_loss_list[i]

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
                    self.instance_train_op = self.reverse_gradients()
                    self.transition_train_op = self.reverse_gradients_transition_classifier(
                    )
        # Merge summaries from this network and the shared network (but not the value net)
        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [
            s for s in summary_ops
            if "policy_net" in s.name or "shared" in s.name
        ]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)

    def reverse_gradients(self):
        instance_optimizer = tf.train.RMSPropOptimizer(self.learning_rate,
                                                       0.99, 0.0, 1e-6)
        grads_and_vars = self.optimizer.compute_gradients(self.instance_loss)
        grads, vars = list(zip(*grads_and_vars))
        # Clip gradients
        grads, _ = tf.clip_by_global_norm(grads, 5.0)

        self.grads_and_vars = []

        for i in range(len(vars)):
            target_name = vars[i].name
            if (grads[i] is not None):
                if ("classifier" in target_name):
                    self.grads_and_vars.append((grads[i], vars[i]))
                else:
                    self.grads_and_vars.append((tf.scalar_mul(
                        self.lambda_grad, tf.negative(grads[i])), vars[i]))
        return instance_optimizer.apply_gradients(
            self.grads_and_vars,
            global_step=tf.contrib.framework.get_global_step())

    def reverse_gradients_transition_classifier(self):
        transition_optimizer = tf.train.RMSPropOptimizer(
            self.learning_rate, 0.99, 0.0, 1e-6)
        grads_and_vars = self.optimizer.compute_gradients(
            self.transition_classification_loss)
        grads, vars = list(zip(*grads_and_vars))
        # Clip gradients
        grads, _ = tf.clip_by_global_norm(grads, 5.0)

        self.grads_and_vars_transition = []

        for i in range(len(vars)):
            target_name = vars[i].name
            if (grads[i] is not None):
                if ("classifier" in target_name):
                    print(("positive gradient in {}".format(vars[i])))
                    self.grads_and_vars_transition.append((grads[i], vars[i]))
                else:
                    print(("negative gradient in {}".format(vars[i])))
                    self.grads_and_vars_transition.append((tf.scalar_mul(
                        self.lambda_grad, tf.negative(grads[i])), vars[i]))
        return transition_optimizer.apply_gradients(
            self.grads_and_vars_transition,
            global_step=tf.contrib.framework.get_global_step())


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
                 num_attention_dim,
                 num_rl_hidden,
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
        self.input_size_list = [(num_inputs / self.fluent_feature_dims,
                                 self.feature_dims)
                                for num_inputs in self.num_inputs_list]
        self.num_gcn_hidden = num_gcn_hidden
        self.num_attention_dim = num_attention_dim
        self.num_rl_hidden = num_rl_hidden
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

        self.inputs = tf.sparse_placeholder(
            tf.float32, shape=[None, self.feature_dims], name="inputs")
        self.placeholders_hidden = {
            'support': [tf.sparse_placeholder(tf.float32)],
            'dropout': tf.placeholder_with_default(0., shape=()),
            # helper variable for sparse dropout
            'num_features_nonzero': tf.placeholder(tf.int32)
        }

        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        self.env_num = tf.placeholder(tf.int32, name="env_num")
        # Build network
        # TODO: add support
        with tf.variable_scope("value_net"):
            gconv1 = GraphConvolution(
                input_dim=self.feature_dims,
                output_dim=self.num_gcn_hidden,
                placeholders=self.placeholders_hidden,
                act=self.activation_fn,
                dropout=True,
                sparse_inputs=True,
                name='gconv1',
                logging=True)
            self.gcn_hidden = gconv1(self.inputs)

            self.attention_output = tf.layers.dense(
                inputs=self.gcn_hidden,
                units=self.num_attention_dim,
                use_bias=False,
                name="attention",
                reuse=tf.AUTO_REUSE)

            self.gcn_hidden_flat = tf.reshape(self.attention_output, [
                self.batch_size,
                tf.cast(
                    tf.gather(self.input_size_list, self.env_num)[0],
                    tf.int32), self.num_attention_dim
            ])
            self.gcn_max = tf.reduce_max(
                self.gcn_hidden_flat, axis=1, name="gcn_max")

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

                self.hidden_list[i] = tf.layers.dense(
                    inputs=self.gcn_max,
                    units=self.num_rl_hidden,
                    activation=self.activation_fn,
                    name="fcn_hidden_{}".format(i))

                self.logits_list[i] = tf.layers.dense(
                    inputs=self.hidden_list[i],
                    units=self.num_outputs,
                    activation=self.activation_fn,
                    name="output_{}".format(i))
                self.logits_list[i] = tf.squeeze(
                    self.logits_list[i],
                    squeeze_dims=[1],
                    name="logits_{}".format(i))

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
                tf.summary.scalar("{}/mean_value_{}".format(prefix, i),
                                  tf.reduce_mean(self.logits_list[i]))
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
