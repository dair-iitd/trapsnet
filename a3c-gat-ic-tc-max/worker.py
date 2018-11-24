import sys
import os
import copy
import itertools
import collections
import random
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf
# from python_toolbox import caching

# from lib import plotting
from estimators import ValueEstimator, PolicyEstimator
from parse_instance import InstanceParser
from gat.utils import process

Transition = collections.namedtuple(
    "Transition",
    ["instance", "state", "action", "reward", "next_state", "done"])


def make_copy_params_op(v1_list, v2_list):
    """
    Creates an operation that copies parameters from variable in v1_list to variables in v2_list.
    The ordering of the variables in the lists must be identical.
    """
    v1_list = list(sorted(v1_list, key=lambda v: v.name))
    v2_list = list(sorted(v2_list, key=lambda v: v.name))

    update_ops = []
    for v1, v2 in zip(v1_list, v2_list):
        op = v2.assign(v1)
        update_ops.append(op)

    return update_ops


def make_train_op(local_estimator, global_estimator, instance):
    """
    Creates an op that applies local estimator gradients
    to the global estimator.
    """
    local_grads, _ = list(zip(*local_estimator.grads_and_vars_list[instance]))
    # Clip gradients
    local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
    _, global_vars = list(zip(*global_estimator.grads_and_vars_list[instance]))
    local_global_grads_and_vars = list(zip(local_grads, global_vars))
    return global_estimator.optimizer.apply_gradients(
        local_global_grads_and_vars, global_step=tf.train.get_global_step())


class Worker(object):
    """
    An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.

    Args:
      name: A unique name for this worker
      env: The Gym environment used by this worker
      policy_net: Instance of the globally shared policy net
      value_net: Instance of the globally shared value net
      global_counter: Iterator that holds the global step
      discount_factor: Reward discount factor
      summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
      max_global_steps: If set, stop coordinator when global_counter > max_global_steps
    """

    def __init__(self,
                 name,
                 envs,
                 policy_net,
                 value_net,
                 global_counter,
                 domain,
                 instances,
                 neighbourhood,
                 discount_factor=0.99,
                 summary_writer=None,
                 max_global_steps=None):
        self.name = name
        self.domain = domain
        self.instances = instances
        self.dropout = 0.0
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.global_step = tf.train.get_global_step()
        self.global_policy_net = policy_net
        self.global_value_net = value_net
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.summary_writer = summary_writer
        self.envs = envs

        self.N = len(instances)
        self.current_instance = 0

        assert (self.N == len(self.envs))
        self.num_inputs_list = policy_net.num_inputs_list

        # Construct adjacency lists
        self.instance_parser_list = [None] * self.N
        self.adjacency_lists = [None] * self.N
        self.nf_features = [None] * self.N
        self.adjacency_lists_with_biases = [None] * self.N

        for i in range(self.N):
            self.instance_parser_list[i] = InstanceParser(
                self.domain, self.instances[i])
            self.fluent_feature_dims, self.nonfluent_feature_dims = self.instance_parser_list[
                i].get_feature_dims()
            self.nf_features[i] = self.instance_parser_list[
                i].get_nf_features()
            adjacency_list = self.instance_parser_list[i].get_adjacency_list()
            self.adjacency_lists[i] = nx.adjacency_matrix(
                nx.from_dict_of_lists(adjacency_list))
            self.adjacency_lists[i] = self.adjacency_lists[i].todense()
            self.adjacency_lists_with_biases[i] = process.adj_to_bias(
                np.array([self.adjacency_lists[i]]), [self.num_inputs_list[i]],
                nhood=neighbourhood)[0]

        # Create local policy/value nets that are not updated asynchronously
        with tf.variable_scope(name):
            self.policy_net = PolicyEstimator(
                policy_net.num_inputs_list, self.N, policy_net.num_gcn_hidden,
                policy_net.num_attention_dim, policy_net.num_rl_hidden,
                policy_net.num_hidden_transition, policy_net.num_action_embed,
                policy_net.num_outputs_list, policy_net.fluent_feature_dims,
                policy_net.nonfluent_feature_dims, policy_net.activation,
                policy_net.learning_rate)
            self.value_net = ValueEstimator(
                value_net.num_inputs_list, self.N, value_net.num_gcn_hidden,
                value_net.num_attention_dim, value_net.num_rl_hidden,
                value_net.fluent_feature_dims,
                value_net.nonfluent_feature_dims, value_net.activation,
                value_net.learning_rate)

        # Op to copy params from global policy/valuenets
        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(
                scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(
                scope=self.name, collection=tf.GraphKeys.TRAINABLE_VARIABLES))

        self.vnet_train_op_list = [None] * self.N
        self.pnet_train_op_list = [None] * self.N

        for i in range(self.N):
            self.vnet_train_op_list[i] = make_train_op(
                self.value_net, self.global_value_net, i)
            self.pnet_train_op_list[i] = make_train_op(
                self.policy_net, self.global_policy_net, i)

        self.state = None

    def get_processed_adj_biases(self, i, batch_size):
        adj_list_biases = []
        for _ in range(batch_size):
            adj_list_biases.append(self.adjacency_lists_with_biases[i])

        return np.array(adj_list_biases)

    def get_processed_input(self, states, i):
        def state2feature(state):
            feature_arr = np.array(state).astype(np.float32).reshape(
                self.instance_parser_list[i].input_size)
            if self.nf_features[i] is not None:
                feature_arr = np.hstack((feature_arr, self.nf_features[i]))
            return feature_arr

        features = np.array(list(map(state2feature, states)))
        return features

    def run(self, sess, coord, t_max):
        with sess.as_default(), sess.graph.as_default():
            # Initial state
            self.state = np.array(
                self.envs[self.current_instance].initial_state)
            try:
                while not coord.should_stop():
                    # Copy Parameters from the global networks
                    sess.run(self.copy_params_op)

                    # Collect some experience
                    transitions, local_t, global_t = self.run_n_steps(
                        t_max, sess)

                    if self.max_global_steps is not None and global_t >= self.max_global_steps:
                        tf.logging.info(
                            "Reached global step {}. Stopping.".format(
                                global_t))
                        coord.request_stop()
                        return

                    # Update the global networks
                    self.update(transitions, sess)

            except tf.errors.CancelledError:
                return

    def _policy_net_predict(self, state, input_features_preprocessed,
                            adj_preprocessed, instance, sess):
        feed_dict = {
            self.policy_net.inputs: np.array([input_features_preprocessed]),
            self.policy_net.adj_biases_placeholder:
            np.array([adj_preprocessed]),
            self.policy_net.is_train: False,
            self.policy_net.states_list[instance]: np.array([state]),
            self.policy_net.batch_size: 1,
            self.policy_net.env_num: instance
        }
        preds = sess.run(self.policy_net.predictions_list[instance], feed_dict)
        return preds["probs"][0]

    def _value_net_predict(self, input_features_preprocessed, adj_preprocessed,
                           instance, sess):
        feed_dict = {
            self.value_net.inputs: np.array([input_features_preprocessed]),
            self.value_net.adj_biases_placeholder:
            np.array([adj_preprocessed]),
            self.policy_net.is_train: False,
            self.value_net.batch_size: 1,
            self.value_net.env_num: instance
        }
        preds = sess.run(self.value_net.predictions_list[instance], feed_dict)
        return preds["logits"][0]

    def run_n_steps(self, n, sess):
        transitions = []
        for _ in range(n):
            # Take a step
            input_features_preprocessed = self.get_processed_input(
                [self.state], self.current_instance)
            action_probs = self._policy_net_predict(
                self.state, input_features_preprocessed[0],
                self.adjacency_lists_with_biases[self.current_instance],
                self.current_instance, sess)
            action = np.random.choice(
                np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = self.envs[
                self.current_instance].step(action)

            # Store transition
            transitions.append(
                Transition(
                    instance=self.current_instance,
                    state=self.state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done))

            # Increase local and global counters
            local_t = next(self.local_counter)
            global_t = next(self.global_counter)

            if local_t % 100 == 0:
                tf.logging.info("{}: local Step {}, global step {}".format(
                    self.name, local_t, global_t))

            if done:
                # reset state and TODO: check if reset end-of-episode flag
                # self.current_instance = random.choice(range(self.N))    # Randomly choose next instance to train
                # Choose next instance
                self.current_instance = (self.current_instance + 1) % self.N
                if random.uniform(0, 1) < 0.5:
                    initial_state, done = self.envs[
                        self.current_instance].reset()
                else:
                    initial_state, done = self.envs[
                        self.current_instance].random_reset()
                self.state = initial_state
                break
            else:
                self.state = next_state

        return transitions, local_t, global_t

    def update(self, transitions, sess):
        """
        Updates global policy and value networks based on collected experience

        Args:
          transitions: A list of experience transitions
          sess: A Tensorflow session
        """

        instance = transitions[0].instance
        # If we episode was not done we bootstrap the value from the last state
        reward = 0.0
        if not transitions[-1].done:
            input_features_preprocessed = self.get_processed_input(
                [transitions[-1].next_state], instance)
            reward = self._value_net_predict(
                input_features_preprocessed[0],
                self.adjacency_lists_with_biases[instance], instance, sess)

        # Accumulate minibatch exmaples
        states = []
        policy_targets = []
        value_targets = []
        actions = []
        action_probs = []

        l = len(transitions)
        instance_target = []

        transitions_reverse = transitions[::-1]
        for i, transition in enumerate(transitions_reverse):
            reward = transition.reward + self.discount_factor * reward
            input_features_preprocessed = self.get_processed_input(
                [transition.state], instance)
            policy_target = (reward - self._value_net_predict(
                input_features_preprocessed[0],
                self.adjacency_lists_with_biases[instance], instance, sess))
            if i < l - 1:
                # get curr and next states - note that transitions is reversed
                next_state = transitions_reverse[i].state
                curr_state = transitions_reverse[i + 1].state
                action_prob = self.instance_parser_list[
                    instance].get_action_probs(curr_state, next_state)
                action_probs.append(action_prob)

            # Accumulate updates
            states.append(transition.state)
            actions.append(transition.action)
            policy_targets.append(policy_target)
            value_targets.append(reward)
            instance_target.append(instance)

        if len(action_probs) > 0:
            batch_size = len(states)
            adj_biases_preprocessed = self.get_processed_adj_biases(
                instance, batch_size)
            input_features_preprocessed = self.get_processed_input(
                states, instance)

            feed_dict = {
                self.policy_net.inputs: input_features_preprocessed,
                self.policy_net.adj_biases_placeholder:
                adj_biases_preprocessed,
                self.policy_net.is_train: True,
                self.policy_net.targets: policy_targets,
                self.policy_net.actions: actions,
                self.policy_net.states_list[instance]: states,
                self.policy_net.action_probs_list[instance]: action_probs,
                self.policy_net.instance: instance_target,
                self.policy_net.env_num: instance,
                self.policy_net.batch_size: batch_size,
                self.value_net.inputs: input_features_preprocessed,
                self.value_net.adj_biases_placeholder: adj_biases_preprocessed,
                self.value_net.is_train: True,
                self.value_net.targets: value_targets,
                self.value_net.batch_size: batch_size,
                self.value_net.env_num: instance
            }

            # Train the global estimators using local gradients
            global_step, pnet_loss, vnet_loss, _, _, pnet_summaries, vnet_summaries, _, _ = sess.run(
                [
                    self.global_step, self.policy_net.loss_list[instance],
                    self.value_net.loss_list[instance],
                    self.pnet_train_op_list[instance],
                    self.vnet_train_op_list[instance],
                    self.policy_net.summaries, self.value_net.summaries,
                    self.policy_net.instance_train_op,
                    self.policy_net.transition_train_op
                ], feed_dict)

            # Write summaries
            if self.summary_writer is not None:
                self.summary_writer.add_summary(pnet_summaries, global_step)
                self.summary_writer.add_summary(vnet_summaries, global_step)
                self.summary_writer.flush()

            return pnet_loss, vnet_loss, pnet_summaries, vnet_summaries
