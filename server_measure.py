#!/usr/bin/python

import sys
import socket
import argparse
import json

import gym

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, help='server host', default='localhost')
parser.add_argument('--port', type=int, help='server port', default=9000)
parser.add_argument('--env', type=str, help='environment')
parser.add_argument('--num_episodes', type=int, help='number of episodes', default=100)
parser.add_argument('--outfile', type=str, help='file to store rewards')
args = parser.parse_args()


class RDDLMeasureServer():
    def __init__(self, host, port, env):
        self.serverhost = host
        self.serverport = port
        self.init_server()
        self.init_env(env)

    def init_server(self):
        # create a socket object
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # start server
        self.serversocket.bind((self.serverhost, self.serverport))
        print(('Starting server at {}:{}'.format(self.serverhost, self.serverport)))
        MAX_REQUESTS = 1
        self.serversocket.listen(MAX_REQUESTS)

    def init_env(self, env):
        print(('Environment', env))
        self.env = gym.make('RDDL-' + env)
        self.env.seed(0)

    def init_episode(self):
        # Initialize env
        self.state, self.done = self.env.reset()
        self.immediate_reward = 0
        self.reward = 0

    def connect_to_client(self):
        # establish a connection
        self.clientsocket, self.client_address = self.serversocket.accept()
        print(('Got a connection from {}'.format(self.client_address)))

    def simulate_episode(self):
        self.init_episode()

        MAX_BYTES = 1024
        while not(self.done):

            # send data to client
            send_data = json.dumps({'state': self.state.tolist(), 'reward': self.immediate_reward, 'done': self.done})
            self.clientsocket.sendall(send_data)

            # receive data from client
            recv_data = self.clientsocket.recv(MAX_BYTES)
            recv_dict = json.loads(recv_data)
            action = recv_dict['action']

            # take action received from client
            next_state, self.immediate_reward, self.done, _ = self.env.step(action)
            # print('state: {}  action: {}  reward: {} next: {}'.format(
            #     self.state, action, self.immediate_reward, next_state))
            if not self.done:
                self.state = next_state
                self.reward += self.immediate_reward

    def end_connection(self):
        self.env.close()
        self.clientsocket.close()


def main():

    # Initialize server
    server = RDDLMeasureServer(args.host, args.port, args.env)
    server.connect_to_client()

    # Simulate episodes
    rewards = []
    for i in range(args.num_episodes):
        server.simulate_episode()
        print(('Done episode', i))
        print(('Reward: ', server.reward))
        rewards.append(server.reward)

    avg_reward = sum(rewards) / float(len(rewards))
    print("Avg reward:", avg_reward)

    with open(args.outfile, 'w') as f:
        rewards_str = [str(x) + '\n' for x in rewards]
        rewards_str += str(avg_reward)
        f.writelines(rewards_str)

    # end connection
    server.end_connection()


if __name__ == '__main__':
    main()
