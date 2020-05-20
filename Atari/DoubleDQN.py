import os
import random
from collections import deque

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from QNet import DeepQCNN
from util import plot_learning_curve
from memory import Memory, AgentMemory

from preprocessing_funcs import make_env

# 2 networks, one target, one online, update online network with batches, transfer weights periodically to the target network. target network does not get loss backpropped.  use replay memory to sample training data for the online network

# functions to choose actions, copy target  network weights, dec eps, learn and store new memories

import torch as T


class Agent():
    def __init__(self, input_dims, num_actions, learning_rate=2e-4, discount_factor=0.99,  eps=1.0, eps_decrement_factor=1e-5, eps_min=0.1, replay_memory_size=10000, mini_batch_size=32):
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.eps_min = eps_min
        self.eps = eps
        self.eps_decrement_factor = eps_decrement_factor
        self.mini_batch_size = mini_batch_size
        #self.Q = LinearDQN(learning_rate, num_actions, input_dims)
        self.online_network = DeepQCNN(
            input_dims, self.num_actions, name='OnlineNetwork')
        self.target_network = DeepQCNN(
            input_dims, self.num_actions, name='TargetNetwork')
        self.replay_memory_size = replay_memory_size
        self.memory_bank = AgentMemory(self.replay_memory_size)

    def decrement_epsilon(self):
        new_eps = self.eps - self.eps_decrement_factor
        self.eps = new_eps if new_eps > self.eps_min else self.eps_min

    def store_memory(self, memory):
        self.memory_bank.remember(memory)

    def make_memory(self, state, action, reward, new_state, done):
        return np.array([state,
                         np.long(action),
                         float(reward),
                         new_state,
                         bool(done)])

    def get_greedy_action(self, observation):
        # convert obs to tensor, pass to device, forward pass, argmax
        obs_t = T.tensor(observation).to(
            self.online_network.device, dtype=T.float)
        action = self.target_network.forward(obs_t)

        return action.argmax().item()

    def get_random_action(self, observation):
        # randint return is inclusive of final value
        return random.randint(0, num_actions-1)

    def train_online_network(self):
        pass

    def save_models(self):
        self.online_network.save_checkpoint()
        self.target_network.save_checkpoint()

    def load_models(self):
        self.online_network.load_checkpoint()
        self.target_network.load_checkpoint()

        #replay_memory_training_data = self.memory_bank.recall_batch(mini_batch_size)
        # need is an array of arrays outer array (batchsize, 2), inner array(training data, targets)
        # self.online_network.fit()

    def update_target_network(self):
        pass

    def copy_online_nn_to_target_nn(self):
        self.target_network.load_state_dict(self.online_network.state_dict())


env = make_env('PongNoFrameskip-v4')
obs = env.reset()
eps_min = .1
eps_max = 1.0
num_episodes = 500
final_exploration_frame = 50
eps_dec_factor = 1e-4
replay_start_size = 2 if 2 <= num_episodes else num_episodes
num_actions = env.action_space.n
target_network_update_frequency = 1
replay_memory_size = 100000

agent = Agent(obs.shape, num_actions, eps_decrement_factor=eps_dec_factor,
              eps_min=eps_min, replay_memory_size=replay_memory_size)

scores = []
eps_history = []
step = 0

for episode in tqdm(range(num_episodes)):

    observation = env.reset()
    observation = observation.reshape(1, *observation.shape)
    score = 0
    done = False

    while not done:
        # reshape to batch size 1

        if random.random() < agent.eps:
            # explore
            action = agent.get_random_action(observation)
        else:
            # exploit
            action = agent.get_greedy_action(observation)

        obs_, reward, done, info = env.step(action)
        obs_ = obs_.reshape(1, *obs_.shape)

        memory = agent.make_memory(observation, action, reward, obs_, done)
        agent.store_memory(memory)

        if episode > replay_start_size:

            states, actions, rewards, states_, dones = agent.memory_bank.recall_batch(
                32)

            inds = np.arange(agent.mini_batch_size)

            q_pred = agent.online_network.forward(T.tensor(states).to(
                agent.online_network.device, dtype=T.float))[inds, actions]

            q_eval = agent.online_network.forward(
                T.tensor(states_).to(agent.online_network.device, dtype=T.float))

            max_actions = q_eval.max(1)[1]
            
            q_next = agent.target_network.forward(
                T.tensor(states_).to(
                    agent.target_network.device, dtype=T.float)
                )[inds, max_actions]

            rewards = T.tensor(rewards).to(agent.online_network.device)

            dones = np.asarray(dones, dtype=np.bool)

            q_next[dones] = 0.0
            
            q_target = rewards + agent.discount_factor*q_next

            loss = agent.online_network.loss(
                q_pred, q_target).to(agent.online_network.device)

            loss.backward()
            agent.online_network.optimizer.step()
            agent.online_network.optimizer.zero_grad()
            agent.decrement_epsilon()

        observation = obs_
        score += reward
        step += 1

        if step % 1000 == 0:
            print(step)
            agent.copy_online_nn_to_target_nn()

    # if episode % 10:
    #    agent.save_models()

    eps_history.append(agent.eps)
    scores.append(score)
    print('episode ', episode, 'score', np.mean(
        scores[-10:]), 'eps', agent.eps)
