import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
print(os.getcwd())
from util import plot_learning_curve

class LinearDQN(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDQN, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)
        return actions
    
    def backprop_loss(self, preds, targets):
        loss = self.loss(preds, targets).to(self.device)
        loss.backward()
        self.optimizer.step()
        self.zero_grad()

    def to_device_tensor(self, val):
        return T.tensor(val, dtype=T.float).to(self.device)

class Agent(object):
    def __init__(self, input_dims, num_actions, learning_rate = 1e-4, discount_factor=0.99,  eps=1.0, eps_decrement_factor=1e-5, eps_min=0.01):
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.eps_min = eps_min
        self.eps = eps
        self.eps_decrement_factor = eps_decrement_factor
        self.Q = LinearDQN(learning_rate, num_actions, input_dims)

    def choose_greedy_action(self, observation) -> int:
        state = T.tensor(observation, dtype=T.float).to(self.Q.device)
        return self.Q.forward(state).argmax().item()

    def choose_action(self, observation) -> int:
        if np.random.random() >  self.eps:
            action = self.choose_greedy_action(observation)
            return action
        else:
            return random.choice(range(self.num_actions))
    
    def decrement_eps(self):
        new_eps = self.eps - self.eps_decrement_factor
        self.eps = new_eps if new_eps > self.eps_min else self.eps_min
    
    def get_action_state_pair_value(self,state):
        stateT = self.Q.to_device_tensor(state)
        return self.Q.forward(stateT)

    def get_q_pred(self, state):
        return self.get_action_state_pair_value(state)

    def learn(self, state, action, reward, state_):
        gamma = self.discount_factor
        q_pred = self.get_q_pred(state)[action]
        q_prime = self.get_q_pred(state_).max()
        q_target = reward + gamma * q_prime
        self.Q.backprop_loss(q_pred, q_target)
        self.decrement_eps()

if __name__ == "__main__":

    env = gym.make('CartPole-v1')

    scores = []
    winpct = []
    eps_history = []
    n_episodes = 10000

    agent = Agent(env.observation_space.shape, env.action_space.n)

    n_plays = 5000

    for i in tqdm(range(n_plays)):
        observation = env.reset()
        done = None
        score = 0
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            agent.learn(observation, action, reward, next_observation)
            observation = next_observation
        scores.append(score)
        eps_history.append(agent.eps)

        if i % 100==0:
            avg_score = np.mean(scores[-100:])
            print('episode', i , 'score %.1f avg score %.1f epsilon %.2f' % (score, avg_score, agent.eps))

    filename = 'cartpole_naive_dqn.png'
    plot_learning_curve(range(n_plays), scores, eps_history, filename)
    




