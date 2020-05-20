import os
import random
from collections import deque

import torch as T

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from QNet import DeepQCNN
from DuelingQNet import DualDeepQCNN
from util import plot_learning_curve
from memory import Memory, AgentMemory

from preprocessing_funcs import make_env

class Agent():
    '''Base class implementing functionality for different Deep Q Learning methods'''
    def __init__(self, env_name, input_dims, num_actions, learning_rate=2e-4, discount_factor=0.99,  eps=1.0, eps_decrement_factor=1e-5, eps_min=0.1, replay_memory_size=10000, mini_batch_size=32):
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.eps_min = eps_min
        self.eps = eps
        self.eps_decrement_factor = eps_decrement_factor
        self.mini_batch_size = mini_batch_size
        self.replay_memory_size = replay_memory_size
        self.memory_bank = AgentMemory(self.replay_memory_size)
        self.env_name = env_name

    def get_greedy_action(self, observation):
        raise NotImplementedError

    def save_models(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def save_models(self):
        self.online_network.save_checkpoint()
        self.target_network.save_checkpoint()

    def load_models(self):
        self.online_network.load_checkpoint()
        self.target_network.load_checkpoint()

    def store_memory(self, memory):
        self.memory_bank.remember(memory)

    def make_memory(self, state, action, reward, new_state, done):
        return np.array([state,
                         np.long(action),
                         float(reward),
                         new_state,
                         bool(done)])
    
    def get_random_action(self, observation):
        # randint return is inclusive of final value
        return random.randint(0, num_actions-1)

    def decrement_epsilon(self):
        new_eps = self.eps - self.eps_decrement_factor
        self.eps = new_eps if new_eps > self.eps_min else self.eps_min
    
    def sample_memory(self):
        self.memory_bank.recall_batch(
                self.mini_batch_size)

    def copy_online_nn_to_target_nn(self):
        self.target_network.load_state_dict(self.online_network.state_dict())


class DQNAgent(Agent):
    ''' Implementation of Deep Mind 2015, 'Human level control through deep reinforcement learning' with different parameters to reduce computation
    
    Implements Deep Q learning technique where a NN is used to obtain the state transition value associated with a given action taken in the initial state.

    Network is trained to predict the value(rewards) of the next state and disounted future states, by solving the bellman equation.  Immediate rewards are obtained from the environment, future rewards are disounted and added, and the NN is trained to minimize loss between true reward and current prediction.

    Experience replay is utilized to remove environment correlations to action and rewards obtained.  

    Seperate target and online networks are used to predict the state transition value, and the online network is trained to minimize error between targets output and actual future reward.  Online network weights are periodically copied to target network.  This improves stability of the target networks prediction, as the target is more stable. 
    '''

    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs) 
        
        self.online_network = DeepQCNN(
            self.input_dims, self.num_actions, name='OnlineNetwork')
        
        self.target_network = DeepQCNN(
            self.input_dims, self.num_actions, name='TargetNetwork')

    def get_greedy_action(self, observation):
        # convert obs to tensor, pass to device, forward pass, argmax
        obs_t = T.tensor(observation).to(
            self.online_network.device, dtype=T.float)
        action = self.target_network.forward(obs_t)

        return action.argmax().item()
    
    def learn(self):
        states, actions, rewards, states_, dones = self.sample_memory()

        inds = np.arange(self.mini_batch_size)

        # get the predicted action,state pair values
        q_pred = self.online_network.forward(T.tensor(states).to(
            self.online_network.device, dtype=T.float))[inds, actions]

        # get the next state values, to use in the recursive solution to bellman equation.  Select the maximally valued action for each action state pair in the batch  
        q_next = self.target_network.forward(
                T.tensor(states_).to(
                    self.target_network.device, dtype=T.float
                )
            ).max(dim=1)[0]
        
        #send rewards to gpu
        rewards = T.tensor(rewards).to(self.online_network.device)

        dones = np.asarray(dones, dtype=np.bool)
        
        # done without losing is a win, finish without losing a point, set the 0 points as reward
        q_next[dones] = 0.0
        
        #q' = rewards given plus disounted maximal reward from next action
        q_target = rewards + self.discount_factor*q_next

        loss = self.online_network.loss(
            q_pred, q_target).to(self.online_network.device)

        loss.backward()
        self.online_network.optimizer.step()
        self.online_network.optimizer.zero_grad()
        self.decrement_epsilon()

class DoubleDQNAgent(DQNAgent):
    '''
    Implements Double Deep Q Learning
    
    Double Deep Q is a modified version of Deep Q Learning utilizing different networks for action selection and action, state pair valuation. One network is used to get the predictions of the  state transition value, and used to select the maximal action of the next state.  This action is then used to select the value of the next state from the target network.   This is said to remove the bias for maximal action selection that would otherwise be observed by letting the same network both select and value the action.  
    '''
    def __init__(self, *args, **kwargs):
        super(DoubleDQNAgent, self).__init__(*args, **kwargs)
    
    def learn(self):
        
        states, actions, rewards, states_, dones = self.sample_memory

        inds = np.arange(self.mini_batch_size)

        #get predicted q value for state and actions
        q_pred = self.online_network.forward(T.tensor(states).to(
            self.online_network.device, dtype=T.float))[inds, actions]

        # use same online network to get the value of the next state
        q_eval = self.online_network.forward(
            T.tensor(states_).to(self.online_network.device, dtype=T.float))

        # get the maximal 
        max_actions = q_eval.max(1)[1]
        
        q_next = self.target_network.forward(
            T.tensor(states_).to(
                self.target_network.device, dtype=T.float)
            )[inds, max_actions]

        rewards = T.tensor(rewards).to(self.online_network.device)

        dones = np.asarray(dones, dtype=np.bool)

        q_next[dones] = 0.0
        
        q_target = rewards + self.discount_factor*q_next

        loss = self.online_network.loss(
            q_pred, q_target).to(self.online_network.device)

        loss.backward()
        self.online_network.optimizer.step()
        self.online_network.optimizer.zero_grad()
        self.decrement_epsilon()

class DuelingDDQN(self):
    '''
    Implementation of 'Dueling Network Architectures for Deep Reinforcement Learning'.  This implementation breaks up the value of the state transition/action pair into the value of the of the current state, and the marginal advantage of taking any action.  The marginal advantage is taken as the added value of the action over the average value of all the possible actions given some state.  This change is applied onto the DDQN framework by modifing the state transiton values to incorporate the averaging of the action values seen in next state.
    '''
    def __init__(self, *args, **kwargs):
        super(DuelingDDQN, self).__init__(*args, **kwargs)

        self.online_network = DualDeepQCNN(
            self.input_dims, self.num_actions, name='OnlineNetwork')

        self.target_network = DualDeepQCNN(
            self.input_dims, self.num_actions, name='TargetNetwork')
    
    def get_greedy_action(self, observation):
        # convert obs to tensor, pass to device, forward pass, argmax
        obs_t = T.tensor(observation).to(
            self.online_network.device, dtype=T.float)
        
        #current value of state, subtracting average value of best action does not matter as it only results in scaling of the actions without any change in ordering. 
        action_v, action_a = self.target_network.forward(obs_t)
        return action_a.argmax().item()
    
    def learn(self):
        states, actions, rewards, states_, dones = self.sample_memory()

        inds = np.arange(self.mini_batch_size)

        q_pred_v, q_pred_a = self.online_network.forward(T.tensor(states).to(self.online_network.device, dtype=T.float))
        
        q_pred = q_pred_v.reshape(self.mini_batch_size) + (q_pred_a[inds, actions] - q_pred_a.mean(1))

        q_eval_v, q_eval_a = self.online_network.forward(
            T.tensor(states_).to(self.online_network.device, dtype=T.float))

        q_eval = q_eval_v + (q_eval_a - q_eval_a.mean(1).reshape(32,1))

        max_actions = q_eval.max(1)[1]
        
        q_next_v, q_next_a = self.target_network.forward(
            T.tensor(states_).to(
                self.target_network.device, dtype=T.float)
            )

        q_next = q_next_v.reshape(self.mini_batch_size) + (q_next_a[inds, max_actions] - q_next_a.mean(1))
        
        rewards = T.tensor(rewards).to(self.online_network.device)

        dones = np.asarray(dones, dtype=np.bool)

        q_next[dones] = 0.0
        
        q_target = rewards + self.discount_factor*q_next

        loss = self.online_network.loss(
            q_pred, q_target).to(self.online_network.device)

        loss.backward()
        self.online_network.optimizer.step()
        self.online_network.optimizer.zero_grad()
        self.decrement_epsilon()



if __name__ == "__main__":
    '''Example usage'''

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

    agent = DuelingDDQN(obs.shape, num_actions,eps_decrement_factor=eps_dec_factor, eps_min=eps_min, replay_memory_size=replay_memory_size)

    scores = []
    eps_history = []
    step = 0

    for episode in tqdm(range(num_episodes)):
        
        # get the first state
        observation = env.reset()
        
        #add batch size dimension
        observation = observation.reshape(1, *observation.shape)

        score = 0
        done = False
        while not done:
            
            #choose an action based on epsilon value
            if random.random() < agent.eps:
                # explore
                action = agent.get_random_action(observation)
            else:
                # exploit
                action = agent.get_greedy_action(observation)

            # do the action and get the next state, reward for action, done signal, and env info
            obs_, reward, done, info = env.step(action)

            # add batch dim
            obs_ = obs_.reshape(1, *obs_.shape)

            #make and save state transition in experience replay memory
            memory = agent.make_memory(observation, action, reward, obs_, done)
            agent.store_memory(memory)

            # if we have enough state transtions in the experience replay, start training the NNs
            if episode > replay_start_size:
                
                agent.learn()
            
            observation = obs_
            score += reward
            step += 1

            # copy the trained weights to online weights every so often
            if step % 1000 == 0:
                print(step)
                agent.copy_online_nn_to_target_nn()
        
        ## save models if desired
        # if episode % 10:
        #    agent.save_models()

        eps_history.append(agent.eps)
        scores.append(score)
        print('episode ', episode, 'score', np.mean(
            scores[-10:]), 'eps', agent.eps)
