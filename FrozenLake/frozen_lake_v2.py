''' Beats the OpenAI Gym Frozen Lake environment by learning a policy and a simple state, action table, keeping track of paths, and using Gym's reward'''

from random import random

import gym

env = gym.make('FrozenLake-v0')

env.reset()

node_paths = {}

paths = []

def take_action(obs, state_rewards, q_table):
    if random() > .9 or q_table == {}:
        action = env.action_space.sample()
    else:
        #take greedy actions
        action = sorted(q_table[obs].items(), key = lambda x: x[1], reverse = True)[0][0]
    
    obs, reward, done, info = env.step(action)
    
    if done:
        return [obs, action, reward]
    
    state_rewards.append(take_action(obs, state_rewards, q_table))
    return state_rewards

take_action(0, [], q_table = {})

state_rewards = {}
for i in range(1000):
    env.reset()
    play_instance = check_paths(0)
    if play_instance[-1][-1] == 1:
        new_reward = None
        for action_state_pair in reversed(play_instance):
            reward = action_state_pair[-1]
            reward = new_reward if new_reward else reward
            new_reward = reward * .95            
            current_cum_reward = state_rewards.get(action_state_pair, 0)
            state_rewards[action_state_pair] = current_cum_reward + new_reward

