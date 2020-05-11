''' Beats the OpenAI Gym Frozen Lake environment by using Q learning and a simple state, action table'''

import heapq
from collections import defaultdict
from random import random

import gym

env = gym.make('FrozenLake-v0')

env.reset()

state_rewards = defaultdict(dict)
obs_actions = defaultdict(dict)
obs = None

for _ in range(10000):
    #greedy strategy if below eps else random
    env.reset()
    eps = random()
    done = None
    while not done:
        
        # randomly sample moves 10% of time
        if eps > .9 or obs is None:
            action = env.action_space.sample() 
        else:
            # take the learned greedy move
            action = sorted(state_rewards[obs].items(), key = lambda x: x[1], reverse = True)[0][0]
        
        obs, reward, done, info = env.step(action)
                
        if obs == 15:
            # check if we reached termination (success!)
            print(f'Reached goal in {_} intervals.')
            print(state_rewards)
            print(env.render())
            exit()

        #state:action
        if not done:
            # we moved successfully without dying, + 1 point
            new_reward = state_rewards.get(obs,{}).get(action,0) + 1
            #action:reward
            state_rewards[obs][action] = new_reward
        
        else:
            # if we're done without being in space 15, we fell in a hole in died, -1 point
            new_reward = state_rewards.get(obs,{}).get(action,0) - 1
            state_rewards[obs][action] = new_reward
            