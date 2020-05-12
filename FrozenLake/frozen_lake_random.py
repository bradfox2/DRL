''' Randomly sampled states in frozen lake

https://gym.openai.com/envs/FrozenLake-v0/

'''

import heapq
from collections import defaultdict
from random import random

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
scores = []
winpct = []

for _ in range(1000):
    env.reset()
    done = None
    score = 0 

    while not done:
        action = env.action_space.sample() 
        obs, reward, done, info = env.step(action)
        score += reward
    scores.append(score)

    if _ % 10 == 0:
        average = np.mean(scores[-10:])
        winpct.append(average)

plt.plot(winpct)
plt.show()