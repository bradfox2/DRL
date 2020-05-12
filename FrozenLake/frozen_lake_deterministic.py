''' Human programmed paths in Frozen Lake.

https://gym.openai.com/envs/FrozenLake-v0/

'''

import heapq
from collections import defaultdict
import random

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
scores = []
winpct = []
# SFFF       (S: starting point, safe)
# FHFH       (F: frozen surface, safe)
# FFFH       (H: hole, fall to your doom)
# HFFG       (G: goal, where the frisbee is located)

winning_routes = [[2,2,1,1,1,2],[1,1,2,1,2,2]]
policy = {0:2, 1:2, 2:1, 3:0, 4:1, 5:2, 6:1, 7:1, 8:2, 9:2, 10:1, 11:1, 12:2, 13:2, 14:2, 15:1}

for _ in range(1000):
    obs = env.reset()
    done = False
    score = 0 
    #route = random.choice(winning_routes)       
    while not done:
        #for action in route:
         #   obs, reward, done, info = env.step(action)
        obs, reward, done, info = env.step(policy[obs])
    score += reward
    scores.append(score)

    if _ % 10 == 0:
        average = np.mean(scores[-10:])
        winpct.append(average)

plt.plot(winpct)
plt.show()