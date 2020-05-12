''' Beats the OpenAI Gym Frozen Lake environment by using Q learning and a simple state, action table.  Gets to about 80% wins in 100k episodes.

https://gym.openai.com/envs/FrozenLake-v0/

'''

from collections import defaultdict
from random import random
from QAgent import Agent
import random
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make('FrozenLake-v0')

scores = []
winpct = []

n_episodes = 100000

agent = Agent(env.observation_space.n, env.action_space.n, epsilon = 1.0)
eps_dec = .9995

for ep_num in tqdm(range(n_episodes)):
    s = env.reset()
    done = False
    score = 0
    while not done:
        if random.random() < agent.epsilon:
            action = agent.choose_random_action(state=s)
        else:        
            action = agent.choose_greedy_action(s)
        s_prime, reward, done, info = env.step(action)
        agent.update_state(s, action, s_prime, reward)
        agent.decrement_epsilon(eps_dec)
        s = s_prime

        score += reward
    scores.append(score)

    if ep_num % 100 == 0:
        average = np.mean(scores[-100:])
        winpct.append(average)
        if ep_num % 1000 == 0:
            print('episode ', ep_num, 'win pct %.2f' % winpct, 'eps %.2f' % agent.epsilon)


plt.plot(winpct)
plt.show()
