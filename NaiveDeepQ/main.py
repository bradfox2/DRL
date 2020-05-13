''' trains a naive deep q network on frozenlake'''

from network import LinearClassifier
from QAgent import Agent

from collections import defaultdict
from random import random
from QAgent import Agent
import random
import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make('CartPole-v1')

scores = []
winpct = []

n_episodes = 10000

agent = Agent(env.observation_space.n, env.action_space.n, epsilon = 1.0, discount_factor = 0.99)

eps_dec = .99995

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
        if ep_num % 100 == 0:
            print('episode ', ep_num, 'win pct %.2f' % average, 'eps %.2f' % agent.epsilon)

plt.plot(winpct)
plt.show()
