'''implements agent class for q learning'''
import random
from collections import defaultdict

class Agent():
    def __init__(self,n_states, n_actions, learning_rate=.01, discount_factor=.9, epsilon_max=1.0, epsilon_min = 0.01, epsilon=.1):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = self._init_q_table() # nested dicts, states:actions:values
    
    def get_max_q(self, state):
        q_vals = sorted(self.q_table[state].items(), key = lambda kv: kv[1], reverse = True)
        return q_vals[0][1]

    def choose_greedy_action(self, state):
        action = self.q_table[state]
        # take first action, sorted by value 
        return sorted(action.items(), key = lambda kv: kv[1], reverse = True)[0][0]
        
    def choose_random_action(self,state):
        actions = self.q_table.get(state, {})
        return random.choice(list(actions.keys()))

    def _init_q_table(self):
        return {k:{i:0 for i in range(self.n_actions)} for k in range(self.n_states)}

    def add_to_q_table(self, observation, action, value):
        self.q_table[observation][action] += value 
    
    def get_state_act_pair_value(self, state, action):
        return self.q_table[state][action]
    
    def update_state(self, state, action, new_state, reward):
        #q(s,a) = q(s,a) + alpha(r + gamma*max(Q(s',a_max)- q(s,a)))
        q = self.get_state_act_pair_value(state, action)
        alpha = self.learning_rate
        gamma = self.discount_factor
        q_delta = alpha * (reward + gamma * self.get_max_q(new_state) - q)
        self.add_to_q_table(state, action, q_delta)
    
    def decrement_epsilon(self, amount):
        neweps = self.epsilon * amount
        if neweps > self.epsilon_min and neweps < self.epsilon_max:
            self.epsilon = neweps
