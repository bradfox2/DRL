'''implements agent class for q learning'''
import random
from collections import defaultdict
from network import LinearClassifier
import numpy as np

class Agent():
    def __init__(self,n_states, n_actions, discount_factor=.9, epsilon_max=1.0, epsilon_min = 0.01, epsilon=.1, learning_rate = 0.001):
        self.discount_factor = discount_factor
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.q_model = LinearClassifier(learning_rate, n_actions, (n_states,))
    
    def get_max_q(self, state):
        return max(self.get_state_act_pair_value(state)).item()

    def choose_greedy_action(self, state):
        _state = np.zeros(self.n_states, dtype=np.float32)
        _state[state] = 1
        return np.argmax(self.q_model.forward(self.q_model.to_tensor(_state)).cpu().detach().numpy())
        
    def choose_random_action(self,state):
        return random.randint(0, self.n_actions-1)
    
    def get_state_act_pair_value(self, state):
        _state = np.zeros(self.n_states, dtype=np.float32)
        _state[state] = 1
        data_tensor = self.q_model.to_tensor(_state)
        action_probs = self.q_model.forward(data_tensor)
        return action_probs
        
    def update_q_model(self, state, target_q):
        _state = np.zeros(self.n_states, dtype=np.float32)
        _state[state] = 1
        self.q_model.learn(_state, target_q)

    def update_state(self, state, action, new_state, reward):
        #q(s,a) = q(s,a) + alpha(r + gamma*max(Q(s',a_max)- q(s,a)))
        q = self.get_state_act_pair_value(state)
        alpha = self.learning_rate
        gamma = self.discount_factor
        
        max_q = self.get_max_q(new_state)
        
        for i in range(len(q)):
            q[i] = q[i] + (reward + gamma * max_q - q[i])
        
        self.update_q_model(state, q)
    
    def decrement_epsilon(self, amount):
        neweps = self.epsilon * amount
        if neweps > self.epsilon_min and neweps < self.epsilon_max:
            self.epsilon = neweps
    
