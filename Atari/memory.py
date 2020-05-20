'''classes implementing replay memory for q agent'''

from collections import deque
import numpy as np
import random

class Memory():
    def __init__(self, state, action, reward, new_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.done = done

class AgentMemory():
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)
    
    def remember(self, memory):
        self.memory.append(memory)

    def recall(self):
        '''samples a memory uniformly from all memories'''
        return np.random.choice(self.memory, replace = False)

    def recall_batch(self, quantity):
        '''samples a batch of memories, returns arrays of states, actions, rewards, next_states, and dones'''
        memory_batch = np.stack(random.sample(self.memory, quantity))
    
        return np.concatenate(memory_batch[:,0]), \
        memory_batch[:,1].astype(np.long), \
        memory_batch[:,2].astype(np.long), \
        np.concatenate(memory_batch[:,3]), \
        memory_batch[:,4].astype(np.bool) 
    
    def recall_batch_numpy(self, quantity):
        batch = np.zeros([32,2])
        batch = self.recall_batch(quantity)
        
        states = [mem.state for mem in batch]
        rewards = [mem.reward for mem in batch]
        return np.asarray(list(zip(states,rewards)))
    