import gym
from collections import deque
import numpy as np
import cv2

env = gym.make('PongNoFrameskip-v4')
env.reset()

class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env, repeat_n_frames=4, clip_reward=False, no_ops=0, fire_first=False):
        ''' 
        repeat_n_frames: apply action to this number of frames
        env: open ai gym env object 
        '''
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat_n_frames = repeat_n_frames
        self.frame_buffer = self.make_frame_buffer()
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        '''overload of env.step to handle finding the maximal frame to remove emulator artifacts, and applying the same action on multiple frames(reduce computational complexity)

        returns same data structure as gym.step'''
        total_reward = 0.0
        done = False
        # repeat actions on subsequent frames
        for i in range(self.repeat_n_frames):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(reward)
            total_reward += reward
            self.frame_buffer.append(obs)
            if done:
                break

        max_frame = self.find_max_frame()
        return max_frame, total_reward, done, info

    def find_max_frame(self):
        '''finds the maximal value for each element in frame buffer arrays, returns single array of maximal values'''
        return np.maximum(*self.frame_buffer)

    def reset(self):
        ''' Resets environment, resets framebuffer, and inserts first frame into frame buffer, returns initial_frame passed from env'''
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)
        self.reset_frame_buffer()
        initial_frame = self.env.reset()
        self.frame_buffer.append(initial_frame)
        return initial_frame

    def make_frame_buffer(self):
        ''' returns empty frame buffer data structure'''
        return deque(maxlen=2)

    def reset_frame_buffer(self):
        ''' resets frame buffer to initial data structure'''
        self.frame_buffer = self.make_frame_buffer()


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super(PreprocessFrame, self).__init__(env)
        # num_channels, width, height
        self.shape = (shape[2], shape[0], shape[1])
        self.last_frame = None
        self.converted_frame = None
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=self.shape)

    def show_frame(self):
        ''' display the last frame in original resoultion'''
        cv2.imshow('Frame', self.last_frame)
        cv2.waitKey(0)

    def show_converted_frame(self):
        ''' display the last frame in original resoultion'''
        cv2.imshow('Frame', self.converted_frame)
        cv2.waitKey(0)

    def observation(self, observation):
        ''' resize, normalize and move channel dim to 0th position'''
        self.last_frame = observation
        # move channel to first dim
        gray_scale_observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(
            gray_scale_observation, self.shape[1:], interpolation=cv2.INTER_AREA)
        self.converted_frame = observation
        observation = observation.reshape(self.shape)
        # normalize
        observation = observation/255.0

        return observation

# ppf = PreprocessFrame(env, (84, 84, 1))
# observation,_,_,_ = env.step(1)
# obs = ppf.observation(observation)
# ppf.show_frame()
# ppf.show_converted_frame()


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat_frames):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat_frames, axis=0),
            env.observation_space.high.repeat(repeat_frames, axis=0),
            dtype=np.float32)

        self.stack = deque(maxlen=repeat_frames)
        self.repeat_frame_num = repeat_frames

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.repeat_frame_num):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.low.shape)

# s = StackFrames(ppf, 4)
# obs = s.reset()
# s.stack[0]
# s.reset().shape
# s.observation_space


def make_env(env_name, shape=(84,84,1), repeat_frames=4, clip_rewards=False, no_ops=0, fire_first=False):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat_frames)
    env = PreprocessFrame(env, shape)
    env = StackFrames(env, repeat_frames)

    return env

# learn s,a,r,s', dones
# take arbitrary state shapes
# uniform sampling of memory
# distinct memories only

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
        if memory not in self.memory:
            self.memory.append(memory)

    def recall(self):
        return np.random.choice(self.memory, replace=False)
