import gym
import numpy as np

from baselines.common.atari_wrappers import FrameStack

class ActionsDiscretizer(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionsDiscretizer, self).__init__(env)

        self._actions = np.array([
           [0, 0, 0, 0, 0, 0], #0 - no button",
           [1, 0, 0, 0, 0, 0], #1 - up only (to climb vine)",
           [0, 0, 1, 0, 0, 0], #2 - left only",
           [0, 0, 0, 1, 0, 0], #3 - right only",
           [0, 0, 0, 0, 0, 1], #4 - run only",
           [0, 0, 0, 0, 1, 0], #5 - jump only",
           [0, 0, 1, 0, 0, 1], #6 - left run",
           [0, 0, 1, 0, 1, 0], #7 - left jump",
           [0, 0, 0, 1, 0, 1], #8 - right run",
           [0, 0, 0, 1, 1, 0], #9 - right jump",
           [0, 0, 1, 0, 1, 1], #10 - left run jump",
           [0, 0, 0, 1, 1, 1]]) #11 - right run jump",

        self.action_space = gym.spaces.Discrete(len(self._actions))

    # take an action
    def action(self, a):
        return self._actions[a].copy()


class ProcessRewards(gym.Wrapper):
    def __init__(self, env):
        super(ProcessRewards, self).__init__(env)
        self._max_x = 0
    
    def reset(self, **kwargs):
        self._max_x = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if 'distance' in info:
            self._max_x = max(self._max_x, info['distance'])
        return obs, reward, done, info

def make_env():
    ''' function for editing and returning the environment for mario '''
    env = gym.make('SuperMarioBros-1-1-v0')
    env = ActionsDiscretizer(env)
    env = ProcessRewards(env)
    #env = FrameStack(env, 2)
    return env