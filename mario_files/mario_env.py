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
        self._max_x  = 0
        self._time_  = 400
        self._score_ = 0
    
    def reset(self, **kwargs):
        self._max_x  = 0
        self._time_  = 400
        self._score_ = 0

        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        score_coef   = 0.0001 # tune the score reward
        time_penalty = 0.1 # for every second that passes, give -'time_penalty' reward

        # Check first if distance is in info, this is mario-specific
        if 'distance' in info:
            score_dif = (info['score'] - self._score_) * score_coef
            reward += score_dif

            # time penalty every second
            if info['time'] < self._time_:
                reward -= time_penalty

            # if mario died
            if done and info['life'] == 0:
                reward -= 1

        self._max_x  = max(self._max_x, info['distance'])
        self._score_ = info['score']
        self._time_  = info['time']

        return obs, reward, done, info

def make_env():
    ''' function for editing and returning the environment for mario '''
    env = gym.make('SuperMarioBros-1-1-v0')
    env = ActionsDiscretizer(env)
    env = ProcessRewards(env)
    env.close()
    #env = FrameStack(env, 2)
    return env