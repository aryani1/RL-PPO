import tensorflow as tf
import numpy as np
import gym

import Model
import architecture

def main():
    model.play(policy=architecture.PPO,
               env=gym.make('SuperMarioBros-1-1-v0'),
               update=120)