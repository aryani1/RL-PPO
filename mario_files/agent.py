import tensorflow as tf
import numpy as np
import gym

import Model
import architecture
import mario_env

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

def main():
    nenvs = 2
    policy   = architecture.PPO
    env_list = []

    for _ in range(nenvs):
        env_list.append(mario_env.make_env)

    with tf.Session():
        Model.learn(policy=policy,
                    env=SubprocVecEnv(env_list),
                    nsteps=2048,
                    total_timesteps=1000000,
                    gamma=0.99,
                    lam = 0.95,
                    v_coef=0.5,
                    ent_coef=0.01,
                    lr = 2e-4,
                    cliprange = 0.1,
                    max_grad_norm=0.5,
                    log_interval=10)

if __name__ == '__main__':
    main() 