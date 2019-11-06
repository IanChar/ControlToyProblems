"""
Random Agent that will run any of the toy environments set up.
Mostly a copy of gym's random agent:
https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
"""

import argparse
import sys

import gym
from gym import wrappers, logger

import os
sys.path.append('..')
from toy_envs import all_envs

class RandomAgent(object):

        def __init__(self, action_space):
            self.action_space = action_space

        def act(self, observation, reward, done):
            return self.action_space.sample()

def make_env(env_name):
    return all_envs[env_name]()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_name', nargs='?', default='CtsCartPoleEnv',
                        help='Select environment to run.')
    args = parser.parse_args()

    logger.set_level(logger.INFO)
    env = make_env(args.env_name)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    episode_count = 10
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        cum_reward = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            cum_reward += reward
            env.render()
            if done:
                break
        print('Cumulative Episode Score: %f' % cum_reward)
    env.close()
