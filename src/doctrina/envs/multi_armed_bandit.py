#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class MultiArmedBanditEnv(gym.Env):
    """
    Multi-armed bandit environment corresponding to Chapter 2 of
    Reinforcement Learning: An Introduction (2nd ed.) by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html

    Description:
        The object of the multi-armed bandit problem is to minimize the
        cumulative regret by discovering and selecting the arm with the 
        highest expected reward as often as possible.

    Observations:
        There is a single unobservable state.

    Actions:
        There are k actions, one for each of the k arms to select.

    Rewards:
        There is an unbounded continuous reward range.
    """

    def __init__(self, k=10, mu=0, sigma=1, s=1, steps=1_000, stationary=True):
        """
        Initialize the state of the environment.

        Args:
            k (int): the number of arms to select from. Defaults to 10.
            mu (float): the mean when sampling the arms' true values. Defaults to 0.
            sigma (float): the standard deviation when sampling the arms' true values. Defaults to 1.
            s (float): the standard deviation when sampling from a selected arm. Defaults to 1.
            steps (int): the number of steps over which to amortize random number generation. Defaults to 1,000.

        Notes:
            The default arguments correspond to the multi-armed bandit testbed in Chapter 2 of Sutton and Barto's 'Reinforcement Learning' (2018).
        """
        self.k, self.mu, self.sigma, self.s, self.steps = k, mu, sigma, s, steps
        self.time = 0
        self.seed()
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(k)
        self.reward_range = (-np.inf, +np.inf)

    def _amortize_rand(self):
        self.reward = self.np_random.normal(0, self.s, self.steps)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self._amortize_rand()
        self.q_star = np.sort(self.np_random.normal(self.mu, self.sigma, self.k))[::-1]
        return [seed]

    def reset(self):        
        return 0

    def step(self, action):
        if self.time >= self.steps:
            self.time = 0
        reward = self.q_star[action] + self.reward[self.time]        
        self.time += 1
        return 0, reward, True, {}

