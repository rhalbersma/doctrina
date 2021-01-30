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
        There are nA actions, one for each of the nA arms to select.

    Rewards:
        There is an unbounded continuous reward range.
    """

    def __init__(self, nS=2_000, nA=10, mu_a=0, sigma_a=1, sigma_r=1):
        """
        Initialize the state of the environment.

        Args:
            nS (int): the number of bandits. Defaults to 2,000.
            nA (int): the number of arms to select from. Defaults to 10.
            mu_a (float): the mean when sampling the arms' true values. Defaults to 0.
            sigma_a (float): the standard deviation when sampling the arms' true values. Defaults to 1.
            sigma_r (float): the standard deviation when sampling the rewards from a selected arm. Defaults to 1.

        Notes:
            The default arguments correspond to the multi-armed bandit testbed in Chapter 2 of Sutton and Barto's 'Reinforcement Learning' (2018).
        """
        self.nS, self.nA, self.mu_a, self.sigma_a, self.sigma_r = nS, nA, mu_a, sigma_a, sigma_r
        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)
        self.reward_range = (-np.inf, +np.inf)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.q_star = np.flip(np.sort(self.np_random.normal(self.mu_a, self.sigma_a, (self.nS, self.nA))), axis=-1)
        return [seed]

    def reset(self):
        self.s = 0
        return self.s

    def explore(self, start):
        self.s = start
        return self.s

    def step(self, action):
        reward = self.q_star[self.s, action] + self.np_random.normal(0, self.sigma_r)
        return self.s, reward, False, {}

