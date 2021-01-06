#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from itertools import product

import gym
from gym import spaces
from gym.envs.toy_text import discrete
from gym.utils import seeding
import numpy as np
from scipy.stats import poisson

rent = 10
cost =  2
lam_pick_1, lam_pick_2 = 3, 4
lam_drop_1, lam_drop_2 = 3, 2
max_cars = 20
max_move =  5

nE = max_cars + 1   # Number of cars in the evening.
nM = nE + max_move  # Number of cars in the morning.
nS = nE**2
nA = 2 * max_move + 1

def poisson_pdf(max_k, mu):
    return np.array([
        poisson.pmf(k, mu)
        for k in range(max_k)
    ])

def P_pick(mu):
    poisson = poisson_pdf(nM, mu)
    P = np.zeros((nM, nM))
    for m in range(nM):
        for p in range(nM):
            pick = min(p, m)
            P[m, pick] += poisson[p]
    assert np.isclose(P.sum(axis=1), 1).all()
    return P

def P_drop(mu):
    poisson = poisson_pdf(nE, mu)
    P = np.zeros((nM, nE))
    for m in range(nM):
        for d in range(nE):
            drop = min(d, max(0, nE - m - 1))
            P[m, drop] += poisson[d]
    assert np.isclose(P.sum(axis=1), 1).all()
    return P

P1 = P_pick(lam_pick_1)
P2 = P_pick(lam_pick_2)
D1 = P_drop(lam_drop_1)
D2 = P_drop(lam_drop_2)

P = {
    (s1, s2): {
        a: [
            (
                P1[s1 - a, p1] * D1[s1 - a - p1, d1] * P2[s2 + a, p2] * D2[s2 + a - p2, d2],
                (s1 - a - p1 + d1, s2 + a - p2 + d2), 
                -abs(a) * cost + (p1 + p2) * rent, 
                False
            )
            for p1 in range(s1 - a + 1)
            for d1 in range(max(0, nE - (s1 - a - p1) - 1) + 1)
            for p2 in range(s2 + a + 1)
            for d2 in range(max(0, nE - (s2 + a - p2) - 1) + 1)
        ]
        for a in range(-max_move, +max_move + 1)
    }
    for s1, s2 in product(range(nE), range(nE))
}


class JacksCarRentalEnv(discrete.DiscreteEnv):
    """
    Jack's Car Rental environment corresponding to Chapter 4 of
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

    def __init__(self):
        """
        Initialize the state of the environment.
        """




        #super(JacksCarRentalEnv, self).__init__(nS, nA, P, isd)


