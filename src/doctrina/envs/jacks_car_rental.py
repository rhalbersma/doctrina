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

mu_request_1, mu_request_2 = 3, 4
mu_return_1, mu_return_2 = 3, 2

max_evening = 20
max_movable =  5
max_morning = max_evening + max_movable

actions = np.arange(-max_movable, +max_movable + 1)

nE = max_evening + 1
nM = max_morning + 1

nS = nE**2
nA = np.size(actions)
nR = 2 * max_evening + 1


def action_index(a):
    return a + max_movable


def clamp(x, lower, upper):
    return np.maximum(lower, np.minimum(x, upper))


def P_move(sign):
    P = np.zeros((nA, nE, nM))
    for evening in range(nE):
        moved = clamp(sign * actions, -max_movable, min(evening, max_movable))
        for a in range(nA):
            P[a, evening, evening - moved[a]] = 1
    assert np.isclose(P.sum(axis=2), 1).all()
    return P


def P_request(mu_request):
    P = np.zeros((nM, nM, nM))
    for morning in range(nM):
        for rented in range(morning + 1):
            P[rented, morning, morning - rented] = poisson.pmf(rented, mu_request)
        # Excess requests beyond what's available are captured by the survival function (== 1 - CDF)
        P[morning, morning, 0] += poisson.sf(morning, mu_request)
    assert np.isclose(P.sum(axis=(0, 2)), 1).all()
    return P


def P_return(mu_return):
    P = np.zeros((nM, nE))
    for afternoon in range(nM):
        for returned in range(nE - afternoon):
            P[afternoon, afternoon + returned] = poisson.pmf(returned, mu_return)
        # Excess returns beyond what can be kept are captured by teh survival function (== 1 - CDF)
        P[afternoon, max_evening] += poisson.sf(max_evening - afternoon, mu_return)
    assert np.isclose(P.sum(axis=1), 1).all()
    return P


P_move_1 = P_move(+1)
P_move_2 = P_move(-1)
assert (P_move_1 == P_move_2[::-1]).all()

P_request_1 = P_request(mu_request_1)
P_request_2 = P_request(mu_request_2)

P_return_1 = P_return(mu_return_1)
P_return_2 = P_return(mu_return_2)

P1 = np.zeros((nM, nA, nE, nE))
P2 = np.zeros((nM, nA, nE, nE))
for a, r in product(range(nA), range(nM)):
    P1[r, a] = P_move_1[a] @ P_request_1[r] @ P_return_1
    P2[r, a] = P_move_2[a] @ P_request_2[r] @ P_return_2
assert np.isclose(P1.sum(axis=(0, 3)), 1).all()
assert np.isclose(P2.sum(axis=(0, 3)), 1).all()

P12 = np.zeros((nR, nE, nE, nA, nE, nE))
for r1, r2 in product(range(nM), range(nM)):
    if (r1 + r2) < nR:
        for s1, s2 in product(range(nE), range(nE)):
            moved = clamp(actions, -s2, s1)
            for a, m in zip(range(nA), action_index(moved)):
                P12[r1 + r2, s1, s2, a] += P1[r1, m, s1].reshape((nE, 1)) * P2[r2, m, s2].reshape((1, nE))

rent = 10
cost =  2
reward = np.zeros((nE, nE, nA))
for r, s1, s2 in product(range(nR), range(nE), range(nE)):
    moved = clamp(actions, -s2, s1)
    for a, m in zip(range(nA), moved):
        reward[s1, s2, a] += P12[r, s1, s2, a].sum() * (-abs(m) * cost + r * rent)
reward = reward.reshape((nS, nA))

P = P12.reshape((nR, nS, nA, nS)).transpose(1, 2, 3, 0)
assert np.isclose(P.sum(axis=(2, 3)), 1).all()

transition = P.sum(axis=3)
assert np.isclose(transition.sum(axis=2), 1).all()


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


