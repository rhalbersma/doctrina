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

################################################################################
# Environment parameters.
################################################################################

# If Jack has a car available, he rents it out and is credited $10 by the national company.
rent = 10

# [...] Jack can move them between the two locations overnight, at a cost of $2 per car moved.
cost =  2

# Suppose lambda is 3 and 4 for rental requests at the first and second locations and 3 and 2 for returns.
mu_request_1 = 3
mu_request_2 = 4
mu_return_1 = 3
mu_return_2 = 2

# [...] we assume that there can be no more than 20 cars at each location [...]
max_evening = 20

# [...] and a maximum of five cars can be moved from one location to the other in one night.
max_movable =  5
max_morning = max_evening + max_movable
nE = max_evening + 1
nM = max_morning + 1

actions = np.arange(-max_movable, +max_movable + 1, dtype=int)
direction_1 = +1
direction_2 = -1

nS = nE**2
nA = np.size(actions)
nR = 2 * max_evening + 1

################################################################################
# Auxiliary functions.
################################################################################


def clamp(a, lower, upper):
    """Clamp an action between lower and upper values."""
    return np.maximum(lower, np.minimum(a, upper))


def index(action):
    """Compute the index of an action."""
    return action + max_movable


# clamped_actions[s1, s2, a] = clamped actions when moving a cars from s1 to s2.
# For positive a, s1 is the upper bound; for negative a, s2 is the lower bound.
clamped_actions = np.zeros((nS, nA), dtype=int)
for s, (s1, s2) in enumerate(product(range(nE), range(nE))):
    clamped_actions[s] = index(clamp(actions, -s2, s1))

################################################################################
# Transition probability functions.
#
# action:           [move]            [request]              [return]
# state:  [evening] ------> [morning] ---------> [afternoon] --------> [evening]
################################################################################


def prob_move(direction):
    """
    p[a, s, s'] = the probability of moving a cars and transitioning from state s to state s'.

    Notes:
        Positive (negative) actions correspond to cars are moved from (to) state s.
        Negative actions are bounded by max_movable (=5); positive actions by the state s.
    """
    prob = np.zeros((nA, nE, nM))
    for evening in range(nE):
        moved = clamp(direction * actions, -max_movable, min(evening, max_movable))
        for a in range(nA):
            morning = evening - moved[a]
            prob[a, evening, morning] = 1
    assert np.isclose(prob.sum(axis=2), 1).all()
    return prob


def prob_request(mu_request):
    """
    p[r, s, s'] = the probability of fullfilling r rental requests and transitioning from state s to state s'.

    Notes:
        Rental requests are bounded by the state s.
        The probabilities are given by the Poisson distribution with mu = mu_request.
    """
    prob = np.zeros((nM, nM, nM))
    for morning in range(nM):
        for rented in range(morning + 1):
            afternoon = morning - rented
            prob[rented, morning, afternoon] = poisson.pmf(rented, mu_request)
        # Excess requests beyond what's available are captured by the survival function (== 1 - CDF)
        assert afternoon == 0
        prob[morning, morning, afternoon] += poisson.sf(morning, mu_request)
    assert np.isclose(prob.sum(axis=(0, 2)), 1).all()
    return prob


def prob_return(mu_return):
    """
    p[s, s'] = the probability of transitioning from state s to state s' after cars have been returned.

    Notes:
        Car returns are bounded by max_evening (=20).
        The probabilities are given by the Poisson distribution with mu = mu_return.
    """
    prob = np.zeros((nM, nE))
    for afternoon in range(nM):
        for returned in range(nE - afternoon):
            evening = afternoon + returned
            prob[afternoon, evening] = poisson.pmf(returned, mu_return)
        # Excess returns beyond what can be kept are captured by the survival function (== 1 - CDF)
        assert evening == max_evening
        prob[afternoon, evening] += poisson.sf(evening - afternoon, mu_return)
    assert np.isclose(prob.sum(axis=1), 1).all()
    return prob


def prob_location(direction, mu_request, mu_return):
    """
    p[s, a, s', r] = probability of transition to state s' with reward r, from state s and action a.
    """
    prob_mov = prob_move(direction)
    prob_req = prob_request(mu_request)
    prob_ret = prob_return(mu_return)
    prob = np.zeros((nA, nM, nE, nE))       # The computation is most natural in this index order.
    for a, r in product(range(nA), range(nM)):
        prob[a, r] = prob_mov[a] @ prob_req[r] @ prob_ret
    prob = prob.transpose(2, 0, 3, 1)
    assert prob.shape == (nE, nA, nE, nM)   # The result will be used in this index order.
    assert np.isclose(prob.sum(axis=(2, 3)), 1).all()
    return prob


def prob_transition():
    """
    p[s, a, s'] = probability of transition to state s', from state s and action a.
    """
    prob_sas_1 = prob_location(direction_1, mu_request_1, mu_return_1).sum(axis=3).reshape((nE, nA, nE, 1))
    prob_sas_2 = prob_location(direction_2, mu_request_2, mu_return_2).sum(axis=3).reshape((nE, nA, 1, nE))
    prob_sas_12 = np.zeros((nS, nA, nE, nE))
    for s, (s1, s2) in enumerate(product(range(nE), range(nE))):
        for a in range(nA):                         # An action can involve up to 5 moved cars,
            m = clamped_actions[s, a]               # but only if there are enough cars to start with.
            prob_sas_12[s, a] = prob_sas_1[s1, m] * prob_sas_2[s2, m]
    prob = prob_sas_12.reshape((nS, nA, nS))
    assert np.isclose(prob.sum(axis=2), 1).all()
    return prob


def prob_rentals():
    """
    prob[s, a, r] = the probability of r rentals, from state s and action a.
    """
    prob_sar_1 = prob_location(direction_1, mu_request_1, mu_return_1).sum(axis=2).reshape((nE, nA, nM, 1))
    prob_sar_2 = prob_location(direction_2, mu_request_2, mu_return_2).sum(axis=2).reshape((nE, nA, 1, nM))
    prob_sar_12 = np.zeros((nS, nA, nM, nM))
    for s, (s1, s2) in enumerate(product(range(nE), range(nE))):
        for a in range(nA):                         # An action can involve up to 5 moved cars,
            m = clamped_actions[s, a]               # but only if there are enough cars to start with.
            prob_sar_12[s, a] = prob_sar_1[s1, m] * prob_sar_2[s2, m]
    prob = np.zeros((nS, nA, nR))
    for r1, r2 in product(range(nM), range(nM)):    # An individual location can rent up to 25 cars,
        if r1 + r2 < nR:                            # but both locations combined can rent up to 40 cars.
            prob[..., r1 + r2] += prob_sar_12[..., r1, r2]
    assert np.isclose(prob.sum(axis=2), 1).all()
    return prob


def immediate_reward():
    """
    R[s, a, r] = the immediate reward from renting r cars, from state s and action a.
    """
    reward = np.zeros((nS, nA, nR))
    for s, a in product(range(nS), range(nA)):
        m = clamped_actions[s, a]
        d = actions[a] - actions[m]
        for r in range(nR):
            reward[s, a, r] = -(abs(actions[m]) + 0.001 * abs(d)) * cost + r * rent
    return reward


################################################################################
# Sparse representation of the environment's model.
################################################################################


def step_location(direction, mu_request, mu_return):
    """
    P[s][a] = a nested dictionary of lists of (prob, next, rentals) tuples, from state s and action a.
    """
    prob = prob_location(direction, mu_request, mu_return)
    P = {
        s: {
            a: [
                (prob[s, a, next, r], next, r)
                for next, r in product(range(nE), range(nM))
                if prob[s, a, next, r] > 0  # All probabilities are non-zero, but we keep this for generality's sake.
            ]
            for a in range(nA)
        }
        for s in range(nE)
    }
    return P


def step_cdf(P):
    cdf = {
        s: {
            a: np.array([
                t[0]
                for t in P[s][a]
            ]).cumsum()
            for a in P[s].keys()
        }
        for s in P.keys()
    }
    return cdf


################################################################################
# Jack's Car Rental environment.
################################################################################


class JacksCarRentalEnv(discrete.DiscreteEnv):
    """
    Jack's Car Rental environment corresponding to Chapter 4 of
    Reinforcement Learning: An Introduction (2nd ed.) by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html

    Description:
        The object of Jacks's Car Rental problem is to maximize cumulative profits.

    Observations:
        There are 21 * 21 = 441 states.

    Actions:
        There are -5...+5 = 11 actions.

    Rewards:
        There is an unbounded continuous reward range.
    """

    def __init__(self):
        """
        Initialize the state of the environment.
        """
        # Equation (3.4) in Sutton & Barto (p.49):
        # p(s'|s, a) = probability of transition to state s', from state s taking action a.
        self.transition = prob_transition()
        assert np.isclose(self.transition.sum(axis=2), 1).all()

        # Equation (3.5) in Sutton & Barto (p.49):
        # r(s, a) = expected immediate reward from state s after action a.
        self.immediate_reward = immediate_reward()
        self.reward = np.sum(prob_rentals() * self.immediate_reward, axis=2)
        self.reward_range = (np.min(self.reward), np.max(self.reward))

        self.P = (
            step_location(direction_1, mu_request_1, mu_return_1),
            step_location(direction_2, mu_request_2, mu_return_2)
        )
        self.isd = np.full(nS, 1 / nS)

        # For better sampling performance, we precompute the cumulative distributions
        # for both the P dictionaries and the initial state distribution.
        self.step_cdf = (
            step_cdf(self.P[0]),
            step_cdf(self.P[1])
        )
        self.reset_cdf = self.isd.cumsum()

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)
        self.nS = nS
        self.nA = nA

        # For continuing tasks, there is no terminal state.
        self.nSp = self.nS

        self.seed()
        self.reset()

    def _sample_from_categorical(self, cdf):
        return int((cdf > self.np_random.rand()).argmax())

    def _step_location(self, a, s, loc):
        cdf = self.step_cdf[loc][s][a]
        idx = self._sample_from_categorical(cdf)
        return self.P[loc][s][a][idx]

    def step(self, a):
        m = clamped_actions[self.s, a]
        s = divmod(self.s, nE)
        prob, next, rentals = tuple(
            list(t)
            for t in zip(*[
                self._step_location(m, s[i], i)
                for i in range(2)
            ])
        )
        prob = np.prod(prob)
        next = np.array(next) @ np.array([nE, 1])
        rentals = np.sum(rentals)
        reward = self.immediate_reward[self.s, m, rentals]
        self.s = next
        self.lastaction = m
        return next, reward, False, { 'prob': prob }

    def reset(self):
        self.s = self._sample_from_categorical(self.reset_cdf)
        self.lastaction = None
        return self.s

    def explore(self, start):
        """
        Explore a specific starting state of the environment.

        Notes:
            This is an extension of the OpenAI Gym interface.
            Monte Carlo Exploring Starts should use this method instead of reset().
        """
        self.s = start
        self.lastaction = None
        return self.s

