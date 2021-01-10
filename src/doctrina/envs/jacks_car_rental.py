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

# [...] ]Jack can move them between the two locations overnight, at a cost of $2 per car moved.
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

# An individual location can rent up to 25 cars,
# but both locations combined can rent up to 40 cars.
rentals = [
    (r1, r2)
    for r1, r2 in product(range(nM), range(nM))
    if (r1 + r2) < nR
]

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
clamped_actions = np.zeros((nE, nE, nA), dtype=int)
for s1, s2 in product(range(nE), range(nE)):
    clamped_actions[s1, s2] = index(clamp(actions, -s2, s1))

################################################################################
# Transition probability functions.
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
            prob[a, evening, evening - moved[a]] = 1
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
            prob[rented, morning, morning - rented] = poisson.pmf(rented, mu_request)
        # Excess requests beyond what's available are captured by the survival function (== 1 - CDF)
        prob[morning, morning, 0] += poisson.sf(morning, mu_request)
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
            prob[afternoon, afternoon + returned] = poisson.pmf(returned, mu_return)
        # Excess returns beyond what can be kept are captured by the survival function (== 1 - CDF)
        prob[afternoon, max_evening] += poisson.sf(max_evening - afternoon, mu_return)
    assert np.isclose(prob.sum(axis=1), 1).all()
    return prob


def prob_location(direction, mu_request, mu_return):
    """
    p[a, r, s, s'] = the probability of moving a cars, renting r cars and transitioning from state s to s' after cars have been returned.
    """
    prob_mov = prob_move(direction)
    prob_req = prob_request(mu_request)
    prob_ret = prob_return(mu_return)
    prob = np.zeros((nA, nM, nE, nE))
    for a, r in product(range(nA), range(nM)):
        prob[a, r] = prob_mov[a] @ prob_req[r] @ prob_ret
    assert np.isclose(prob.sum(axis=(1, 3)), 1).all()
    return prob


def prob_daily():
    """
    p[s1, s2, a, r, s1', s2'] = the probability of moving a cars, renting r cars and transitioning from state (s1, s2) to (s1', s2') after cars have been returned.
    """
    prob_location_1 = prob_location(direction_1, mu_request_1, mu_return_1)
    prob_location_2 = prob_location(direction_2, mu_request_2, mu_return_2)
    prob = np.zeros((nE, nE, nA, nR, nE, nE))
    for s1, s2, a in product(range(nE), range(nE), range(nA)):
        m = clamped_actions[s1, s2, a]
        for r1, r2 in rentals:
            prob[s1, s2, a, r1 + r2] += prob_location_1[m, r1, s1].reshape((nE, 1)) * prob_location_2[m, r2, s2].reshape((1, nE))
    return prob.transpose(0, 1, 2, 4, 5, 3).reshape((nS, nA, nS, nR))


def reward_state_action_rentals():
    reward = np.zeros((nS, nA, nR))
    for s, (s1, s2) in enumerate(product(range(nE), range(nE))):
        for a, r in product(range(nA), range(nR)):
            m = clamped_actions[s1, s2, a]
            reward[s, a, r] = -abs(actions[a]) * cost + r * rent
    return reward


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
        # Equation (3.2) in Sutton & Barto (p.48):
        # p(s', r|s, a) = probability of transition to state s' with reward r, from state s and action a.
        P_tensor = prob_daily()

        # Equation (3.3) in Sutton & Barto (p.48):
        assert np.isclose(P_tensor.sum(axis=(2, 3)), 1).all()

        # Equation (3.4) in Sutton & Barto (p.49):
        # p(s'|s, a) = probability of transition to state s', from state s taking action a.
        self.transition = P_tensor.sum(axis=3)
        assert np.isclose(self.transition.sum(axis=2), 1).all()

        # Equation (3.5) in Sutton & Barto (p.49):
        # r(s, a) = expected immediate reward from state s after action a.
        immediate_reward = reward_state_action_rentals()
        self.reward = np.sum(P_tensor.sum(axis=2) * immediate_reward, axis=2)

        P = {
            s: {
                a: [
                    (P_tensor[s, a, next, r], next, immediate_reward[s, a, r], False)
                    for next, r in product(range(nS), range(nR))
                ]
                for a in range(nA)
            }
            for s in range(nS)
        }

        isd = np.full(nS, 1 / nS)

        # For continuing tasks, there is no terminal state.
        self.nSp = nS

        super(JacksCarRentalEnv, self).__init__(nS, nA, P, isd)

