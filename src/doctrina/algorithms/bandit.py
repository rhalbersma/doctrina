#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy.special import softmax
from tqdm import tqdm

from doctrina import spaces

rng = default_rng()


def upper_confidence_bound(Q, N, t, c=2, N0=1e-6):
    # We use 0-based time indexing, so we add 1 inside the logarithm here.
    return Q + c * np.sqrt(np.log(t + 1) / (N + N0))


################################################################################
# Action selection: policy -> action.
################################################################################


def select_random_choice(policy):
    return rng.choice(np.arange(policy.size), p=policy)


def select_random_uniform(policy):
    return rng.integers(policy.size)


def select_greedy_argmax(policy):
    return policy.argmax()


def select_greedy_equal(policy):
    return rng.choice(np.flatnonzero(np.equal(policy, policy.max())))


def select_greedy_isclose(policy):
    return rng.choice(np.flatnonzero(np.isclose(policy, policy.max())))


def select_epsilon_greedy(policy, epsilon):
    if epsilon and rng.random() < epsilon:
        return select_random_uniform(policy)
    else:
        return select_greedy_equal(policy)


################################################################################
# Policy improvement.
################################################################################


def greedy(x):
    policy = np.isclose(x, x.max()).astype(float)
    policy /= policy.sum()
    return policy
 

def epsilon_soft(policy, epsilon):
    policy *= 1 - epsilon
    policy += epsilon / policy.size
    return policy


def epsilon_greedy(x, epsilon):
    return epsilon_soft(greedy(x), epsilon)


################################################################################
# Action-value control for multi-armed bandits.
################################################################################


def Q_control_eps(env, num_steps, alpha=0, epsilon=0.1, Q0=0):
    Q = np.full((env.nS, env.nA), Q0, dtype=float)
    N = np.zeros((env.nS, env.nA), dtype=int)
    for s in tqdm(range(env.nS)):
        env.explore(s)
        for _ in range(num_steps):
            a = select_epsilon_greedy(Q[s], epsilon)
            _, r, _, _ = env.step(a)
            N[s, a] += 1
            learning_rate = 1 / N[s, a] if not alpha else alpha
            Q[s, a] += learning_rate * (r - Q[s, a])
    return Q


def Q_control_ucb(env, num_steps, alpha=0, c=2, Q0=0):
    Q = np.full((env.nS, env.nA), Q0, dtype=float)
    N = np.zeros((env.nS, env.nA), dtype=int)
    for s in tqdm(range(env.nS)):
        env.explore(s)
        for t in range(num_steps):
            a = select_greedy_equal(upper_confidence_bound(Q[s], N[s], t + 1, c))
            _, r, _, _ = env.step(a)
            N[s, a] += 1
            learning_rate = 1 / N[s, a] if not alpha else alpha
            Q[s, a] += learning_rate * (r - Q[s, a])
    return Q


def Q_control_grad(env, num_steps, alpha=0.1, baseline=True, tau=1, Q0=0):
    policy = np.full((env.nS, env.nA), 1 / env.nA)
    Q = np.full((env.nS, env.nA), Q0, dtype=float)
    R = np.zeros(env.nS)
    id = np.identity(env.nA)
    for s in tqdm(range(env.nS)):
        env.explore(s)
        for t in range(num_steps):
            a = select_random_choice(policy[s])
            _, r, _, _ = env.step(a)
            Q[s] += alpha * (r - R[s] * baseline) * (id[a] - policy[s])
            R[s] += (r - R[s]) / (t + 1)
            policy[s] = softmax(Q[s] / tau)
    return Q


################################################################################
# Post-processing for multi-armed bandit action and reward histories.
################################################################################


def action_history(a_hist, description, **kwargs):
    return (pd
        .DataFrame(a_hist)
        .apply(lambda x: x.value_counts(normalize=True))
        .fillna(0)
        .T
        .rename_axis('steps')
        .reset_index()
        .melt(id_vars='steps', var_name='arm', value_name='selected')
        .assign(
            description = description,
            **kwargs
        )
    )


def reward_history(r_hist, description, **kwargs):
    return (pd
        .DataFrame(r_hist.mean(axis=0), columns=['reward'])
        .rename_axis('steps')
        .reset_index()
        .assign(
            description = description,
            **kwargs
        )
    )

