#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from collections import defaultdict
import copy
import numbers

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import spaces
from .dp import Q_policy_impr

def is_deterministic(policy):
    return np.issubdtype(policy.dtype, np.integer)

def is_stochastic(policy):
    return np.issubdtype(policy.dtype, np.floating)

def epsilon_soft(policy0, A, epsilon):
    assert is_deterministic(policy0)
    policy = (policy0[..., None] == np.arange(A)).astype(float)
    policy *= 1 - epsilon
    policy += epsilon / A
    assert is_stochastic(policy)
    return policy


def argmax_random(x):
    return np.random.choice(np.flatnonzero(x == x.max()))


def epsilon_greedy(Q, epsilon):
    if epsilon == 0 or np.random.rand(1) > epsilon:
        return argmax_random(Q)
    else:
        return np.random.randint(len(Q))


def upper_confidence_bound(Q, N, t, c, N0=1e-6):
    return Q + c * np.sqrt(np.log(t + 1) / (N + N0))


def sample(policy):
    return np.random.choice(np.arange(np.size(policy)), p=policy)



################################################################################
# Policy selection.
################################################################################


def policy_select_stoch(policy):
    return sample(policy)


def policy_select_deter(policy):
    return policy


# Dispatcher for policy selection.

def dispatch_policy_select(format):
    return {
        'stoch': policy_select_stoch,
        'deter': policy_select_deter,
    }[format]


################################################################################
# Monte Carlo prediction.
################################################################################


def generate_episode(env, policy, format, start=None):
    select = dispatch_policy_select(format)
    trajectory = []
    s = env.reset() if start is None else env.explore(start)
    while True:
        a = select(policy[s])
        next, r, done, _ = env.step(a)
        trajectory.append((s, a, r))
        if done:
            break
        s = next
    return trajectory


def generate_episode_es(env, policy, format):
    select = dispatch_policy_select(format)
    trajectory = []
    s = env.explore(env.observation_space.sample())
    a = env.action_space.sample()
    while True:
        next, r, done, _ = env.step(a)
        trajectory.append((s, a, r))
        if done:
            break
        s = next
        a = select(policy[s])
    return trajectory


def V_predict_ev(env, policy, num_episodes, format='deter', gamma=1., start=None, V0=None, N0=None):
    V = np.zeros(env.nS)            if V0 is None else V0.copy()
    N = np.zeros(env.nS, dtype=int) if N0 is None else N0.copy()
    for _ in tqdm(range(num_episodes)):
        trajectory = generate_episode(env, policy, format, start)
        G = 0.
        for s, _, r in reversed(trajectory):
            G *= gamma
            G += r
            N[s] += 1
            V[s] += (G - V[s]) / N[s]
    return V, N


def Q_predict_ev(env, policy, num_episodes, format='deter', gamma=1., start=None, Q0=None, N0=None):
    Q = np.zeros((env.nS, env.nA))            if Q0 is None else Q0.copy()
    N = np.zeros((env.nS, env.nA), dtype=int) if N0 is None else N0.copy()
    for _ in tqdm(range(num_episodes)):
        trajectory = generate_episode(env, policy, format, start)
        G = 0.
        for s, a, r in reversed(trajectory):
            G *= gamma
            G += r
            N[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / N[s, a]
    return Q, N


def Q_control_ev_eps(env, num_episodes, epsilon, gamma=1., policy0=None, Q0=None, N0=None):
    Q = np.zeros((env.nS, env.nA))            if Q0 is None else Q0.copy()
    N = np.zeros((env.nS, env.nA), dtype=int) if N0 is None else N0.copy()
    policy = np.full((env.nS, env.nA), 1 / env.nA) if policy0 is None else policy0.copy()
    for _ in range(num_episodes):
        trajectory = generate_episode(env, policy, 'stoch')
        G = 0.
        for s, a, r in reversed(trajectory):
            G *= gamma
            G += r
            N[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / N[s, a]
            policy[s] = epsilon_greedy(Q[s], epsilon)
    return policy, Q, N


def Q_control_es(env, num_episodes, format='deter', gamma=1., policy0=None, Q0=None, N0=None):
    Q = np.zeros((env.nS, env.nA))            if Q0 is None else Q0.copy()
    N = np.zeros((env.nS, env.nA), dtype=int) if N0 is None else N0.copy()
    policy = np.full((env.nS, env.nA), 1 / env.nA) if policy0 is None else policy0.copy()
    for _ in tqdm(range(num_episodes)):
        trajectory = generate_episode_es(env, policy, format)
        G = 0.
        for s, a, r in reversed(trajectory):
            G *= gamma
            G += r
            N[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / N[s, a]
            policy[s] = Q_policy_impr(env, Q[s], format)
    assert not (N == 0).any()
    assert (policy == Q.argmax(axis=-1)).all()
    return policy, Q, N




def Q_policy_iter_ucb(env, episodes, policy0=None, c=2., eps=1e-6, gamma=1.):
    """
    On-policy every-visit MC control (using Upper-Confidence Bounds).
    """
    state_shape        = spaces.shape(env.observation_space)
    action_shape       = spaces.shape(env.action_space)
    state_action_shape = (*state_shape, *action_shape)
    Q       = np.zeros(state_action_shape           )
    N       = np.zeros(state_action_shape, dtype=int)
    policy  = np.zeros(state_shape,        dtype=int) if policy0 is None else copy.deepcopy(policy0)
    N_total = 0
    for _ in tqdm(range(episodes)):
        episode = []
        s = env.reset()
        while True:
            a = policy[s]
            next, R, done, _ = env.step(a)
            episode.append((s, a, R))
            if done:
                break
            s = next
        G = 0.
        for s, a, R in reversed(episode):
            G *= gamma
            G += R
            N_total += 1
            N[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / N[s][a]
            policy[s] = np.argmax(Q[s] + c * np.sqrt(np.log(N_total) / (N[s] + eps)))
    assert N_total == N.sum()
    policy = N.argmax(axis=2)
    return policy, Q, N
