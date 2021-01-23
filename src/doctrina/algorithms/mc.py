#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
from tqdm import tqdm

from .dp import Q_policy_impr


################################################################################
# Action selection.
################################################################################


def policy_select_stoch(policy):
    return np.random.choice(np.arange(np.size(policy)), p=policy)


def policy_select_deter(policy):
    return policy


# Dispatcher for policy selection.

def dispatch_policy_select(format):
    return {
        'stoch': policy_select_stoch,
        'deter': policy_select_deter,
    }[format]


def epsilon_soft(policy, epsilon):
    policy *= 1 - epsilon
    policy += epsilon / np.size(policy)
    return policy


################################################################################
# Episode generation.
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


################################################################################
# Monte Carlo prediction.
################################################################################


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


################################################################################
# Monte Carlo control.
################################################################################


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
    return Q, N


def Q_control_ev(env, num_episodes, epsilon=0.1, gamma=1., policy0=None, Q0=None, N0=None):
    Q = np.zeros((env.nS, env.nA))            if Q0 is None else Q0.copy()
    N = np.zeros((env.nS, env.nA), dtype=int) if N0 is None else N0.copy()
    policy = np.full((env.nS, env.nA), 1 / env.nA) if policy0 is None else policy0.copy()
    for _ in tqdm(range(num_episodes)):
        trajectory = generate_episode(env, policy, 'stoch')
        G = 0.
        for s, a, r in reversed(trajectory):
            G *= gamma
            G += r
            N[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / N[s, a]
            policy[s] = epsilon_soft(Q_policy_impr(env, Q[s], 'stoch'), epsilon)
    return Q, N

