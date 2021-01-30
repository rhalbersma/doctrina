#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
from tqdm import tqdm

from .bandit import select_epsilon_greedy, epsilon_greedy
from .mc import dispatch_policy_select


################################################################################
# Temporal-Difference prediction.
################################################################################


def V_predict(env, policy, num_episodes, alpha=0.1, format='deter', gamma=1., start=None, V0=None):
    V = np.zeros(env.nS + 1) if V0 is None else V0.copy()
    select = dispatch_policy_select(format)
    for _ in tqdm(range(num_episodes)):
        s = env.reset() if start is None else env.explore(start)
        while True:
            a = select(policy[s])
            next, r, done, _ = env.step(a)
            V[s] += alpha * (r + gamma * V[next]- V[s])
            if done:
                break
            s = next            
    return (V[:-1],)


################################################################################
# Temporal-Difference control.
################################################################################


def sarsa(env, num_episodes, alpha=0.5, epsilon=0.1, gamma=1., Q0=None):
    Q = np.zeros((env.nS + 1, env.nA)) if Q0 is None else Q0.copy()
    for _ in tqdm(range(num_episodes)):
        s = env.reset()
        a = select_epsilon_greedy(Q[s], epsilon)
        while True:
            s_next, r, done, _ = env.step(a)
            a_next = select_epsilon_greedy(Q[s_next], epsilon)
            Q[s, a] += alpha * (r + gamma * Q[s_next, a_next] - Q[s, a])
            if done:
                break
            s, a = s_next, a_next
    return Q[:-1]


def Q_learning(env, num_episodes, alpha=0.5, epsilon=0.1, gamma=1., Q0=None):
    Q = np.zeros((env.nS + 1, env.nA)) if Q0 is None else Q0.copy()
    for _ in tqdm(range(num_episodes)):
        s = env.reset()
        while True:
            a = select_epsilon_greedy(Q[s], epsilon)
            next, r, done, _ = env.step(a)
            Q[s, a] += alpha * (r + gamma * Q[next].max(axis=-1) - Q[s, a])
            if done:
                break
            s = next
    return Q[:-1]


def expected_sarsa(env, num_episodes, alpha=0.5, epsilon=0.1, gamma=1., Q0=None):
    Q = np.zeros((env.nS + 1, env.nA)) if Q0 is None else Q0.copy()
    policy = np.full((env.nS + 1, env.nA), 1 / env.nA)
    for _ in tqdm(range(num_episodes)):
        s = env.reset()
        while True:
            a = select_epsilon_greedy(Q[s], epsilon)
            next, r, done, _ = env.step(a)
            Q[s, a] += alpha * (r + gamma * np.sum(policy[next] * Q[next], axis=-1) - Q[s, a])
            policy[s] = epsilon_greedy(Q[s], epsilon)
            if done:
                break
            s = next
    return Q[:-1]

