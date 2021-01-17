#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import pandas as pd
from scipy.special import softmax
from tqdm import tqdm

from doctrina import spaces


################################################################################
# Action selections.
################################################################################


def argmax_random(policy):
    return np.random.choice(np.flatnonzero(policy == policy.max()))


def epsilon_greedy(Q, epsilon):
    if epsilon == 0 or np.random.rand(1) > epsilon:
        return argmax_random(Q)
    else:
        return np.random.randint(np.size(Q))


def sample(policy):
    return np.random.choice(np.arange(np.size(policy)), p=policy)


################################################################################
# Policy updates.
################################################################################


def upper_confidence_bound(Q, N, t, c, N0=1e-6):
    # We use 0-based time indexing, so add 1 inside log here.
    return Q + c * np.sqrt(np.log(t + 1) / (N + N0))


################################################################################
# Action-value control for multi-armed bandits.
################################################################################


def Q_control_eps(envs, nT, epsilon=0.1, Q0=0, alpha=0):
    nS, nA = np.size(envs), spaces.size(envs[0].action_space)
    Q = np.full((nS, nA), Q0, dtype=float)
    N = np.zeros((nS, nA), dtype=int)
    R = np.zeros(nS)
    a_hist = np.full((nS, nT),   -1, dtype=int)
    r_hist = np.full((nS, nT), None, dtype=float)
    for s, env in enumerate(tqdm(envs)):
        for t in range(nT):
            a = epsilon_greedy(Q[s], epsilon)
            _, r, _, _ = env.step(a)
            a_hist[s, t] = a
            r_hist[s, t] = r
            N[s, a] += 1
            step_size = 1 / N[s, a] if not alpha else alpha
            Q[s, a] += step_size * (r - Q[s, a])
            R[s] += (r - R[s]) / (t + 1)
    return None, Q, N, R, a_hist, r_hist


def Q_control_ucb(envs, nT, c=2, Q0=0, alpha=0):
    nS, nA = np.size(envs), spaces.size(envs[0].action_space)
    policy = np.zeros((nS, nA))
    Q = np.full((nS, nA), Q0, dtype=float)
    N = np.zeros((nS, nA), dtype=int)
    R = np.zeros(nS)
    a_hist = np.full((nS, nT),   -1, dtype=int)
    r_hist = np.full((nS, nT), None, dtype=float)
    for s, env in enumerate(tqdm(envs)):
        for t in range(nT):
            a = argmax_random(policy[s])
            _, r, _, _ = env.step(a)
            a_hist[s, t] = a
            r_hist[s, t] = r
            N[s, a] += 1
            step_size = 1 / N[s, a] if not alpha else alpha
            Q[s, a] += step_size * (r - Q[s, a])
            R[s] += (r - R[s]) / (t + 1)
            policy[s] = upper_confidence_bound(Q[s], N[s], t + 1, c)
    return policy, Q, N, R, a_hist, r_hist


def Q_control_grad(envs, nT, alpha=0.1, Q0=0, tau=1, baseline=True):
    nS, nA = np.size(envs), spaces.size(envs[0].action_space)
    policy = np.full((nS, nA), 1 / nA)
    Q = np.full((nS, nA), Q0, dtype=float)
    N = np.zeros((nS, nA), dtype=int)
    R = np.zeros(nS)
    a_hist = np.full((nS, nT),   -1, dtype=int)
    r_hist = np.full((nS, nT), None, dtype=float)
    id = np.identity(nA)
    for s, env in enumerate(tqdm(envs)):
        for t in range(nT):
            a = sample(policy[s])
            _, r, _, _ = env.step(a)
            a_hist[s, t] = a
            r_hist[s, t] = r
            N[s, a] += 1
            Q[s] += alpha * (r - R[s] * baseline) * (id[a] - policy[s])
            R[s] += (r - R[s]) / (t + 1)
            policy[s] = softmax(Q[s] / tau)
    return policy, Q, N, R, a_hist, r_hist


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

