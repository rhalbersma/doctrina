#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import copy
import numbers

import numpy as np
import pandas as pd
from scipy.special import softmax
from tqdm import tqdm

from doctrina import spaces

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


def Q_control_bandit_eps(envs, T, epsilon=0.1, Q0=0, alpha=0):
    S, A = np.size(envs), spaces.size(envs[0].action_space)
    Q = np.full((S, A), Q0, dtype=float)
    N = np.zeros((S, A), dtype=int)
    R = np.zeros(S)
    a_hist = np.full((S, T),   -1, dtype=int)
    r_hist = np.full((S, T), None, dtype=float)
    for s, env in enumerate(tqdm(envs)):
        for t in range(T):
            a = epsilon_greedy(Q[s], epsilon)
            _, r, _, _ = env.step(a)
            a_hist[s, t] = a
            r_hist[s, t] = r
            N[s, a] += 1
            step_size = 1 / N[s, a] if not alpha else alpha
            Q[s, a] += (r - Q[s, a]) * step_size
            R[s] += (r - R[s]) / (t + 1)
    return None, Q, N, R, a_hist, r_hist


def Q_control_bandit_ucb(envs, T, c=2, Q0=0, alpha=0):
    S, A = np.size(envs), spaces.size(envs[0].action_space)
    policy = np.zeros((S, A))
    Q = np.full((S, A), Q0, dtype=float)
    N = np.zeros((S, A), dtype=int)
    R = np.zeros(S)
    a_hist = np.full((S, T),   -1, dtype=int)
    r_hist = np.full((S, T), None, dtype=float)
    for s, env in enumerate(tqdm(envs)):
        for t in range(T):
            a = argmax_random(policy[s])
            _, r, _, _ = env.step(a)
            a_hist[s, t] = a
            r_hist[s, t] = r
            N[s, a] += 1
            step_size = 1 / N[s, a] if not alpha else alpha
            Q[s, a] += (r - Q[s, a]) * step_size
            R[s] += (r - R[s]) / (t + 1)
            policy[s] = upper_confidence_bound(Q[s], N[s], t + 1, c)
    return policy, Q, N, R, a_hist, r_hist


def Q_control_bandit_grad(envs, T, alpha=0.1, Q0=0, tau=1, baseline=True):
    S, A = np.size(envs), spaces.size(envs[0].action_space)
    policy = np.full((S, A), 1 / A)
    Q = np.full((S, A), Q0, dtype=float)
    N = np.zeros((S, A), dtype=int)
    R = np.zeros(S)
    a_hist = np.full((S, T),   -1, dtype=int)
    r_hist = np.full((S, T), None, dtype=float)
    id = np.identity(A)
    for s, env in enumerate(tqdm(envs)):
        for t in range(T):
            a = sample(policy[s])
            _, r, _, _ = env.step(a)
            a_hist[s, t] = a
            r_hist[s, t] = r
            N[s, a] += 1
            Q[s] += alpha * (r - R[s] * baseline) * (id[a] - policy[s])
            R[s] += (r - R[s]) / (t + 1)
            policy[s] = softmax(tau * Q[s])
    return policy, Q, N, R, a_hist, r_hist


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


def V_policy_eval(env, policy, episodes, V0=None, N0=None, gamma=1.):
    """
    Every-visit Monte Carlo prediction.
    """
    S = spaces.size(env.observation_space)
    V = np.zeros((S))            if V0 is None else V0
    N = np.zeros((S), dtype=int) if N0 is None else N0
    for _ in tqdm(range(episodes)):
        trajectory = []
        s = env.reset() if start is None else env.explore(start)
        while True:
            a = policy[s]
            next, R, done, _ = env.step(a)
            trajectory.append((s, R))
            if done:
                break
            s = next
        G = 0.
        for s, R in reversed(trajectory):
            G *= gamma
            G += R
            N[s] += 1
            V[s] += (G - V[s]) / N[s]
    return V, N

def Q_policy_eval(env, policy, episodes, Q0=None, N0=None, gamma=1.):
    S = spaces.size(env.observation_space)
    A = spaces.size(env.action_space)
    Q = np.zeros((S, A))            if Q0 is None else Q0
    N = np.zeros((S, A), dtype=int) if N0 is None else N0
    for _ in tqdm(episodes):
        s = env.reset()
        while True:
            a = sample(env.np_random, policy[s])
            next, R, done, _ = env.step(a)
            trajectory.append((s, a, R))
            if done:
                break
            s = next
        G = 0.
        for s, a, R in reversed(trajectory):
            G *= gamma
            G += R
            N[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / N[s, a]
    return Q, N

def value_predict(env, policy, start=None, gamma=1., episodes=10**6):
    """
    Last-visit Monte Carlo prediction.
    """
    V = 0.
    for _ in tqdm(range(episodes)):
        s = env.reset(start)
        while True:
            a = policy[s]
            next, R, done, _ = env.step(a)
            if done:
                break
            s = next
        V += R
    return V / episodes

def control_es(env, episodes, policy0=None, gamma=1.):
    """
    Monte Carlo ES (Exploring Starts).
    """
    state_shape        = spaces.shape(env.observation_space)
    action_shape       = spaces.shape(env.action_space)
    state_action_shape = (*state_shape, *action_shape)
    Q      = np.zeros(state_action_shape           )
    N      = np.zeros(state_action_shape, dtype=int)
    policy = np.zeros(state_shape,        dtype=int) if policy0 is None else copy.deepcopy(policy0)
    for _ in tqdm(range(episodes)):
        trajectory = []
        s = env.explore(env.observation_space.sample())
        a = env.action_space.sample()
        while True:
            next, R, done, _ = env.step(a)
            trajectory.append((s, a, R))
            if done:
                break
            s = next
            a = policy[s]
        G = 0.
        for s, a, R in reversed(trajectory):
            G *= gamma
            G += R
            N[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / N[s][a]
            policy[s] = np.argmax(Q[s])
    assert not (N == 0).any()
    assert (policy == Q.argmax(axis=2)).all()
    return policy, Q, N

def Q_policy_iter_ev_eps(env, episodes, epsilon, Q0=None, N0=None, gamma=1.):
    """
    On-policy every-visit MC control (for epsilon-soft policies).
    """
    S = spaces.size(env.observation_space)
    A = spaces.size(env.action_space)
    Q = np.zeros((S, A))            if Q0 is None else Q0
    N = np.zeros((S, A), dtype=int) if N0 is None else N0
    policy = np.full((S, A), 1 / A)
    for _ in range(episodes):
        episode = []
        s = env.reset()
        while True:
            a = sample(env.np_random, policy[s])
            next, R, done, _ = env.step(a)
            episode.append((s, a, R))
            if done:
                break
            s = next
        G = 0.
        for s, a, R in reversed(episode):
            G *= gamma
            G += R
            N[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / N[s][a]
            policy[s] = epsilon_greedy(Q[s], epsilon)
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
