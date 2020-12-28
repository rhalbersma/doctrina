#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import copy
import numbers

import numpy as np
import pandas as pd
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


def upper_confidence_bound(Q, N, t, c):
    ucb = Q + c * np.sqrt(np.log(t + 1) / N)
    return argmax_random(ucb)


def sample(np_random, policy):   
    return np_random.choice(np.arange(np.size(policy)), p=policy)

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


def Q_policy_iter_bandit_eps(envs, steps, epsilon, Q0=0, alpha=0):
    runs = len(envs)
    a_hist = np.full((runs, steps),   -1, dtype=int)
    R_hist = np.full((runs, steps), None, dtype=float)
    for r, env in enumerate(tqdm(envs)):
        assert spaces.size(env.observation_space) == 1
        env.seed()
        A = spaces.size(env.action_space)
        Q = np.full(A, Q0, dtype=float)
        if not alpha:
            N = np.zeros(A, dtype=int)
        for t in range(steps):
            a = epsilon_greedy(Q, epsilon) 
            _, R, _, _ = env.step(a)
            if not alpha:
                N[a] += 1
                Q[a] += (R - Q[a]) / N[a]
            else:
                Q[a] += (R - Q[a]) * alpha
            a_hist[r, t] = a
            R_hist[r, t] = R
    return a_hist, R_hist


def Q_policy_iter_bandit_ucb(envs, steps, c, epsilon=1e-6, Q0=0, alpha=0):
    runs = len(envs)
    a_hist = np.full((runs, steps),   -1, dtype=int)
    R_hist = np.full((runs, steps), None, dtype=float)
    for r, env in enumerate(tqdm(envs)):
        assert spaces.size(env.observation_space) == 1
        env.seed()
        A = spaces.size(env.action_space)
        Q = np.full(A, Q0, dtype=float)
        N = np.zeros(A, dtype=int)
        for t in range(steps):
            a = upper_confidence_bound(Q, N + epsilon, t, c)
            _, R, _, _ = env.step(a)
            N[a] += 1
            if not alpha:
                Q[a] += (R - Q[a]) / N[a]
            else:
                Q[a] += (R - Q[a]) * alpha
            a_hist[r, t] = a
            R_hist[r, t] = R
    return a_hist, R_hist


def action_history(a, id, parameter): 
    return (pd
        .DataFrame(a)
        .apply(lambda x: x.value_counts(normalize=True))
        .fillna(0)
        .T
        .rename_axis('steps')
        .reset_index()
        .melt(id_vars='steps', var_name='arm', value_name='selected')
        .assign(
            id = id,
            parameter = parameter
        )
    )


def reward_history(R, id, parameter):
    return (pd
        .DataFrame(R.mean(axis=0), columns=['reward'])
        .rename_axis('steps')
        .reset_index()
        .assign(
            id = id,
            parameter = parameter
        )
    )


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
