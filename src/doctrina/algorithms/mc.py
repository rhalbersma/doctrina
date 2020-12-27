#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import copy
import numbers

import numpy as np
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
    A = len(Q)
    policy = np.full(A, epsilon / A)
    policy[argmax_random(Q)] += 1 - epsilon
    return policy

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

def Q_policy_iter_bandit_eps(envs, episodes, epsilon, Q0=None, N0=None):
    runs = len(envs)
    a = np.full((runs, episodes),   -1, dtype=int)
    R = np.full((runs, episodes), None, dtype=float)
    for r, env in enumerate(tqdm(envs)):
        assert spaces.size(env.observation_space) == 1
        A = spaces.size(env.action_space)
        Q = np.zeros(A)            if Q0 is None else Q0
        N = np.zeros(A, dtype=int) if N0 is None else N0
        policy = np.full(A, 1 / A)
        for t in range(episodes):
            env.reset()
            a[r, t] = sample(env.np_random, policy) 
            _, R[r, t], done, _ = env.step(a[r, t])
            assert done
            N[a[r, t]] += 1
            Q[a[r, t]] += (R[r, t] - Q[a[r, t]]) / N[a[r, t]]
            policy = epsilon_greedy(Q, epsilon)            
    return a, R

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
