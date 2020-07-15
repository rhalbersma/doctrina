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

def epsilon_soft(policy0, nA, epsilon):
    assert is_deterministic(policy0)
    policy = (policy0[..., None] == np.arange(nA)).astype(float)
    policy *= 1. - epsilon
    policy += epsilon / nA
    assert is_stochastic(policy)
    return policy

def epsilon_greedy(Q, epsilon):
    nA = len(Q)
    policy = np.full(nA, epsilon / nA)
    policy[np.argmax(Q)] += 1. - epsilon
    return policy

def sample(np_random, policy):   
    return np_random.choice(np.arange(np.size(policy)), p=policy)

def predict_ev(env, episodes, policy, start=None, gamma=1.):
    """
    Every-visit Monte Carlo prediction.
    """    
    V = np.zeros(spaces.shape(env.observation_space)           )
    N = np.zeros(spaces.shape(env.observation_space), dtype=int)
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

def value_predict(env, policy, start=None, gamma=1., episodes=10**6):
    """
    Last-visit Monte Carlo prediction.
    
    Args:
        env: An OpenAI Gym environment.
        policy: A NumPy array of the same shape as the environment's observation space. 
        start: Defaults to None.
        episodes: The number of episodes to be evaluated. Defaults to one million episodes.
        gamma: The discount rate between successive steps within each episode. Defaults to 1..

    Returns:
        the value function of the policy.

    _Sutton and Barto:
        http://incompleteideas.net/book/the-book.html
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

def control_ev_eps(env, episodes, policy0=None, epsilon0=None, gamma=1.):
    """
    On-policy every-visit MC control (for epsilon-soft policies).
    """
    state_shape        = spaces.shape(env.observation_space)
    action_shape       = spaces.shape(env.action_space)
    state_action_shape = (*state_shape, *action_shape)
    Q      = np.zeros(state_action_shape           )
    N      = np.zeros(state_action_shape, dtype=int)
    nA = spaces.size(env.action_space)
    epsilon = 1. / nA if epsilon0 is None else epsilon0
    if policy0 is None:
        policy = epsilon_soft(np.zeros(state_shape, dtype=int), nA, epsilon) 
    elif is_deterministic(policy0):
        policy = epsilon_soft(policy0, nA, epsilon) 
    else:
        assert is_stochastic(policy0)
        assert (policy0.sum(axis=2) == 1.).all()
        policy = copy.deepcopy(policy0)
    for _ in tqdm(range(episodes)):
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

def control_ev_ucb(env, episodes, policy0=None, c=2., eps=1e-6, gamma=1.):
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
