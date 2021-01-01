#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

from itertools import product

import numpy as np

from doctrina.utils import one_hot_encode

################################################################################
# Policy initialization.
################################################################################


def policy_init_deter(env):
    return np.zeros(env.nS + 1, dtype=int)


def policy_init_stoch(env):
    policy = policy_init_deter(env)
    return one_hot_encode(policy, env.nA)


################################################################################
# Policy evaluation: policy -> value function.
################################################################################


def V_policy_eval_stoch_sync(env, policy, V0=None, gamma=1., tol=1e-6, maxiter=100):
    nS = env.nS + 1
    P, R = env.transition, env.reward
    V = np.zeros(nS) if V0 is None else V0
    iter = 0
    while True:
        v = V
        V = np.sum(policy * (R + gamma * P @ V), axis=1)
        delta = np.max(np.abs(v - V))
        iter += 1
        if delta < tol or iter >= maxiter:
            break
    return V, delta, iter


def Q_policy_eval_stoch_sync(env, policy, Q0=None, gamma=1., tol=1e-6, maxiter=100):
    nS, nA = env.nS + 1, env.nA
    P, R = env.transition, env.reward
    Q = np.zeros((nS, nA)) if Q0 is None else Q0
    iter = 0
    while True:
        q = Q
        Q = R + gamma * P @ np.sum(policy * Q, axis=1)
        delta = np.max(np.abs(q - Q))
        iter += 1
        if delta < tol or iter >= maxiter:
            break
    return Q, delta, iter


def V_policy_eval_stoch_async(env, policy, V0=None, gamma=1., tol=1e-6, maxiter=100):
    nS = env.nS + 1
    P, R = env.transition, env.reward
    V = np.zeros(nS) if V0 is None else V0
    iter = 0
    while True:
        delta = 0.
        for s in range(nS):
            v = V[s]
            V[s] = np.sum(policy[s] * (R[s] + gamma * P[s] @ V))
            delta = np.maximum(delta, np.abs(v - V[s]))
        iter += 1
        if delta < tol or iter >= maxiter:
            break
    return V, delta, iter


def Q_policy_eval_stoch_async(env, policy, Q0=None, gamma=1., tol=1e-6, maxiter=100):
    nS, nA = env.nS + 1, env.nA
    P, R = env.transition, env.reward
    Q = np.zeros((nS, nA)) if Q0 is None else Q0
    iter = 0
    while True:
        delta = 0.
        for s, a in product(range(nS), range(nA)):
            q = Q[s, a]
            Q[s, a] = R[s, a] + gamma * P[s, a] @ np.sum(policy * Q, axis=1)
            delta = np.maximum(delta, np.abs(q - Q[s, a]))
        iter += 1
        if delta < tol or iter >= maxiter:
            break
    return Q, delta, iter


def V_policy_eval_deter_sync(env, policy, V0=None, gamma=1., tol=1e-6, maxiter=100):
    nS = env.nS + 1
    P, R = env.transition, env.reward
    V = np.zeros(nS) if V0 is None else V0
    iter = 0
    while True:
        v = V
        V = np.choose(policy, (R + gamma * P @ V).T)
        delta = np.max(np.abs(v - V))
        iter += 1
        if delta < tol or iter >= maxiter:
            break
    return V, delta, iter


def Q_policy_eval_deter_sync(env, policy, Q0=None, gamma=1., tol=1e-6, maxiter=100):
    nS, nA = env.nS + 1, env.nA
    P, R = env.transition, env.reward
    Q = np.zeros((nS, nA)) if Q0 is None else Q0
    iter = 0
    while True:
        q = Q
        Q = R + gamma * P @ np.choose(policy, Q.T)
        delta = np.max(np.abs(q - Q))
        iter += 1
        if delta < tol or iter >= maxiter:
            break
    return Q, delta, iter


def V_policy_eval_deter_async(env, policy, V0=None, gamma=1., tol=1e-6, maxiter=100):
    nS = env.nS + 1
    P, R = env.transition, env.reward
    V = np.zeros(nS) if V0 is None else V0
    iter = 0
    while True:
        delta = 0.
        for s in range(nS):
            v = V[s]
            V[s] = R[s, policy[s]] + gamma * P[s, policy[s]] @ V
            delta = np.maximum(delta, np.abs(v - V[s]))
        iter += 1
        if delta < tol or iter >= maxiter:
            break
    return V, delta, iter


def Q_policy_eval_deter_async(env, policy, Q0=None, gamma=1., tol=1e-6, maxiter=100):
    nS, nA = env.nS + 1, env.nA
    P, R = env.transition, env.reward
    Q = np.zeros((nS, nA)) if Q0 is None else Q0
    iter = 0
    while True:
        delta = 0.
        for s, a in product(range(nS), range(nA)):
            q = Q[s, a]
            Q[s, a] = R[s, a] + gamma * P[s, a] @ np.choose(policy, Q.T)
            delta = np.maximum(delta, np.abs(q - Q[s, a]))
        iter += 1
        if delta < tol or iter >= maxiter:
            break
    return Q, delta, iter


################################################################################
# Policy improvement: value function -> policy.
################################################################################


def V_policy_impr_deter(env, V, gamma=1.):
    P, R = env.transition, env.reward
    return np.argmax(R + gamma * P @ V, axis=1)


def Q_policy_impr_deter(env, Q):
    return np.argmax(Q, axis=1)


def V_policy_impr_stoch(env, V, gamma=1.):
    policy = V_policy_impr_deter(env, V, gamma)
    return one_hot_encode(policy, env.nA)


def Q_policy_impr_stoch(env, Q):
    policy = Q_policy_impr_deter(env, Q)
    return one_hot_encode(policy, env.nA)


################################################################################
# Policy iteration: alternate between policy evaluation and policy improvement.
################################################################################


def V_policy_iter(env, init=policy_init_deter, eval=V_policy_eval_deter_sync, impr=V_policy_impr_deter, policy0=None, V0=None, gamma=1., tol=1e-6, maxiter=100):
    nS = env.nS + 1
    V = np.zeros(nS) if V0 is None else V0
    policy = init(env) if policy0 is None else policy0
    evaluations = improvements = 0
    while True:
        V, delta, iter = eval(env, policy, V, gamma, tol, maxiter)
        evaluations += iter
        old_policy = policy
        policy = impr(env, V, gamma)
        improvements += 1
        policy_stable = (policy == old_policy).all()
        if policy_stable or (evaluations + improvements) >= maxiter:
            break
    return policy, V, delta, evaluations, improvements


def Q_policy_iter(env, init=policy_init_deter, eval=Q_policy_eval_deter_sync, impr=Q_policy_impr_deter, policy0=None, Q0=None, gamma=1., tol=1e-6, maxiter=100):
    nS, nA = env.nS + 1, env.nA
    Q = np.zeros((nS, nA)) if Q0 is None else Q0
    policy = init(env) if policy0 is None else policy0
    evaluations = improvements = 0
    while True:
        Q, delta, iter = eval(env, policy, Q, gamma, tol, maxiter)
        evaluations += iter
        old_policy = policy
        policy = impr(env, Q)
        improvements += 1
        policy_stable = (policy == old_policy).all()
        if policy_stable or (evaluations + improvements) >= maxiter:
            break
    return policy, Q, delta, evaluations, improvements


################################################################################
# Value update: value function -> value function.
################################################################################


def V_value_update_sync(env, V, gamma=1.):
    P, R = env.transition, env.reward
    v = V
    V = np.max(R + gamma * P @ V, axis=1)
    delta = np.max(np.abs(v - V))
    return V, delta


def Q_value_update_sync(env, Q, gamma=1.):
    P, R = env.transition, env.reward
    q = Q
    Q = R + gamma * P @ np.max(Q, axis=1)
    delta = np.max(np.abs(q - Q))
    return Q, delta


def V_value_update_async(env, V, gamma=1.):
    nS = env.nS + 1
    P, R = env.transition, env.reward
    delta = 0.
    for s in range(nS):
        v = V[s]
        V[s] = np.max(R[s] + gamma * P[s] @ V)
        delta = np.maximum(delta, np.abs(v - V[s]))
    return V, delta


def Q_value_update_async(env, Q, gamma=1.):
    nS, nA = env.nS + 1, env.nA
    P, R = env.transition, env.reward
    delta = 0.
    for s, a in product(range(nS), range(nA)):
        q = Q[s, a]
        Q[s, a] = R[s, a] + gamma * P[s, a] @ np.max(Q, axis=1)
        delta = np.maximum(delta, np.abs(q - Q[s, a]))
    return Q, delta


################################################################################
# Value iteration.
################################################################################


def V_value_iter(env, update=V_value_update_sync, impr=V_policy_impr_deter, V0=None, gamma=1., tol=1e-6, maxiter=100):
    nS = env.nS + 1
    V = np.zeros(nS) if V0 is None else V0
    iter = 0
    while True:
        V, delta = update(env, V, gamma)
        iter += 1
        if delta < tol or iter >= maxiter:
            break
    policy = impr(env, V, gamma)
    return policy, V, delta, iter


def Q_value_iter(env, update=Q_value_update_sync, impr=Q_policy_impr_deter, Q0=None, gamma=1., tol=1e-6, maxiter=100):
    nS, nA = env.nS + 1, env.nA
    Q = np.zeros((nS, nA)) if Q0 is None else Q0
    iter = 0
    while True:
        Q, delta = update(env, Q, gamma)
        iter += 1
        if delta < tol or iter >= maxiter:
            break
    policy = impr(env, Q)
    return policy, Q, delta, iter

