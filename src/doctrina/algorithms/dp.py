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
    return np.zeros(env.nSp, dtype=int)


def policy_init_stoch(env):
    return np.full((env.nSp, env.nA), 1 / env.nA)


################################################################################
# Transforming from state value to state-action value functions.
################################################################################


def Q_from_V_sync(env, V, gamma=1.):
    P, R = env.transition, env.reward
    return R + gamma * P @ V


def Q_from_V_async(env, V, s, gamma=1.):
    P, R = env.transition, env.reward
    return R[s] + gamma * P[s] @ V


################################################################################
# Policy evaluation: policy -> value function.
################################################################################


def V_policy_eval_stoch_sync(env, policy, V0=None, gamma=1., tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    V = np.zeros(env.nSp) if V0 is None else V0
    iter = 0
    while True:
        v = V
        V = np.sum(policy * (R + gamma * P @ V), axis=1)
        delta = np.max(np.abs(v - V))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return V, delta, iter


def Q_policy_eval_stoch_sync(env, policy, Q0=None, gamma=1., tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    Q = np.zeros((env.nSp, env.nA)) if Q0 is None else Q0
    iter = 0
    while True:
        q = Q
        Q = R + gamma * P @ np.sum(policy * Q, axis=1)
        delta = np.max(np.abs(q - Q))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return Q, delta, iter


def V_policy_eval_stoch_async(env, policy, V0=None, gamma=1., tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    V = np.zeros(env.nSp) if V0 is None else V0
    iter = 0
    while True:
        delta = 0.
        for s in range(env.nSp):
            v = V[s]
            V[s] = np.sum(policy[s] * (R[s] + gamma * P[s] @ V))
            delta = np.maximum(delta, np.abs(v - V[s]))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return V, delta, iter


def Q_policy_eval_stoch_async(env, policy, Q0=None, gamma=1., tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    Q = np.zeros((env.nSp, env.nA)) if Q0 is None else Q0
    iter = 0
    while True:
        delta = 0.
        for s, a in product(range(env.nSp), range(env.nA)):
            q = Q[s, a]
            Q[s, a] = R[s, a] + gamma * P[s, a] @ np.sum(policy * Q, axis=1)
            delta = np.maximum(delta, np.abs(q - Q[s, a]))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return Q, delta, iter


def V_policy_eval_deter_sync(env, policy, V0=None, gamma=1., tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    V = np.zeros(env.nSp) if V0 is None else V0
    iter = 0
    while True:
        v = V
        V = np.choose(policy, (R + gamma * P @ V).T)
        delta = np.max(np.abs(v - V))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return V, delta, iter


def Q_policy_eval_deter_sync(env, policy, Q0=None, gamma=1., tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    Q = np.zeros((env.nSp, env.nA)) if Q0 is None else Q0
    iter = 0
    while True:
        q = Q
        Q = R + gamma * P @ np.choose(policy, Q.T)
        delta = np.max(np.abs(q - Q))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return Q, delta, iter


def V_policy_eval_deter_async(env, policy, V0=None, gamma=1., tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    V = np.zeros(env.nSp) if V0 is None else V0
    iter = 0
    while True:
        delta = 0.
        for s in range(env.nSp):
            v = V[s]
            V[s] = R[s, policy[s]] + gamma * P[s, policy[s]] @ V
            delta = np.maximum(delta, np.abs(v - V[s]))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return V, delta, iter


def Q_policy_eval_deter_async(env, policy, Q0=None, gamma=1., tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    Q = np.zeros((env.nSp, env.nA)) if Q0 is None else Q0
    iter = 0
    while True:
        delta = 0.
        for s, a in product(range(env.nSp), range(env.nA)):
            q = Q[s, a]
            Q[s, a] = R[s, a] + gamma * P[s, a] @ np.choose(policy, Q.T)
            delta = np.maximum(delta, np.abs(q - Q[s, a]))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return Q, delta, iter


################################################################################
# Policy improvement: value function -> policy.
################################################################################


def Q_policy_impr_deter(Q):
    return np.argmax(Q, axis=1)


def V_policy_impr_deter(env, V, gamma=1.):
    Q = Q_from_V_sync(env, V, gamma)
    return Q_policy_impr_deter(Q)


def Q_policy_impr_stoch(Q):
    policy = np.isclose(Q, Q.max(axis=1, keepdims=True)).astype(float)
    return policy / policy.sum(axis=1, keepdims=True)


def V_policy_impr_stoch(env, V, gamma=1.):
    Q = Q_from_V_sync(env, V, gamma)
    return Q_policy_impr_stoch(Q)


################################################################################
# Policy iteration: alternate between policy evaluation and policy improvement.
################################################################################


def V_policy_iter(env, stoch=False, sync=True, policy0=None, V0=None, gamma=1., tol=1e-8, maxiter=None):
    if stoch:
        init =   policy_init_stoch
        eval = V_policy_eval_stoch_sync if sync else V_policy_eval_stoch_async
        impr = V_policy_impr_stoch
    else:
        init =   policy_init_deter
        eval = V_policy_eval_deter_sync if sync else V_policy_eval_deter_async
        impr = V_policy_impr_deter
    V = np.zeros(env.nSp) if V0 is None else V0
    policy = init(env) if policy0 is None else policy0
    evaluations = improvements = 0
    while True:
        V, delta, iter = eval(env, policy, V, gamma, tol, maxiter)
        evaluations += iter
        old_policy = policy
        policy = impr(env, V, gamma)
        improvements += 1
        if delta < tol and (policy == old_policy).all():
            break
    return policy, V, { 'delta': delta, 'evaluations': evaluations, 'improvements': improvements }


def Q_policy_iter(env, stoch=False, sync=True, policy0=None, Q0=None, gamma=1., tol=1e-8, maxiter=None):
    if stoch:
        init =   policy_init_stoch
        eval = Q_policy_eval_stoch_sync if sync else Q_policy_eval_stoch_async
        impr = Q_policy_impr_stoch
    else:
        init =   policy_init_deter
        eval = Q_policy_eval_deter_sync if sync else Q_policy_eval_deter_async
        impr = Q_policy_impr_deter
    Q = np.zeros((env.nSp, env.nA)) if Q0 is None else Q0
    policy = init(env) if policy0 is None else policy0
    evaluations = improvements = 0
    while True:
        Q, delta, iter = eval(env, policy, Q, gamma, tol, maxiter)
        evaluations += iter
        old_policy = policy
        policy = impr(Q)
        improvements += 1
        if delta < tol and (policy == old_policy).all():
            break
    return policy, Q, { 'delta': delta, 'evaluations': evaluations, 'improvements': improvements }


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
    P, R = env.transition, env.reward
    delta = 0.
    for s in range(env.nSp):
        v = V[s]
        V[s] = np.max(R[s] + gamma * P[s] @ V)
        delta = np.maximum(delta, np.abs(v - V[s]))
    return V, delta


def Q_value_update_async(env, Q, gamma=1.):
    P, R = env.transition, env.reward
    delta = 0.
    for s, a in product(range(env.nSp), range(env.nA)):
        q = Q[s, a]
        Q[s, a] = R[s, a] + gamma * P[s, a] @ np.max(Q, axis=1)
        delta = np.maximum(delta, np.abs(q - Q[s, a]))
    return Q, delta


################################################################################
# Value iteration.
################################################################################


def V_value_iter(env, stoch=False, sync=True, V0=None, gamma=1., tol=1e-8):
    update = V_value_update_sync if sync  else V_value_update_async
    impr   = V_policy_impr_stoch if stoch else V_policy_impr_deter
    V = np.zeros(env.nSp) if V0 is None else V0
    iter = 0
    while True:
        V, delta = update(env, V, gamma)
        iter += 1
        if delta < tol:
            break
    policy = impr(env, V, gamma)
    return policy, V, { 'delta': delta, 'iter': iter }


def Q_value_iter(env, stoch=False, sync=True, Q0=None, gamma=1., tol=1e-8):
    update = Q_value_update_sync if sync  else Q_value_update_async
    impr   = Q_policy_impr_stoch if stoch else Q_policy_impr_deter
    Q = np.zeros((env.nSp, env.nA)) if Q0 is None else Q0
    iter = 0
    while True:
        Q, delta = update(env, Q, gamma)
        iter += 1
        if delta < tol:
            break
    policy = impr(Q)
    return policy, Q, { 'delta': delta, 'iter': iter }

