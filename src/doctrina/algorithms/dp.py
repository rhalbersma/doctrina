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


def policy_init_stoch(env):
    return np.full((env.nS, env.nA), 1 / env.nA)


def policy_init_deter(env):
    return np.zeros(env.nS, dtype=int)


# Dispatcher to the various policy initialization algorithms.

def dispatch_policy_init(format):
    return {
        'stoch': policy_init_stoch,
        'deter': policy_init_deter,
    }[format]


def policy_init(env, format='stoch'):
    init = dispatch_policy_init(format)
    return init(env)


################################################################################
# Policy evaluation: policy -> value function.
################################################################################

# Asynchronous / in-place (Gauss-Seidel-style) iterative approach.

def V_policy_eval_stoch_async(env, policy, gamma=1., V0=None, tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    V = np.zeros(env.nS) if V0 is None else V0.copy()
    iter = 0
    while True:
        delta = 0.
        for s in range(env.nS):
            v = V[s]
            V[s] = np.sum(policy[s] * (R[s] + gamma * P[s] @ V))
            delta = np.maximum(delta, np.abs(v - V[s]))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return V, delta, iter


def V_policy_eval_deter_async(env, policy, gamma=1., V0=None, tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    V = np.zeros(env.nS) if V0 is None else V0.copy()
    iter = 0
    while True:
        delta = 0.
        for s in range(env.nS):
            v = V[s]
            V[s] = R[s, policy[s]] + gamma * P[s, policy[s]] @ V
            delta = np.maximum(delta, np.abs(v - V[s]))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return V, delta, iter


def Q_policy_eval_stoch_async(env, policy, gamma=1., Q0=None, tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    Q = np.zeros((env.nS, env.nA)) if Q0 is None else Q0.copy()
    iter = 0
    while True:
        delta = 0.
        for s, a in product(range(env.nS), range(env.nA)):
            q = Q[s, a]
            Q[s, a] = R[s, a] + gamma * P[s, a] @ np.sum(policy * Q, axis=1)
            delta = np.maximum(delta, np.abs(q - Q[s, a]))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return Q, delta, iter


def Q_policy_eval_deter_async(env, policy, gamma=1., Q0=None, tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    Q = np.zeros((env.nS, env.nA)) if Q0 is None else Q0.copy()
    iter = 0
    while True:
        delta = 0.
        for s, a in product(range(env.nS), range(env.nA)):
            q = Q[s, a]
            Q[s, a] = R[s, a] + gamma * P[s, a] @ np.choose(policy, Q.T)
            delta = np.maximum(delta, np.abs(q - Q[s, a]))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return Q, delta, iter


# Synchronous (Jacobi-style) iterative approach.

def V_policy_eval_stoch_sync(env, policy, gamma=1., V0=None, tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    V = np.zeros(env.nS) if V0 is None else V0.copy()
    iter = 0
    while True:
        v = V
        V = np.sum(policy * (R + gamma * P @ V), axis=1)
        delta = np.max(np.abs(v - V))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return V, delta, iter


def V_policy_eval_deter_sync(env, policy, gamma=1., V0=None, tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    V = np.zeros(env.nS) if V0 is None else V0.copy()
    iter = 0
    while True:
        v = V
        V = np.choose(policy, (R + gamma * P @ V).T)
        delta = np.max(np.abs(v - V))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return V, delta, iter


def Q_policy_eval_stoch_sync(env, policy, gamma=1., Q0=None, tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    Q = np.zeros((env.nS, env.nA)) if Q0 is None else Q0.copy()
    iter = 0
    while True:
        q = Q
        Q = R + gamma * P @ np.sum(policy * Q, axis=1)
        delta = np.max(np.abs(q - Q))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return Q, delta, iter


def Q_policy_eval_deter_sync(env, policy, gamma=1., Q0=None, tol=1e-8, maxiter=None):
    P, R = env.transition, env.reward
    Q = np.zeros((env.nS, env.nA)) if Q0 is None else Q0.copy()
    iter = 0
    while True:
        q = Q
        Q = R + gamma * P @ np.choose(policy, Q.T)
        delta = np.max(np.abs(q - Q))
        iter += 1
        if (delta < tol) or (maxiter and iter >= maxiter):
            break
    return Q, delta, iter


# Direct matrix inversion of the Bellman equation.

def V_policy_eval_stoch_solve(env, policy, gamma=1., **kwargs):
    P, R = env.transition, env.reward
    V = np.linalg.solve(np.identity(env.nS) - gamma * np.sum(policy[..., None] * P, axis=1), np.sum(policy * R, axis=1))
    return V, 0., 1


def V_policy_eval_deter_solve(env, policy, gamma=1., **kwargs):
    P, R = env.transition, env.reward
    V = np.linalg.solve(np.identity(env.nS) - gamma * P[np.arange(env.nS), policy], R[np.arange(env.nS), policy])
    return V, 0., 1


def Q_policy_eval_stoch_solve(env, policy, gamma=1., **kwargs):
    P, R = env.transition, env.reward
    M = env.nS * env.nA
    Q = np.linalg.solve(np.identity(M) - gamma * (policy[None, None, ...] * P[..., None]).reshape((M, M)), R.reshape(M)).reshape(env.nS, env.nA)
    return Q, 0., 1


def Q_policy_eval_deter_solve(env, policy, gamma=1., **kwargs):
    return Q_policy_eval_stoch_solve(env, one_hot_encode(policy, env.nA), gamma, **kwargs)


# Dispatchers to the various policy evaluation algorithms.

def dispatch_V_policy_eval(format, method):
    return {
        ('stoch', 'async'): V_policy_eval_stoch_async,
        ('deter', 'async'): V_policy_eval_deter_async,
        ('stoch', 'sync' ): V_policy_eval_stoch_sync,
        ('deter', 'sync' ): V_policy_eval_deter_sync,
        ('stoch', 'solve'): V_policy_eval_stoch_solve,
        ('deter', 'solve'): V_policy_eval_deter_solve,
    }[(format, method)]


def dispatch_Q_policy_eval(format, method):
    return {
        ('stoch', 'async'): Q_policy_eval_stoch_async,
        ('deter', 'async'): Q_policy_eval_deter_async,
        ('stoch', 'sync' ): Q_policy_eval_stoch_sync,
        ('deter', 'sync' ): Q_policy_eval_deter_sync,
        ('stoch', 'solve'): Q_policy_eval_stoch_solve,
        ('deter', 'solve'): Q_policy_eval_deter_solve,
    }[(format, method)]


def V_policy_eval(env, policy, format='stoch', method='async', **kwargs):
    eval = dispatch_V_policy_eval(format, method)
    return eval(env, policy, **kwargs)


def Q_policy_eval(env, policy, format='stoch', method='async', **kwargs):
    eval = dispatch_Q_policy_eval(format, method)
    return eval(env, policy, **kwargs)


################################################################################
# Transformations between state value and state-action value functions.
################################################################################


def Q_from_V(env, V, gamma=1.):
    P, R = env.transition, env.reward
    return R + gamma * P @ V


def V_from_Q_stoch(policy, Q):
    return np.sum(policy * Q, axis=1)


def V_from_Q_deter(policy, Q):
    return np.choose(policy, Q.T)


# Dispatcher for transformations between state value and state-actions value functions.

def dispatch_V_from_Q(format):
    return {
        'stoch': V_from_Q_stoch,
        'deter': V_from_Q_deter,
    }[format]


def V_from_Q(policy, Q, format='stoch'):
    trans = dispatch_V_from_Q(format)
    return trans(policy, Q)


################################################################################
# Policy improvement: value function -> policy.
################################################################################


def Q_policy_impr_stoch(Q):
    policy = np.isclose(Q, Q.max(axis=1, keepdims=True)).astype(float)
    return policy / policy.sum(axis=1, keepdims=True)


def V_policy_impr_stoch(env, V, gamma=1.):
    Q = Q_from_V(env, V, gamma)
    return Q_policy_impr_stoch(Q)


def Q_policy_impr_deter(Q):
    return np.argmax(Q, axis=1)


def V_policy_impr_deter(env, V, gamma=1.):
    Q = Q_from_V(env, V, gamma)
    return Q_policy_impr_deter(Q)


# Dispatchers to the various policy improvement algorithms.

def dispatch_V_policy_impr(format):
    return {
        'deter': V_policy_impr_deter,
        'stoch': V_policy_impr_stoch,
    }[format]


def dispatch_Q_policy_impr(format):
    return {
        'deter': Q_policy_impr_deter,
        'stoch': Q_policy_impr_stoch,
    }[format]


def V_policy_impr(env, V, gamma=1., format='stoch'):
    impr = dispatch_V_policy_impr(format)
    return impr(env, V, gamma)


def Q_policy_impr(env, Q, format='stoch'):
    impr = dispatch_Q_policy_impr(format)
    return impr(env, Q)


################################################################################
# Policy comparison.
################################################################################


def Q_policy_cmp(policy0, policy1, Q, format='stoch'):
    value = dispatch_V_from_Q(format)    
    return np.isclose(value(policy0, Q), value(policy1, Q)).all()


def V_policy_cmp(policy0, policy1, env, V, gamma=1., format='stoch'):
    Q = Q_from_V(env, V, gamma)
    return Q_policy_cmp(policy0, policy1, Q, format)


################################################################################
# Policy iteration: alternate between policy evaluation and policy improvement.
################################################################################


def V_policy_iter(env, gamma=1., format='stoch', method='async', policy0=None, V0=None, tol=1e-8, maxiter=None, disp=False):
    init = dispatch_policy_init(format)
    impr = dispatch_V_policy_impr(format)
    eval = dispatch_V_policy_eval(format, method)
    V = np.zeros(env.nS) if V0 is None else V0.copy()
    policy = init(env) if policy0 is None else policy0.copy()
    evaluations = improvements = 0
    while True:
        V, delta, iter = eval(env, policy, gamma, V0=V, tol=tol, maxiter=maxiter)
        evaluations += iter
        old_policy = policy
        policy = impr(env, V, gamma)
        if disp:
            print(f'iter: {iter}, delta: {delta}')
        improvements += 1
        if delta < tol and V_policy_cmp(old_policy, policy, env, V, gamma, format):
            break
    return policy, V, { 'delta': delta, 'evaluations': evaluations, 'improvements': improvements }


def Q_policy_iter(env, gamma=1., format='stoch', method='async', policy0=None, Q0=None, tol=1e-8, maxiter=None, disp=False):
    init = dispatch_policy_init(format)
    impr = dispatch_Q_policy_impr(format)
    eval = dispatch_Q_policy_eval(format, method)
    Q = np.zeros((env.nS, env.nA)) if Q0 is None else Q0.copy()
    policy = init(env) if policy0 is None else policy0.copy()
    evaluations = improvements = 0
    while True:
        Q, delta, iter = eval(env, policy, gamma, Q0=Q, tol=tol, maxiter=maxiter)
        evaluations += iter
        old_policy = policy
        policy = impr(Q)
        if disp:
            print(f'iter: {iter}, delta: {delta}')
        improvements += 1
        if delta < tol and Q_policy_cmp(old_policy, policy, Q, format):
            break
    return policy, Q, { 'delta': delta, 'evaluations': evaluations, 'improvements': improvements }


################################################################################
# Value update: value function -> value function.
################################################################################


def V_value_update_async(env, V, gamma=1.):
    P, R = env.transition, env.reward
    delta = 0.
    for s in range(env.nS):
        v = V[s]
        V[s] = np.max(R[s] + gamma * P[s] @ V)
        delta = np.maximum(delta, np.abs(v - V[s]))
    return V, delta


def Q_value_update_async(env, Q, gamma=1.):
    P, R = env.transition, env.reward
    delta = 0.
    for s, a in product(range(env.nS), range(env.nA)):
        q = Q[s, a]
        Q[s, a] = R[s, a] + gamma * P[s, a] @ np.max(Q, axis=1)
        delta = np.maximum(delta, np.abs(q - Q[s, a]))
    return Q, delta


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


# Dispatchers to the various update algorithms.

def dispatch_V_value_update(method):
    return {
        'async': V_value_update_async,
        'sync' : V_value_update_sync,
    }[method]


def dispatch_Q_value_update(method):
    return {
        'async': Q_value_update_async,
        'sync' : Q_value_update_sync,
    }[method]


def V_value_update(env, V, gamma=1., method='async'):
    update = dispatch_V_value_update(method)
    return update(env, V, gamma)


def Q_value_update(env, Q, method='async'):
    update = dispatch_Q_value_update(method)
    return update(env, Q)


################################################################################
# Value iteration.
################################################################################


def V_value_iter(env, gamma=1., format='stoch', method='async', V0=None, tol=1e-8, disp=False):
    update = dispatch_V_value_update(method)
    impr   = dispatch_V_policy_impr(format)
    V = np.zeros(env.nS) if V0 is None else V0.copy()
    iter = 0
    while True:
        V, delta = update(env, V, gamma)
        if disp:
            print(f'iter: {iter}, delta: {delta}')
        iter += 1
        if delta < tol:
            break
    policy = impr(env, V, gamma)
    return policy, V, { 'delta': delta, 'iter': iter }


def Q_value_iter(env, gamma=1., format='stoch', method='async', Q0=None, tol=1e-8, disp=False):
    update = dispatch_Q_value_update(method)
    impr   = dispatch_Q_policy_impr(format)
    Q = np.zeros((env.nS, env.nA)) if Q0 is None else Q0.copy()
    iter = 0
    while True:
        Q, delta = update(env, Q, gamma)
        if disp:
            print(f'iter: {iter}, delta: {delta}')
        iter += 1
        if delta < tol:
            break
    policy = impr(Q)
    return policy, Q, { 'delta': delta, 'iter': iter }

