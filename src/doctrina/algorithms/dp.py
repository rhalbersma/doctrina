#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

def value_iteration(env, theta=1e-6, gamma=1.):
    V = np.zeros(spaces.shape(env.observation_space))
    delta = 0.
    while True:
        v = V
        Q = env.reward + gamma * np.tensordot(env.transition, V, axes=2)
        V = np.max(Q, axis=1)
        delta = np.max(delta, np.abs(v - V))
        if delta < theta:
            break
    policy = np.argmax(Q, axis=1)
    return policy, V
