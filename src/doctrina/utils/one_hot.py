#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np


def one_hot_encode(policy_deter, size=None):
    size = policy_deter.max() + 1 if size is None else size
    policy_stoch = np.zeros((policy_deter.size, size))
    policy_stoch[np.arange(policy_deter.size), policy_deter] = 1
    return policy_stoch

