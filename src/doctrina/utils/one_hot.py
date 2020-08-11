#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np

def one_hot_encode(arr, size):
    one_hot = np.zeros((arr.size, size))
    one_hot[np.arange(arr.size), arr] = 1.
    return one_hot
