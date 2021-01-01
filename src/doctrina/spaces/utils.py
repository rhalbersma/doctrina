#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import functools
import itertools
import operator

import gym
from gym import spaces


def shape(space):
    if isinstance(space, gym.spaces.tuple.Tuple):
        return tuple(itertools.chain.from_iterable(map(shape, space)))
    elif isinstance(space, gym.spaces.discrete.Discrete):
        return (space.n,)


def size(space):
    return functools.reduce(operator.mul, shape(space))


def state_table(table, env):
    return table.reshape(shape(env.observation_space))

