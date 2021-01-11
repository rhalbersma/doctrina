#          Copyright Rein Halbersma 2020-2021.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import gym


def register(id, entry_point, force=True):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
    gym.register(
        id=id,
        entry_point=entry_point,
    )


environments = [
    ['multi_armed_bandit', 'MultiArmedBandit', 'v0'],
    ['jacks_car_rental',   'JacksCarRental',   'v1']
]

for file, name, version in environments:
    register(
        id=f'{name}-{version}',
        entry_point=f'doctrina.envs.{file}:{name}Env',
    )
