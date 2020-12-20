#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import gym
import pandas as pd

import gym_blackjack_v1 as bj
from doctrina import spaces
from doctrina.algorithms import dp

# Dynamic programming using Markov Chains or How to play Blackjack
# by Craig L. Zirbel (2001)
# https://www.dropbox.com/s/xrntclqyx36jhis/Blackjack_talk_2001.pdf

# The payoff (p. 3)
env = gym.make('Blackjack-v1', winning_blackjack=+1.5, model_based=True)

# Value iteration (p. 12)
V, policy, delta, iter = dp.V_value_iter(env)

# The value matrix (p. 13).
pd.DataFrame(
    V.reshape(spaces.shape(env.state_space))[1:len(bj.Hand), :len(bj.Card)], 
    index=bj.hand_labels[1:], 
    columns=bj.card_labels
).round(4)

# The optimal stopping strategy (p.14).
pd.DataFrame(
    policy.reshape(spaces.shape(env.state_space))[1:len(bj.Hand), :len(bj.Card)], 
    index=bj.hand_labels[1:], 
    columns=bj.card_labels
).applymap(lambda a: bj.action_labels[a].upper()).replace({'S': ' '})
