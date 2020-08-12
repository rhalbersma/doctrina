#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from doctrina.algorithms import dp, mc
from doctrina import spaces
import gym_blackjack_v1 as bj

env = gym.make('Blackjack-v1', model=True)
SEED = 47110815
env.seed(SEED)          # reproducible environment

###############################################################################
# Example 5.1: Blackjack
###############################################################################

# Consider the policy that sticks if the player’s sum is 20 or 21, and otherwise hits.
stand_on_20 = np.full(spaces.shape(env.observation_space), bj.Action.h)
stand_on_20[bj.Hand.H20:(bj.Hand.H21 + 1), :] = bj.Action.s
stand_on_20[bj.Hand.S20:(bj.Hand.BJ  + 1), :] = bj.Action.s

# In any event, after 500,000 games the value function is very well approximated.
runs = [ 10000, 500000 ]
V, N = zip(*[ 
    mc.predict_ev(env, episodes, stand_on_20) 
    for episodes in runs 
])

options = {
    'vmin': min(env.reward_range), 
    'vmax': max(env.reward_range), 
    'cmap': sns.color_palette('coolwarm'), 
    'center': 0.,
    'annot': True, 
    'xticklabels': bj.card_labels
}

hands = [
    np.arange(bj.Hand.S12, bj.Hand.BJ  + 1),
    np.arange(bj.Hand.H12, bj.Hand.H21 + 1)
]

yticklabels = [ 
    np.array(bj.hand_labels)[hands[no_usable_ace]] 
    for no_usable_ace in range(2)
]

axopts = {
    'xlabel': 'Dealer showing',
    'ylabel': 'Player sum'
}

fig, axes = plt.subplots(nrows=2, ncols=len(runs))
fig.suptitle(
    """
    Figure 5.1: Approximate state-value functions for the blackjack policy that
    sticks only on 20 or 21, computed by Monte Carlo policy evaluation.
    """
)
rows = [ 'Usable ace', 'No usable ace']
cols = [ f'After {episodes:,} episodes' for episodes in runs ]

# https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
pad = 5 # in points

for ax, row in zip(axes[:,0], rows):
    ax.annotate(
        row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
        xycoords=ax.yaxis.label, textcoords='offset points',
        size='large', ha='right', va='center', rotation=90
    )

for ax, col in zip(axes[0,:], cols):
    ax.annotate(
        col, xy=(0.5, 1), xytext=(0, pad),
        xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline'
    )

for i, h in enumerate(hands):
    for r, _ in enumerate(runs):
        sns.heatmap(V[r][h, :], yticklabels=yticklabels[i], ax=axes[i, r], **options).set(**axopts)
plt.show()

###############################################################################
# Example 5.3: Solving Blackjack
###############################################################################

episodes = 10**6
policy0 = stand_on_20
policy, Q, N = mc.control_es(env, episodes, policy0)
#policy, Q, N = mc.control_ev_eps(env, episodes, policy0)
#policy, Q, N = mc.control_ev_ucb(env, episodes, policy0)

#assert (policy == Q.argmax(axis=2)).all()
policy = Q.argmax(axis=2)
V = Q.max(axis=2)
A = Q[:,:,bj.Action.h] - Q[:,:,bj.Action.h]

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.suptitle(
    """
    Figure 5.2: The optimal policy and state-value function for blackjack, found by Monte Carlo ES. 
    The state-value function shown was computed from the action-value function found by Monte Carlo ES.
    """
)
rows = [ 'Usable ace', 'No usable ace']
cols = [ 'Optimal policy', 'Optimal state-value function' ]

# https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
pad = 5 # in points

for ax, row in zip(axes[:,0], rows):
    ax.annotate(
        row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
        xycoords=ax.yaxis.label, textcoords='offset points',
        size='large', ha='right', va='center', rotation=90
    )

for ax, col in zip(axes[0,:], cols):
    ax.annotate(
        col, xy=(0.5, 1), xytext=(0, pad),
        xycoords='axes fraction', textcoords='offset points',
        size='large', ha='center', va='baseline'
    )

for i, h in enumerate(hands):
    sns.heatmap(policy[h, :], yticklabels=yticklabels[i], ax=axes[i, 0], **options).set(**axopts)
    sns.heatmap(     V[h, :], yticklabels=yticklabels[i], ax=axes[i, 1], **options).set(**axopts)
plt.show()

###############################################################################
# Example 5.4
###############################################################################

# We evaluated the state in which the dealer is showing a deuce, 
# the sum of the player’s cards is 13, and the player has a usable ace 
# (that is, the player holds an ace and a deuce, or equivalently three aces).
start = (bj.Hand.S13, bj.Card._2)

# The target policy was to stick only on a sum of 20 or 21, as in Example 5.1.
target_policy = stand_on_20

# The value of this state under the target policy is approximately −0.27726 
# (this was determined by separately generating one-hundred million episodes 
# using the target policy and averaging their returns).
episodes = 10**6
V, N = mc.predict_ev(env, episodes, stand_on_20, start)
assert np.isclose(V[start], -0.27726, rtol=1e-2)
assert N[start] == episodes

target_policyM = np.zeros(spaces.shape(env.state_space), dtype=int)
target_policyM[:len(bj.Hand), :len(bj.Card)] = target_policy
target_policyM = target_policyM.reshape(spaces.size(env.state_space))
V, delta, iter = dp.V_policy_eval_deter_sync(env, target_policyM, tol=1e-9)
V.reshape(spaces.shape(env.state_space))[start]
