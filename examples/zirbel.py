#          Copyright Rein Halbersma 2020.
# Distributed under the Boost Software License, Version 1.0.
#    (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import pandas as pd
from toposort import toposort, toposort_flatten

import gym
import gym_blackjack_v1 as bj

env = gym.make('Blackjack-v1', winning_blackjack=1.5)

###############################################################################
# Dynamic programming using Markov Chains
# https://www.dropbox.com/s/xrntclqyx36jhis/Blackjack_talk_2001.pdf
###############################################################################

def format_fsm(fsm):
    format = pd.DataFrame(fsm, index=bj.player_labels, columns=bj.card_labels)
    for player in range(format.shape[0]):
        for card in range(format.shape[1]):
            format.iloc[player, card] = bj.player_labels[fsm[player, card]]
    return format

# Card probabilities
prob = np.ones((len(bj.Card))) / 13.
prob[bj.Card._T] *= 4.  # 10, J, Q, K all count as T

def graph_from_fsm(fsm):
    """
    Convert a finite-state machine into a graph.

    Args:
        fsm (a NumPy array): a finite-state machine.
        
    Returns:
        graph (dict of sets): a graph in adjacency list format.

    Notes:
        The return value format is suitable to be parsed by the toposort library.
    """
    return { 
        state: set(successors) 
        for state, successors in enumerate(fsm) 
    }

def trans_from_fsm(fsm):
    prob_p_p = np.zeros((len(bj.Player), len(bj.Player)))
    for p0, successors in enumerate(fsm):
        for card, p1 in enumerate(successors):
            prob_p_p[p0, p1] += prob[card]
    return prob_p_p

def trans_from_fsm_array(fsm):
    prob_p_a_p = np.zeros((len(bj.Player), len(bj.Action), len(bj.Player)))
    for a in bj.Action:
        prob_p_a_p[:, a, :] = trans_from_fsm(fsm[a])
    return prob_p_a_p

def trans_from_upcard(fsm):
    prob_uc_p = np.zeros((len(bj.Card), len(bj.Hand)))
    for card in range(len(bj.Card)):
        player = fsm[bj.Player.DEAL, card]
        prob_uc_p[card, player] = 1.
    return prob_uc_p

def one_hot_policy(policy):
    one_hot = np.zeros((policy.size, policy.max() + 1), dtype=int)
    one_hot[np.arange(policy.size), policy] = 1
    return one_hot

fsm = np.array([ a for a in np.broadcast_arrays(bj.fsm_stand.reshape(-1,1), bj.fsm_hit) ])

format_fsm(fsm[bj.Action.s])
format_fsm(fsm[bj.Action.h])

pd.DataFrame(env.payout, index=bj.count_labels, columns=bj.count_labels)

prob_p_a_p = trans_from_fsm_array(fsm)
assert np.isclose(prob_p_a_p.sum(axis=2), 1.).all()

id_t = np.identity(len(bj.Player) - len(bj.Terminal))

P_stand = prob_p_a_p[:, bj.Action.s, :]
Q_stand, R_stand = P_stand[:-len(bj.Terminal), :-len(bj.Terminal)], P_stand[:-len(bj.Terminal), -len(bj.Terminal):]
N_stand = np.linalg.inv(id_t - Q_stand)
B_stand = N_stand @ R_stand
pd.DataFrame(P_stand, index=bj.player_labels, columns=bj.player_labels).round(1)
pd.DataFrame(Q_stand, index=bj.player_labels[:-len(bj.Terminal)], columns=bj.player_labels[:-len(bj.Terminal)]).round(2)
pd.DataFrame(R_stand, index=bj.player_labels[:-len(bj.Terminal)], columns=bj.player_labels[-len(bj.Terminal):]).round(2)
pd.DataFrame(N_stand, index=bj.player_labels[:-len(bj.Terminal)], columns=bj.player_labels[:-len(bj.Terminal)]).round(2)
pd.DataFrame(B_stand, index=bj.player_labels[:-len(bj.Terminal)], columns=bj.player_labels[-len(bj.Terminal):]).round(2)

P_hit = prob_p_a_p[:, bj.Action.h, :]
Q_hit, R_hit = P_hit[:-len(bj.Terminal), :-len(bj.Terminal)], P_hit[:-len(bj.Terminal), -len(bj.Terminal):]
N_hit = np.linalg.inv(id_t - Q_hit)
B_hit = N_hit @ R_hit
pd.DataFrame(P_hit, index=bj.player_labels, columns=bj.player_labels).round(1)
pd.DataFrame(Q_hit, index=bj.player_labels[:-len(bj.Terminal)], columns=bj.player_labels[:-len(bj.Terminal)]).round(2)
pd.DataFrame(R_hit, index=bj.player_labels[:-len(bj.Terminal)], columns=bj.player_labels[-len(bj.Terminal):]).round(2)
pd.DataFrame(N_hit, index=bj.player_labels[:-len(bj.Terminal)], columns=bj.player_labels[:-len(bj.Terminal)]).round(2)
pd.DataFrame(B_hit, index=bj.player_labels[:-len(bj.Terminal)], columns=bj.player_labels[-len(bj.Terminal):]).round(2)

pol_dealer = one_hot_policy(np.resize(env.dealer_policy, len(bj.Player)))
fsm_dealer = (fsm * np.expand_dims(pol_dealer, axis=0).T).sum(axis=0)
P_dealer = trans_from_fsm(fsm_dealer)
Q_dealer, R_dealer = P_dealer[:-len(bj.Terminal), :-len(bj.Terminal)], P_dealer[:-len(bj.Terminal), -len(bj.Terminal):]
N_dealer = np.linalg.inv(id_t - Q_dealer)
B_dealer = N_dealer @ R_dealer
pd.DataFrame(P_dealer, index=bj.player_labels, columns=bj.player_labels).round(1)
pd.DataFrame(Q_dealer, index=bj.player_labels[:-len(bj.Terminal)], columns=bj.player_labels[:-len(bj.Terminal)]).round(2)
pd.DataFrame(R_dealer, index=bj.player_labels[:-len(bj.Terminal)], columns=bj.player_labels[-len(bj.Terminal):]).round(2)
pd.DataFrame(N_dealer, index=bj.player_labels[:-len(bj.Terminal)], columns=bj.player_labels[:-len(bj.Terminal)]).round(2)
pd.DataFrame(B_dealer, index=bj.player_labels[:-len(bj.Terminal)], columns=bj.player_labels[-len(bj.Terminal):]).round(2)

prob_uc_h = trans_from_upcard(fsm[bj.Action.h])
pd.DataFrame(prob_uc_h, index=bj.card_labels, columns=bj.hand_labels)

prob_uc_c = prob_uc_h @ N_dealer[:len(bj.Hand), len(bj.Hand):]
pd.DataFrame(prob_uc_c, index=bj.card_labels, columns=bj.count_labels).round(4)

prob_c = prob.T @ prob_uc_c
pd.DataFrame(prob_c, index=bj.count_labels, columns=['prob_c'])

prob_h_s_c = prob_p_a_p[:len(bj.Hand), bj.Action.s, len(bj.Hand):-len(bj.Terminal)]
pd.DataFrame(prob_h_s_c, index=bj.hand_labels, columns=bj.count_labels)

reward_c_uc = env.payout @ prob_uc_c.T
pd.DataFrame(reward_c_uc, index=bj.count_labels, columns=bj.card_labels).round(4)

reward_h_s_uc = prob_h_s_c @ reward_c_uc
pd.DataFrame(reward_h_s_uc, index=bj.hand_labels, columns=bj.card_labels).round(4)

reward_values = np.unique(env.payout)
reward_outcomes = np.array([
    env.payout == r
    for r in reward_values
], dtype=int)
prob_c_uc_r = (reward_outcomes @ prob_uc_c.T).transpose(1, 2, 0)
for r, _ in  enumerate(reward_values):
    pd.DataFrame(prob_c_uc_r[..., r], index=bj.count_labels, columns=bj.card_labels).round(4) 

prob_s_a_r_s = np.zeros((len(bj.Player), len(bj.Dealer), len(bj.Action), len(reward_values), len(bj.Player), len(bj.Dealer)))

for c in range(len(bj.Card)):
    prob_s_a_r_s[:len(bj.Hand), c, :, 1, :-len(bj.Terminal), c] = prob_p_a_p[:len(bj.Hand), :, :-len(bj.Terminal)]
    
for a in range(len(bj.Action)):
    prob_s_a_r_s[len(bj.Hand):-len(bj.Terminal), :-len(bj.Terminal), a, :, bj.Player._END, bj.Dealer._END] = (prob_p_a_p[len(bj.Hand):-len(bj.Terminal), a, bj.Player._END] * prob_c_uc_r.T).T

prob_s_a_r_s[bj.Player._END, bj.Dealer._END, :, 1, bj.Player._END, bj.Dealer._END] = prob_p_a_p[bj.Player._END, :, bj.Player._END]

assert np.isclose(prob_s_a_r_s[..., -len(bj.Terminal):, :-len(bj.Terminal)], 0.).all()
assert np.isclose(prob_s_a_r_s[..., :-len(bj.Terminal), -len(bj.Terminal):], 0.).all()

prob_s_a_r_s[bj.Player._END, :-len(bj.Terminal), :, 1, bj.Player._END, bj.Dealer._END] = 1.
prob_s_a_r_s[:-len(bj.Terminal), bj.Dealer._END, :, 1, bj.Player._END, bj.Dealer._END] = 1.

assert np.isclose(prob_s_a_r_s.sum(axis=(3, 4, 5)), 1.).all()

prob_s_a_s = prob_s_a_r_s.sum(axis=3)
assert np.isclose(prob_s_a_s.sum(axis=(3, 4)), 1.).all()

reward_s_a = prob_s_a_r_s.sum(axis=(4, 5)) @ reward_values
pd.DataFrame(reward_s_a[..., bj.Action.s], index=bj.player_labels, columns=bj.dealer_labels)
pd.DataFrame(reward_s_a[..., bj.Action.h], index=bj.player_labels, columns=bj.dealer_labels)
