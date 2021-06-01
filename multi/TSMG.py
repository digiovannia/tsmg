import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pymc3 as pm
import logging
import itertools
from math import ceil
import theano.tensor as tt
import pandas as pd
import os

np.random.seed(1)
draws_dict = {'2': 2000,
              'p': 3000,
              'b': 3000,
              'g': 2000}#800}

# Parameters for the reward function of the grid world and other invariant
# parameters
MAX_REW = 1 
NEUTRAL = 0.7
COLLISION = 0
T = 100000
D = 3
NUM_CORES = 4
SEED_NUMBER = 3

'''
Inputs for experiments
'''
repl_2 = input('Replicating Fig 2, nonpara? (0/1, default 0): ')
REPL_2 = bool(int(repl_2)) if repl_2 else False
repl_2p = False
repl_4 = False
repl_4a = False
if not REPL_2:
    repl_2p = input('Replicating Fig 2, parametric? (0/1, default 0): ')
REPL_2P = bool(int(repl_2p)) if repl_2p else False
if not (REPL_2 or REPL_2P):
    repl_4 = input('Replicating Fig 4, default prior? (0/1, default 0): ')
REPL_4 = bool(int(repl_4)) if repl_4 else False
if not (REPL_2 or REPL_2P or REPL_4):
    repl_4a = input('Replicating Fig 4, alt prior? (0/1, default 0): ')
REPL_4A = bool(int(repl_4a)) if repl_4a else False
if REPL_2 or REPL_2P or REPL_4 or REPL_4A:
    SELF_PLAY_TEST = False
    THRESHOLD = 1.5
    game_id = input('Game to be tested (2/p/b/g): ')
    N = 30
    verbose = False
    MET = True
    ndraws = draws_dict[game_id]
    nburn = ndraws // 4
    L = 2500
    UCRL = False
    THETA_DIM = 4
    THRESHOLD_LIST = [1.5]
    CT_SEEDS = 10
    ALPHA_VEC = np.array([0.5, 0.5, 0.5, 0.5])
    mus = '0,0,0,0'
    MUS = np.zeros(4)
    ELLS = [6]
    PARAMETRIC = False
    MULTIPLE = (0.375 if game_id == 'g' else 3)
    if REPL_2:
        THRESHOLD_LIST = [1.5, 3, 4]
    elif REPL_2P:
        PARAMETRIC = True
    else:
        SELF_PLAY_TEST = True
        N = 20
        CT_SEEDS = 1
        if REPL_4A:
            if game_id == '2':
                ALPHA_VEC = np.array([2, 0.5, 0.5, 0.5])
            elif game_id == 'g':
                mus = '0,2,0,0'
                MUS = np.array([0, 2, 0, 0])
            else:
                ALPHA_VEC = np.array([0.5, 0.5, 2, 0.5])
else:
    selfplaytest = input('Self-play? (0/1, default 0): ')
    SELF_PLAY_TEST = bool(int(selfplaytest)) if selfplaytest else False
    thresh = input('THRESHOLD (default 1.5) = ')
    THRESHOLD = float(thresh) if thresh else 1.5
    game_id = input('Game to be tested (2/p/b/g): ')
    nval = input('Number of trials to run (default 30, or 20 if self-play): ')
    N = int(nval) if nval else (20 if SELF_PLAY_TEST else 30)
    vb = input('Print output? (1/0, default 0): ')
    verbose = bool(int(vb)) if vb else False
    metrop = input('Use Metropolis? (1/0, default 1): ')
    MET = bool(int(metrop)) if metrop else True
    ndraws = input('# draws? (default 2000/3000/3000/800 for 2/p/b/g): ')
    ndraws = int(ndraws) if ndraws else draws_dict[game_id]
    nburn = ndraws // 4
    epochlength = input('Min epoch length (default 2500) = ')
    L = int(epochlength) if epochlength else 2500
    if THRESHOLD > 2:
        uc = input('Passive resets? (1/0, default 0): ')
        UCRL = bool(int(uc)) if uc else False
    else:
        UCRL = False
    tdim = input('Dimension of theta (default 4) = ')
    THETA_DIM = int(tdim) if tdim else 4
    ctseeds = input('How many seed values for tests? (default 10) ')
    CT_SEEDS = int(ctseeds) if ctseeds else 10
    tl = input('Thresholds (default 1.5,3,4), use 4 for passive: ')
    THRESHOLD_LIST = [float(n) for n in tl.split(',')] if tl else [1.5, 3, 4]
    alp = input('Alpha vector for dirichlet (default 0.5,0.5,0.5,0.5): ')
    if alp:
        ALPHA_VEC = np.array([float(n) for n in alp.split(',')])
    else:
        ALPHA_VEC = np.array([0.5, 0.5, 0.5, 0.5])
    mus = input('Mu vector for lognormal (default 0,0,0,0): ')
    MUS = np.array([float(n) for n in mus.split(',')]) if mus else np.zeros(4)
    ells = input('Values of ell for tests (default 6): ')
    ELLS = [int(n) for n in ells.split(',')] if ells else [6]
    par = input('Parametric CD? (0/1, default 0) ')
    PARAMETRIC = bool(int(par)) if par else False
    multiple = input('Multiple for para CD? (default 0.375 if g, else 3) ')
    MULTIPLE = float(multiple) if multiple else (0.375 if game_id == 'g' else 3)


#%%

'''
Auxiliary functions
'''

def gen_grid(D):
    '''
    Repeated grid world, replicating that of Hu and Wellman
    
    Agents have goal states; get +MAX_REW reward for goal, and transition to
    start location after 1 turn. Repeat.
    '''
    # location of P1, P2
    geom_states = [(i, j) for i in range(D**2) for j in range(D**2) if i != j]
    states = list(range(len(geom_states)))
    # clockwise starting with up = 0
    actions = list(range(4))
    S = len(states)
    A = len(actions)
    rew = np.ones((S, A, A))*NEUTRAL
    rewards_1 = rew
    rewards_2 = rew.copy()
    TP = np.zeros((S, A, A, S))
    return geom_states, states, actions, S, A, rewards_1, rewards_2, TP


def act_map(a):
    '''
    Converts action index into coordinates representing the direction
    '''
    d = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0,-1)}
    return d[a]


def cart_map_for(grid_index, D):
    '''
    Converts wraparound grid index to a coordinate pair
    '''
    x = grid_index // D
    y = grid_index % D
    return x, y


def cart_map_back(coords, D):
    '''
    Converts coordinate pair to wraparound grid index
    '''
    return coords[0]*D + coords[1]


def grid_step(c1, c2, D):
    '''
    Returns state resulting from starting in c1 and moving in direction c2
    '''
    new_state = np.array(c1) + np.array(c2)
    return tuple(np.minimum(np.maximum(new_state, 0), D-1))


def TP_check_1(s, a1, a2, sp, geom_states):
    '''
    Grid world 1
    Returns whether valid transition from s to sp given actions a1, a2
    '''
    start = geom_states[s] #(p1,p2)
    prop_end = geom_states[sp] #proposed end state
    st1 = start[0]
    st2 = start[1]
    end_1 = grid_step(cart_map_for(st1, D), act_map(a1), D)
    end_2 = grid_step(cart_map_for(st2, D), act_map(a2), D)
    # If start state is a goal state, player *must* go back to init
    if st1 == D**2-1:
        end_1 = (0, 0)
        # If other player attempts to go to a teleporting player's init state,
        # teleporting player stays in place
        if end_2 == (0, 0):
            end_1 = cart_map_for(st1, D)
    if st2 == D**2-D:
        end_2 = (0, D-1)
        if end_1 == (0, D-1):
            end_2 = cart_map_for(st2, D)
    # If players try to move into same spot or try to pass through each other, bounce back
    if end_1 == end_2 or (end_1 == cart_map_for(st2, D) and end_2 == cart_map_for(st1, D)):
        end_1 = cart_map_for(st1, D)
        end_2 = cart_map_for(st2, D)
    end = (cart_map_back(end_1, D), cart_map_back(end_2, D))
    return int(end == prop_end)


def TP_check(s, a1, a2, sp, geom_states):
    '''
    (For grid world)
    Returns whether valid transition from s to sp given actions a1, a2
    '''
    start = geom_states[s] #(p1,p2)
    prop_end = geom_states[sp] #proposed end state
    st1 = start[0]
    st2 = start[1]
    end_1 = grid_step(cart_map_for(st1, D), act_map(a1), D)
    end_2 = grid_step(cart_map_for(st2, D), act_map(a2), D)
    # If start state is a goal state, player *must* go back to init
    if st1 == ceil(D**2-(D+1)/2):
        end_1 = (0, 0)
        # If other player attempts to go to a teleporting player's init state,
        # teleporting player stays in place
        if end_2 == (0, 0):
            end_1 = cart_map_for(st1, D)
    if st2 == ceil(D**2-(D+1)/2):
        end_2 = (0, D-1)
        if end_1 == (0, D-1):
            end_2 = cart_map_for(st2, D)
    if (end_1 == end_2 or 
      (end_1 == cart_map_for(st2, D) and end_2 == cart_map_for(st1, D))):
        end_1 = cart_map_for(st1, D)
        end_2 = cart_map_for(st2, D)
    end = (cart_map_back(end_1, D), cart_map_back(end_2, D))
    
    # Accounting for barrier crossing
    both = 0
    if (st1 == 0 or st1 == D-1) and a1 == 0:
        if prop_end[0] == st1:
            if prop_end[1] == cart_map_back(end_2, D):
                return 1
            # clash so players stay in place
            if prop_end[1] == st2 and end_2 == cart_map_for(st1, D):
                return 1
            both += 1
    if (st2 == 0 or st2 == D-1) and a2 == 0:
        if prop_end[1] == st2:
            if prop_end[0] == cart_map_back(end_1, D):
                return 1
            if prop_end[0] == st1 and end_1 == cart_map_for(st2, D):
                return 1
            both += 1
    if both == 2:
        return 1
    return int(end == prop_end)


def softmax(vec, arr=False):
    if arr:
        v = np.exp(vec - np.max(vec, axis=1)[:,None])
        return v / np.sum(v, axis=1)[:,None]
    v = np.exp(vec - np.max(vec))
    return v / np.sum(v)


def parTP(TP, theta, fullgame, game_id, pi_function):
    '''
    Given TPs corresponding to a grid, computes the world model not
    conditioned on P2's policy by summing over the hypothesized pi2 (given by
    theta and the array of policies). Player 1 uses this for planning.
    
    pi_function: one of the P2 policy models below
    '''
    pol = pi_function(theta, fullgame, game_id)
    return np.einsum('ijkl,ik->ijl', fullgame['TP'][game_id], pol) # S,A,S


def policynorm(M1, M2):
    '''
    Norm used for change detection
    '''
    return np.max(np.abs(M1 - M2).sum(axis=1))


#%%
    
'''
Constructing games
'''

def grid_1(D):
    '''
    Grid world 1 from Hu and Wellman (see base_policies_1 below)
    '''
    geom_states, states, actions, S, A, rewards_1, rewards_2, TP = gen_grid(D)
    # putting reward 1 on every state in which player
    # is in goal
    for s in range(len(geom_states)):
        if geom_states[s][0] == D**2-1:
            for a1 in range(A):
                for a2 in range(A):
                    rewards_1[s,a1,a2] = MAX_REW
    for s in range(len(geom_states)):
        if geom_states[s][1] == D**2-D:
            for a1 in range(A):
                for a2 in range(A):
                    rewards_2[s,a1,a2] = MAX_REW
    # Tensor for transitions conditional on both players' actions
    for s in range(S):
        for a1 in range(A):
            for a2 in range(A):
                # Assign probs to resultant states
                for sp in range(S):
                    TP[s,a1,a2,sp] = TP_check_1(s, a1, a2, sp, geom_states)
                # Collision penalty, occurs when actions aimed at same state
                # or running into each other.
                start = geom_states[s]
                st1 = start[0]
                st2 = start[1]
                end_1 = grid_step(cart_map_for(st1, D), act_map(a1), D)
                end_2 = grid_step(cart_map_for(st2, D), act_map(a2), D)
                if end_1 == end_2 or (end_1 == cart_map_for(st2, D) and end_2 == cart_map_for(st1, D)):
                    rewards_1[s,a1,a2] = -COLLISION
                    rewards_2[s,a1,a2] = -COLLISION
    init_state = geom_states.index((0, 2))
    return geom_states, states, actions, rewards_1, rewards_2, TP, init_state



def grid(D):
    '''
    Grid world 2 from Hu and Wellman
    '''
    geom_states, states, actions, S, A, rewards_1, rewards_2, TP = gen_grid(D)
    # same goal state for both players
    for s in range(len(geom_states)):
        if geom_states[s][0] == ceil(D**2-(D+1)/2):
            for a1 in range(A):
                for a2 in range(A):
                    rewards_1[s,a1,a2] = MAX_REW
    for s in range(len(geom_states)):
        if geom_states[s][1] == ceil(D**2-(D+1)/2):
            for a1 in range(A):
                for a2 in range(A):
                    rewards_2[s,a1,a2] = MAX_REW
    # Tensor for transitions conditional on both players' actions
    for s in range(S):
        for a1 in range(A):
            for a2 in range(A):
                # Assign probs to resultant states
                for sp in range(S):
                    TP[s,a1,a2,sp] = TP_check(s, a1, a2, sp, geom_states)
                tot = np.sum(TP[s,a1,a2])
                for sp in range(S):
                    TP[s,a1,a2,sp] /= tot
                # Collision penalty, occurs when actions aimed at same state
                # or running into each other.
                start = geom_states[s]
                st1 = start[0]
                st2 = start[1]
                end_1 = grid_step(cart_map_for(st1, D), act_map(a1), D)
                end_2 = grid_step(cart_map_for(st2, D), act_map(a2), D)
                if (end_1 == end_2 or
                  (end_1 == cart_map_for(st2, D) and
                  end_2 == cart_map_for(st1, D))):
                    rewards_1[s,a1,a2] = -COLLISION
                    rewards_2[s,a1,a2] = -COLLISION
    init_state = geom_states.index((0, 2))
    return geom_states, states, actions, rewards_1, rewards_2, TP, init_state


def iterated_game(base_1, base_2, K, init_state):
    '''
    Creates game objects for an iteration of a pair of base game matrices
    
    If each player has A actions, and K is the number of past actions included
    in the state, there are A^(2K) states
    
    Here "geom_states" will really be tuple representation of past actions,
    kinda misleading but I already picked that term
    
    Here the init_state seems quite crucial, determines how trusting the
    players are going into the game
    '''
    actions = list(range(base_1.shape[0]))
    A = len(actions)
    geom_states = list(itertools.product(list(range(A)), repeat=2*K))
    states = list(range(len(geom_states)))
    S = len(states)
    rew = np.zeros((S, A, A))
    # No matter the history, same payoffs for joint actions
    rewards_1 = rew.copy()
    b1 = base_1 - np.min(base_1)
    rewards_1[:] = b1 / np.max(b1)
    rewards_2 = rew.copy()
    b2 = base_2 - np.min(base_2)
    rewards_2[:] = b2 / np.max(b2)
    TP = np.zeros((S, A, A, S))
    for s in range(S):
        for a1 in range(A):
            for a2 in range(A):
                for sp in range(S):
                    start = geom_states[s]
                    end = geom_states[sp]
                    if K == 2:
                        if start[1] == end[0] and start[3] == end[2]:
                            # Last action becomes second-to-last
                            if end[1] == a1 and end[3] == a2:
                                # Current action becomes last
                                TP[s,a1,a2,sp] = 1
                    if K == 1:
                        if end[0] == a1 and end[1] == a2:
                            TP[s,a1,a2,sp] = 1
    return geom_states, states, actions, rewards_1, rewards_2, TP, init_state


def base_policies_ipd(init_state):
    '''
    Iterated PD base strategies
    '''
    K = 2
    base_1 = np.array([[3, 0],[4, 1]])
    # Note that for the purposes of accessing rewards with emp_avg_return,
    # we represent the reward matrix for player 2 *from P1's perspective*
    base_2 = np.array([[3, 4],[0, 1]])
    geom_states, states, actions, r1, r2, TP, init = iterated_game(base_1,
        base_2, K, init_state)
    bully = np.zeros((len(states), len(actions)))
    tft = np.zeros((len(states), len(actions)))
    pavlov = np.zeros((len(states), len(actions)))
    # Forgiving tit-for-tat: defects on two consecutive defections, otherwise
    # coops
    fortft = np.zeros((len(states), len(actions)))
    bully[:,1] = 1
    for s, gs in enumerate(geom_states):
        tft[s,gs[1]] = 1
        pavlov[s,int(gs[1] != gs[3])] = 1
        fortft[s,int(gs[0] == 1 and gs[1] == 1)] = 1
    return np.array([bully, tft, pavlov, fortft])


def base_policies_bos(init_state):
    '''
    Bach-or-Stravinsky base strategies
    '''
    K = 2
    base_1 = np.array([[1, 0],[0, 2]])
    base_2 = np.array([[2, 0],[0, 1]])
    geom_states, states, actions, r1, r2, TP, init = iterated_game(base_1,
      base_2, K, init_state)
    fair = np.ones((len(states), len(actions))) / 2
    nash = np.ones((len(states), len(actions))) * 2/3
    nash[:,1] /= 2
    # Trades between the two equilibria; if both players coordinated on one
    # last turn, goes to the other. If players failed to coordinate last
    # turn, goes to favored equilibrium.
    seq = np.zeros((len(states), len(actions)))
    # "Forgiving." Again, if both players coordinated last turn, switches.
    # If failed to coordinate last but not second-to-last, goes to player 1's
    # favored equilibrium. Otherwise, goes to favored equilibrium.
    forseq = np.zeros((len(states), len(actions)))
    for s, gs in enumerate(geom_states):
        if gs[1] == gs[3]:
            seq[s,1-gs[3]] = 1
        else:
            seq[s,0] = 1
        if gs[1] == gs[3]:
            forseq[s,1-gs[3]] = 1
        elif gs[0] == gs[2]:
            forseq[s,1] = 1
        else:
            forseq[s,0] = 1
    return np.array([fair, nash, seq, forseq])


def base_policies_1(D):
    '''
    Grid 1 from Hu and Wellman - *not used in final paper*, but included in
    code for reproducibility (consistent random seeds)
    '''
    geom_states, states, actions, rewards_1, rewards_2, TP, init_state = grid_1(D) 
    policies = []
    S = len(states)
    A = len(actions)
    # This is the hypothesized policy that P1 expects P2 to believe P1 is following
    pi1hat = np.zeros((S, A))
    rights = list(range(D**2-D, D**2-1))
    ups = list(range(D-1, D**2, D))
    for s, gs in enumerate(geom_states):
        if gs[0] in ups:
            pi1hat[s,0] = 1
        elif gs[0] in rights:
            pi1hat[s,1] = 1
        else:
            pi1hat[s,0] = 1/2
            pi1hat[s,1] = 1/2
    # Different rewards P1 believes P2 might receive from a collision.
    reward_hypos = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for rval in reward_hypos:
        rew2 = rewards_2.copy()
        rew2[rew2 == 0] = rval
        pTP = np.einsum('ijkl,ik->ijl', TP, pi1hat)
        pol2 = value_iter(S, A, rew2, pTP, pi1hat, 0.99, 1e-3)[0]
        policies.append(pol2)
    return np.array(policies)


def base_policies_gr(D):
    '''
    Computing base policies that the opponent mixes between, in grid.
    
    These are optimal policies w.r.t. a shortest-path prediction of player 1's
    policy, and a set of different reward values.
    '''
    geom_states, states, actions, r1, r2, TP, init = grid(D) 
    policies = []
    S = len(states)
    A = len(actions)
    # This is the hypothesized policy that P1 expects P2 to believe P1
    # is following
    pi1hat = np.zeros((S, A))
    rights = list(range(D**2-D, ceil(D**2-(D+1)/2)))
    lefts = list(range(ceil(D**2-(D+1)/2)+1, D**2))
    ups = list(range(ceil((D-1)/2), ceil(D**2-(D+1)/2)+1, D))
    for s, gs in enumerate(geom_states):
        if gs[0] in ups:
            pi1hat[s,0] = 1
        elif gs[0] in rights:
            # Square on top row, left of center
            pi1hat[s,1] = 1
        elif gs[0] in lefts:
            # Square on top row, right of center
            pi1hat[s,3] = 1
        else:
            pi1hat[s,0] = 1/2
            if cart_map_for(gs[0], D)[1] < ceil((D-1)/2):
                pi1hat[s,1] = 1/2
            else:
                pi1hat[s,3] = 1/2
    # Different rewards P1 believes P2 might receive from a collision.
    reward_hypos = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for rval in reward_hypos:
        rew2 = r2.copy()
        rew2[rew2 == 0] = rval
        pTP = np.einsum('ijkl,ik->ijl', TP, pi1hat)
        pol2 = value_iter(S, A, rew2, pTP, pi1hat, 0.99, 1e-3)[0]
        policies.append(pol2)
    return np.array(policies)


#%%
    
'''
Defining P2 policy models
'''

def good_to_self(a1, base_1, base_2, player_ind=1):
    '''
    Computes how good P1's past action was to P2 conditional on P2 playing
    best response, as a measure of how acceptable P1's behavior is to P2 on
    selfish grounds.
    '''
    if player_ind:
        return np.max(base_2[a1])
    return np.max(base_1[:,a1])


def good_to_other(a1, base_1, base_2, player_ind=1):
    '''
    Computes how good other's past action was to other conditional on self
    playing best response, used in computing how acceptable other's behavior
    is to self on fairness grounds.
    '''
    if player_ind:
        P2_best = np.argmax(base_2[a1])
        return base_1[a1,P2_best]
    P1_best = np.argmax(base_1[:,a1])
    return base_2[P1_best,a1]


def bully_payoff(a2, base_1, base_2, player_ind=1):
    '''
    Computes how good self's action under consideration is for self
    conditional on other playing a best response.
    '''
    if player_ind:
        P1_best = np.argmax(base_1[:,a2])
        return base_2[P1_best,a2]
    P2_best = np.argmax(base_2[a2])
    return base_1[a2,P2_best]


def minimax_payoff(a2, base_1, base_2, player_ind=1):
    '''
    Computes how bad P2's action under consideration is for P1 conditional on
    P1 playing a best response.
    '''
    if player_ind:
        P1_best = np.argmax(base_1[:,a2])
        return 1 - base_1[P1_best,a2]
    P2_best = np.argmax(base_2[a2])
    return 1 - base_2[a2,P2_best]


def godfather(theta, base_1, base_2, player_ind=1):
    '''
    Assumes a theta of the form:
      theta = (bully, punish, fairness, forgiveness)
    '''
    K = 1
    gs_g, states_g, actions_g, r1_g, r2_g, TP_g, init = iterated_game(base_1,
      base_2, K, 0)
    # Rescaled to [0, 1]
    base_1 = r1_g[0]
    base_2 = r2_g[0]
    pol2 = np.zeros((len(states_g), len(actions_g)))
    for s in range(len(states_g)):
        a1 = gs_g[s][1-player_ind]
        for a2 in range(len(actions_g)):
            score = theta[0]*bully_payoff(a2, base_1, base_2)
            punish_1 = minimax_payoff(a2, base_1, base_2, player_ind)
            if player_ind:
                bul = np.max(base_2)
            else:
                bul = np.max(base_1)
            selfval = good_to_self(a1, base_1, base_2, player_ind)
            otherval = good_to_other(a1, base_1, base_2,player_ind)
            egal = np.max(base_1*base_2) - selfval*otherval
            punish_1 *= bul - selfval + theta[2]*egal
            # if punished last turn, downweight punishing actions; we do this
            # by subtracting a term that should be higher if punished last
            # turn, and multiplying this term by the utility of the action
            # for P1, so punishing actions are lower weight
            pun2_a = minimax_payoff(gs_g[s][player_ind], base_1, base_2,
              player_ind)
            punish_2 = -theta[3]*pun2_a*punish_1
            punish = punish_1 + punish_2
            score += theta[1]*punish
            pol2[s,a2] = score
    pol2 = softmax(pol2, arr=True)
    return pol2


def lin_pi2(theta, fullgame, game_id):
    '''
    Linear mixture model for Experiments 1-2
    '''
    policies = fullgame['policies'][game_id]
    return np.einsum('i,ijk->jk', theta, policies)


def for_gen_pi2(theta, fullgame, game_id, player_ind=1):
    '''
    Godfather model for Experiment 3
    '''
    base_1 = fullgame['base_1'][game_id]
    base_2 = fullgame['base_2'][game_id]
    return godfather(theta, base_1, base_2, player_ind=player_ind)


def sp_for_gen_pi2(theta, fullgame, game_id):
    # Used when the P2 policy model is an input to another function, to avoid
    # carrying through player_ind argument
    base_1 = fullgame['base_1'][game_id]
    base_2 = fullgame['base_2'][game_id]
    return godfather(theta, base_1, base_2, player_ind=0)


#%%
    
'''
RL functions
'''

def value_iter(S, A, rewards_1, pTP, pol2, gam, tol):
    '''
    Action-value iteration for Q* and optimal policy (using known pTPs).
    For computational simplicity, using discounted version to approximate
    average-reward optimal.
    '''
    Q = np.random.rand(S, A)
    delta = np.inf
    # expected reward of action in current state, where expectation is over
    # other player's policy
    expect_rewards = np.einsum('ijk,ik->ij', rewards_1, pol2)
    while delta > tol:
        delta = 0
        for s in range(S):
            a = np.random.choice(np.arange(A))
            qval = Q[s,a]
            Q[s,a] = expect_rewards[s,a] + gam*(pTP.dot(Q.max(axis=1)))[s,a]
            delta = np.max([delta, abs(qval - Q[s,a])])
    policy = np.zeros((S, A))
    policy[np.arange(S),np.argmax(Q, axis=1)] = 1
    return policy, Q

    
def emp_avg_return(policy_1, rewards_1, policy_2, L, TP, s, S, A):
    rew_list_1 = []
    for _ in range(L):
        a1 = np.random.choice(np.arange(A), p=policy_1[s])
        a2 = np.random.choice(np.arange(A), p=policy_2[s])
        rew_list_1.append(rewards_1[s,a1,a2])
        s = np.random.choice(np.arange(S), p=TP[s,a1,a2])
    return np.mean(rew_list_1)


def epoch(ucrl, t, L, s, true_thetas, policy_1, policy_2, nu_schedule, m, T,
          fullgame, game_id, pi_function=lin_pi2, selfplay=False):
    '''
    Given any generic state, time horizon, policy, and reward structure indexed
    by the state, generates an epoch starting from that state.
    
    Epoch runs for L time steps.
    
    true_thetas: list of the opponent's parameter vectors
    m: index of param
    '''
    states = []
    acts_1 = []
    acts_2 = []
    
    TP = fullgame['TP'][game_id]
    S = len(fullgame['states'][game_id])
    A = len(fullgame['actions'][game_id])
    rewards_1 = fullgame['rewards_1'][game_id]
    rewards_2 = fullgame['rewards_2'][game_id]
    Nsa_1 = np.zeros((S, A))
    Nsa_2 = np.zeros((S, A))
    rew1 = []
    rew2 = []
    tee = t
    if not selfplay:
        policy_2 = pi_function(true_thetas[m-1], fullgame, game_id)
    while t < min(tee + L, T):
        # If testing the UCRL2-esque reset schedule, reset data at each
        # i**3/M**2. We do not use the resetting schedule in self play.
        if ucrl and t in RESET_LIST:
            Nsa_2[:,:] = 0
        if not selfplay and len(nu_schedule)-1 > m:
            if t == nu_schedule[m]:
                if np.any(true_thetas[m-1] != true_thetas[m]):
                    # only if an actual ``switch"
                    m += 1
                    policy_2 = pi_function(true_thetas[m-1], fullgame, game_id)
        states.append(s)
        a1 = np.random.choice(np.arange(A), p=policy_1[s])
        acts_1.append(a1)
        a2 = np.random.choice(np.arange(A), p=policy_2[s])
        acts_2.append(a2)
        rew1.append(rewards_1[s,a1,a2])
        rew2.append(rewards_2[s,a1,a2])
        Nsa_1[s,a1] += 1
        Nsa_2[s,a2] += 1
        s = np.random.choice(np.arange(S), p=TP[s,a1,a2])
        t += 1
    if selfplay:
        return states, acts_1, acts_2, rew1, rew2, t, m, Nsa_1, Nsa_2
    return states, acts_1, acts_2, rew1, rew2, t, m, Nsa_2


def CD(data, data_pr, threshold, verbose, fullgame, game_id, pi_function,
       parametric=False, th=0, th_pr=0):
    '''
    Computes empirical policy matrices based on the datasets, and
    checks if their difference norm exceeds the threshold
    
    Change detection
    '''
    if parametric:
        # MULTIPLE is used to adjust threshold for parametric CD to match the
        # scale of the parameters
        delta = np.linalg.norm(th - th_pr)*MULTIPLE
    else:
        M1 = data / data.sum(axis=1)[:,None]
        M2 = data_pr / data_pr.sum(axis=1)[:,None]
        M1 = np.nan_to_num(M1, nan=1/M1.shape[1])
        M2 = np.nan_to_num(M2, nan=1/M2.shape[1])
        delta = policynorm(M1, M2)
    if verbose:
        print('delta == ' + str(delta))
    return delta > threshold


#%%
    
'''
Posterior sampling functions
'''


def loglikelihood(theta, data, fullgame, game_id, pi_function):
    '''
    Log-likelihood of data and parameter value theta for a given pi_function,
    for use in posterior sampling
    '''
    if pi_function == lin_pi2:
        if np.sum(theta) == 1 and (theta <= 1).all() and (theta >= 0).all():
            pol2_1 = pi_function(theta, fullgame, game_id)
            lg = np.log(pol2_1)
            lg[lg == -np.inf] = 0
            arr1 = data*lg
        else:
            arr1 = -np.inf
    else:
        if np.any(theta < 0):
            arr1 = -np.inf
        else:
            pol2_1 = pi_function(theta, fullgame, game_id)
            lg = np.log(pol2_1)
            lg[lg == -np.inf] = 0
            arr1 = data*lg
    return np.sum(arr1)
    

class LogLike(tt.Op):
    '''
    CREDIT: PYMC3 TUTORIAL FOR BLACKBOX
    https://docs.pymc.io/notebooks/blackbox_external_likelihood.html
    
    Specify what type of object will be passed and returned to the Op when
    it is called. In our case we will be passing it a vector of values
    (the parameters that define our model) and returning a single "scalar"
    value (the log-likelihood)
    '''
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, fullgame, game_id, pi_function):
        '''
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        '''

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.fullgame = fullgame
        self.game_id = game_id
        self.pi_function = pi_function

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.data, self.fullgame, self.game_id,
                               self.pi_function)

        outputs[0][0] = np.array(logl) # output the log-likelihood
        
        
def post_samp(loglike, data, ndraws, nburn, fullgame, game_id, met, verbose,
              pi_function=lin_pi2, parametric=False):
    '''
    MCMC algorithm for sampling from a posterior.
    '''
    logl = LogLike(loglike, data, fullgame, game_id, pi_function)
    with pm.Model():
        # Setting prior based on model for the experiment
        if game_id == 'g':
            if np.all(MUS == 0):
                vari = pm.distributions.continuous.Lognormal('theta', 0, 0.5,
                  shape=THETA_DIM)
                th = tt.as_tensor_variable(vari)
            else:
                varis = [pm.distributions.continuous.Lognormal('theta_'+str(i),
                  MUS[i], 0.5) for i in range(THETA_DIM)]
                th = tt.stack(varis)
        else:
            vari = pm.distributions.multivariate.Dirichlet('theta', ALPHA_VEC)
            th = tt.as_tensor_variable(vari)
        pm.Potential('likelihood', logl(th))#, lambda v: logl(v), observed={'v': th})
        if met:
            trace = pm.sample(ndraws, tune=nburn, cores=NUM_CORES,
              step=pm.Metropolis(), discard_tuned_samples=True,
              progressbar=False)
        else:
            trace = pm.sample(ndraws, tune=nburn, cores=NUM_CORES,
              discard_tuned_samples=True, progressbar=False)
    if verbose:
        print(pm.summary(trace))
    if game_id == 'g' and np.any(MUS != 0):
        if parametric:
            th_l = [np.mean(trace['theta_' + str(i)][-int(ndraws)//2:],
              axis=0) for i in range(THETA_DIM)]
        else:
            th_l = [trace['theta_' + str(i)][-1] for i in range(THETA_DIM)]
        return np.array(th_l)
    return np.mean(trace['theta'][-int(ndraws)//2:],
      axis=0) if parametric else trace['theta'][-1]


def full_game(ucrl, threshold, T, L, fullgame, game_id, true_thetas,
              nu_schedule, met, ndraws, nburn=1000, gam=0.99, tol=0.001,
              verbose=True, pi_function=lin_pi2, parametric=False):
    '''
    One run of a game for a given nu_schedule and parameters true_thetas
    '''
    S = len(fullgame['states'][game_id])
    A = len(fullgame['actions'][game_id])
    TP = fullgame['TP'][game_id]
    rewards_1 = fullgame['rewards_1'][game_id]
    s0 = fullgame['s0'][game_id]
    s = s0
    t = 0
    data = np.zeros((S, A))
    data_pr = np.zeros((S, A))
    rew_1_all = []
    rew_2_all = []
    Nsa = np.zeros((S, A)) # The whole memory can be summarized in Nsa matrix
    M = len(true_thetas)-1
    m = 1 # Index for phase in sequence of opponent policies
    switched = True
    th = 0
    th_pr = 0
    while t < T:
        if not (data == 0).all() and not (data_pr == 0).all():
            # i.e. if you can even do CD in first place
            if verbose:
                print('m == ' + str(m))
            if parametric:
                th = post_samp(loglikelihood, data, ndraws, nburn, fullgame,
                  game_id, met, verbose, pi_function=pi_function,
                  parametric=True)
                th_pr = post_samp(loglikelihood, data_pr, ndraws, nburn,
                  fullgame, game_id, met, verbose, pi_function=pi_function,
                  parametric=True)
            if CD(data, data_pr, threshold, verbose, fullgame, game_id,
              pi_function, parametric=parametric, th=th, th_pr=th_pr):
                # reset if detect a change
                Nsa = np.zeros((S, A))
                if verbose:
                    print('DETECTED')
        # Sample from posterior
        theta = post_samp(loglikelihood, Nsa, ndraws, nburn, fullgame,
          game_id, met, verbose, pi_function=pi_function)
        if verbose:
            print('Believed Theta == ' + str(theta))
            print('True Theta == ' + str(true_thetas[m-1]))
        # Compute optimal policy w.r.t. belief theta
        policy_1 = value_iter(S, A, rewards_1, parTP(TP, theta, fullgame,
          game_id, pi_function), pi_function(theta, fullgame, game_id),
          gam, tol)[0]
        if verbose:
            opt_pol = value_iter(S, A, rewards_1, parTP(TP, true_thetas[m-1],
              fullgame, game_id, pi_function), pi_function(true_thetas[m-1],
              fullgame, game_id), gam, tol)[0]
            print('Policy Norm Diff == ' + str(policynorm(policy_1, opt_pol)))
        if switched or m > M:
            switched = False
        states, acts_1, acts_2, r_1, r_2, t, new_m, Nsa_e = epoch(ucrl, t, L,
          s, true_thetas, policy_1, np.zeros(1), nu_schedule, m, T, fullgame,
          game_id, pi_function)
        if new_m != m:
            switched = True
        m = new_m
        rew_1_all = rew_1_all + r_1
        rew_2_all = rew_2_all + r_2
        Nsa += Nsa_e
        data = data_pr
        data_pr = Nsa_e
        if verbose:
            print('t == ' + str(t))
            print('Mean Epoch Reward == ' + str(np.mean(r_1)))
            print('Tau Schedule == ' + str(nu_schedule) + '\n\n\n')
    return np.array(rew_1_all), np.array(rew_2_all), nu_schedule


def game_self_play(ucrl, threshold, T, L, fullgame, game_id, met, ndraws,
                   nburn=1000, gam=0.99, tol=0.001, verbose=True,
                   pi_function=lin_pi2, parametric=False):
    '''
    One run of self-play.
    '''
    S = len(fullgame['states'][game_id])
    A = len(fullgame['actions'][game_id])
    TP = fullgame['TP'][game_id]
    rewards_1 = fullgame['rewards_1'][game_id]
    rewards_2 = fullgame['rewards_2'][game_id]
    s0 = fullgame['s0'][game_id]
    s = s0
    t = 0
    data_1 = np.zeros((S, A))
    data_1_pr = np.zeros((S, A))
    data_2 = np.zeros((S, A))
    data_2_pr = np.zeros((S, A))
    rew_1_all = []
    rew_2_all = []
    Nsa_1 = np.zeros((S, A)) # The whole memory can be summarized in Nsa matrix
    Nsa_2 = np.zeros((S, A))
    m = 1 # Index for phase in sequence of opponent policies
    opt_pol_1 = np.ones((S, A)) / A
    opt_pol_2 = np.ones((S, A)) / A
    policy_1 = np.ones((S, A)) / A
    policy_2 = np.ones((S, A)) / A
    nu_schedule = []
    best = np.zeros(T)
    old_t = 0
    th = 0
    th_pr = 0
    while t < T:
        if not (data_2 == 0).all() and not (data_2_pr == 0).all():
            if verbose:
                print('m == ' + str(m))
            if parametric:
                th = post_samp(loglikelihood, data_2, ndraws, nburn, fullgame,
                  game_id, met, verbose, pi_function=pi_function,
                  parametric=True)
                th_pr = post_samp(loglikelihood, data_2_pr, ndraws, nburn,
                  fullgame, game_id, met, verbose, pi_function=pi_function,
                  parametric=True)
            if CD(data_2, data_2_pr, threshold, verbose, fullgame, game_id,
              pi_function, parametric=parametric, th=th, th_pr=th_pr):
                Nsa_2 = np.zeros((S, A))
                if verbose:
                    print('DETECTED')
        theta_2 = post_samp(loglikelihood, Nsa_2, ndraws, nburn, fullgame,
          game_id, met, verbose, pi_function=pi_function)
        if verbose:
            print('Theta Believed by P1 == ' + str(theta_2))
        if not (data_1 == 0).all() and not (data_1_pr == 0).all():
            if parametric:
                if pi_function == for_gen_pi2:
                    th = post_samp(loglikelihood, data_1, ndraws, nburn,
                      fullgame, game_id, met, verbose,
                      pi_function=sp_for_gen_pi2, parametric=True)
                    th_pr = post_samp(loglikelihood, data_1_pr, ndraws, nburn,
                      fullgame, game_id, met, verbose,
                      pi_function=sp_for_gen_pi2, parametric=True)
                else:
                    th = post_samp(loglikelihood, data_1, ndraws, nburn,
                      fullgame, game_id, met, verbose,
                      pi_function=pi_function, parametric=True)
                    th_pr = post_samp(loglikelihood, data_1_pr, ndraws, nburn,
                      fullgame, game_id, met, verbose,
                      pi_function=pi_function, parametric=True)
            if CD(data_1, data_1_pr, threshold, verbose, fullgame, game_id,
              sp_for_gen_pi2 if pi_function == for_gen_pi2 else pi_function,
              parametric=parametric, th=th, th_pr=th_pr):
                Nsa_1 = np.zeros((S, A))
        # In self-play, the general parametric policy is asymmetric;
        # need to use P1 version
        if pi_function == for_gen_pi2:
            theta_1 = post_samp(loglikelihood, Nsa_1, ndraws, nburn, fullgame,
              game_id, met, verbose, pi_function=sp_for_gen_pi2)
        else:
            theta_1 = post_samp(loglikelihood, Nsa_1, ndraws, nburn, fullgame,
              game_id, met, verbose, pi_function=pi_function)
        if verbose:
            print('Theta Believed by P2 == ' + str(theta_1))
        # Compute optimal policy w.r.t. belief theta
        if verbose and t > 0:
            print('Policy Norm Diff for P1 == ' + str(policynorm(policy_1,
              opt_pol_1)))
            print('Policy Norm Diff for P2 == ' + str(policynorm(policy_2,
              opt_pol_2)))
        policy_1 = value_iter(S, A, rewards_1, parTP(TP, theta_2, fullgame,
          game_id, pi_function), pi_function(theta_2, fullgame, game_id),
          gam, tol)[0]
        #print(policy_1[:10])
        if pi_function == for_gen_pi2:
            pTP = np.einsum('ijkl,ij->ikl', fullgame['TP'][game_id],
              sp_for_gen_pi2(theta_1, fullgame, game_id))
            policy_2_new = value_iter(S, A, np.swapaxes(rewards_2, 1, 2),
              pTP, sp_for_gen_pi2(theta_1, fullgame, game_id), gam, tol)[0]
        else:
            pTP = np.einsum('ijkl,ij->ikl', fullgame['TP'][game_id],
              pi_function(theta_1, fullgame, game_id))
            policy_2_new = value_iter(S, A, np.swapaxes(rewards_2, 1, 2),
              pTP, pi_function(theta_1, fullgame, game_id), gam, tol)[0]
        if np.any(policy_2 != policy_2_new):
            m += 1
            if verbose:
                print('nu == ' + str(t))
            nu_schedule.append(t)
        policy_2 = policy_2_new.copy()
        old_t = t
        states, acts_1, acts_2, r_1, r_2, t, m, Nsa_e_1, Nsa_e_2 = epoch(ucrl,
          t, L, s, [], policy_1, policy_2, [], m, T, fullgame, game_id,
          selfplay=True)
        rew_1_all = rew_1_all + r_1
        rew_2_all = rew_2_all + r_2
        Nsa_1 += Nsa_e_1
        Nsa_2 += Nsa_e_2
        data_1 = data_1_pr
        data_2 = data_2_pr
        data_1_pr = Nsa_e_1
        data_2_pr = Nsa_e_2
        if verbose:
            print('t == ' + str(t))
            print('Mean Epoch Reward == ' + str(np.mean(r_1)))
            print('Tau Schedule == ' + str(nu_schedule) + '\n\n\n')
        # Need opt pol to compute regret
        opt_pol_1 = value_iter(S, A, rewards_1, np.einsum('ijkl,ik->ijl',
          fullgame['TP'][game_id], policy_2), policy_2, gam, tol)[0]
        mu = emp_avg_return(opt_pol_1, rewards_1, policy_2, T, TP, s, S, A)
        best[old_t:t] = mu
        if verbose:
            # below is not a typo, need to sum over A1 index rather than A2
            # Need to swap axes for rewards bc of indexing in value_iter
            opt_pol_2 = value_iter(S, A, np.swapaxes(rewards_2, 1, 2),
              np.einsum('ijkl,ij->ikl', fullgame['TP'][game_id], policy_1),
              policy_1, gam, tol)[0]
    return np.array(rew_1_all), np.array(rew_2_all), nu_schedule, best


def experiment(N, T, L, fullgame, game_id, true_thetas, nu_schedule_list, met,
               ndraws, nburn=1000, gam=0.99, tol=0.001, verbose=False,
               pi_function=lin_pi2, parametric=False):
    '''
    Executes N runs of a given game and saves results.
    
    Makes a plot of results, however this is *not* the plot used in the
    final paper.
    '''
    # Makes the necessary directories if they don't already exist
    if not os.getcwd().endswith('multi'):
        os.mkdir(os.getcwd() + '/multi')
        os.chdir(os.getcwd() + '/multi')
        os.mkdir('cd_results')
    if not os.path.isdir('cd_results/' + game_id):
        os.mkdir('cd_results/' + game_id)
    S = len(fullgame['states'][game_id])
    A = len(fullgame['actions'][game_id])
    TP = fullgame['TP'][game_id]
    rewards_1 = fullgame['rewards_1'][game_id]
    s0 = fullgame['s0'][game_id]
    s = s0
    thresholds = THRESHOLD_LIST #[1.5, 3, 4]
    lsty = {1.5: 'solid', 3: 'dashed', 4: 'dashdot'}
    
    for z, thr in enumerate(thresholds):
        THRESHOLD = thr
        ucrl = (thr == 4)
        UCRL = ucrl
        full_best = [np.zeros(T) for ts in nu_schedule_list]
        full_rew_1_list = [np.zeros((N, T)) for ts in nu_schedule_list]
        colors = plt.cm.brg(np.linspace(0,1,len(nu_schedule_list)))
        regrets = []
        unavg_regrets = []
        mu = []
        # Writing raw results to files
        alpstring = '-'.join(str(ALPHA_VEC).split(' '))
        id_string = 'N_' + str(N) + '_MET_' + str(MET) + '_ndraws_'
        id_string += str(ndraws) + '_L_' + str(L) + '_ALPHA_VEC_' + alpstring
        id_string += '_MUS_' + mus + '_NUM_CORES_' + str(NUM_CORES)
        id_string += '_THETA_DIM_' + str(THETA_DIM) + '_PARAMETRIC_'
        id_string += str(PARAMETRIC) + '_MULTIPLE_' + str(MULTIPLE)
        id_string += '_CT_SEEDS_' + str(CT_SEEDS) + '_start_seed_3_ELLS_'
        id_string += (','.join([str(i) for i in ELLS])).replace('.','-')
        dirstring = 'cd_results/' + game_id + '/' + id_string.replace('.','-') 
        if not os.path.isdir(dirstring):
            os.mkdir(dirstring)
        for em, th in enumerate(true_thetas):
            # Computing baselines for regret
            pTP = parTP(TP, th, fullgame, game_id, pi_function)
            policy = value_iter(S, A, rewards_1, pTP, pi_function(th, fullgame,
              game_id), gam, tol)[0]
            policy_2 = pi_function(true_thetas[em], fullgame, game_id)
            mu.append(emp_avg_return(policy, rewards_1, policy_2, T, TP, s, S,
              A))
        for j, ts in enumerate(nu_schedule_list):
            best = np.zeros(T)
            for em in range(len(ts)-1):
                best[int(ts[em]):int(ts[em+1])] = mu[em]
            for i in range(N):
                rew_1, rew_2, ts = full_game(ucrl, thr, T, L, fullgame,
                  game_id, true_thetas, ts, met, ndraws, nburn=nburn, gam=gam,
                  tol=tol, verbose=verbose, pi_function=pi_function,
                  parametric=parametric)
                full_rew_1_list[j][i] += rew_1
                regr = best - rew_1
                plt.plot(np.cumsum(regr), color=colors[j], alpha=0.1,
                  linestyle=lsty[thr])
                print('Round ' + str(i+1))
            full_best[j] = best
            mean = np.mean(full_rew_1_list[j], axis=0)
            plt.plot(np.arange(T), np.cumsum(best - mean), color=colors[j],
              linestyle=lsty[thr])
            regrets.append(full_best[j] - np.mean(full_rew_1_list[j], axis=0))
            unavg_regrets.append(full_best[j][None,:] - full_rew_1_list[j])
            eyedee = 'ts_' + (','.join([str(i) for i in ts])).replace('.','-')
            df = pd.DataFrame(unavg_regrets[j])
            datastring = 'cd_results/' + game_id + '/'
            datastring += id_string.replace('.','-') + '/THRESHOLD_'
            datastring += str(THRESHOLD).replace('.','-') + '_UCRL_'
            datastring += str(UCRL) + '_' + eyedee + '.csv'
            df.to_csv(datastring)
        worst_regs = np.max(np.cumsum(np.array(regrets), axis=1), axis=0)
        plt.plot(worst_regs, color='black', linestyle=lsty[thr])
            
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    labels_dict = {1.5: 'CD-TSMG',
                   3: 'TSMDP',
                   4: 'TSMDP-UCRL'}
    leg = [Line2D([0,1],[0,1],linestyle=lsty[thr],
      color='black') for thr in thresholds]
    plt.legend(leg, [labels_dict[thr] for thr in thresholds])
    figst = 'cd_results/' + game_id + '/' + id_string.replace('.','-') + '.png'
    plt.savefig(figst)
    plt.show()
    
    
def self_play(N, T, L, fullgame, game_id, met, ndraws, nburn=1000, gam=0.99,
              tol=0.001, verbose=False, pi_function=lin_pi2, parametric=False):
    '''
    Executes N runs of self-play and saves results.
    '''    
    # Makes the necessary directories if they don't already exist
    if not os.getcwd().endswith('multi'):
        os.mkdir(os.getcwd() + '/multi')
        os.chdir(os.getcwd() + '/multi')
        os.mkdir('cd_results')
    if not os.path.isdir('cd_results/' + game_id):
        os.mkdir('cd_results/' + game_id)
    
    thresholds = THRESHOLD_LIST
    lsty = {1.5: 'solid', 3: 'dashed', 4: 'dashdot'}
    rew_1_tens = np.zeros((len(thresholds), N, T))
    rew_2_tens = np.zeros((len(thresholds), N, T))
    
    for z, thr in enumerate(thresholds):
        THRESHOLD = thr
        ucrl = (thr == 4)
        UCRL = ucrl
        full_best = np.zeros((N, T))
        full_rew_1 = np.zeros((N, T))
        full_rew_2 = np.zeros((N, T))
        # Under self-play, there's no explicit nu schedule
        regrets = []
        # Writing raw results to files
        alpstring = '-'.join(str(ALPHA_VEC).split(' '))
        id_string = 'N_' + str(N) + '_MET_' + str(MET) + '_ndraws_'
        id_string += str(ndraws) + '_L_' + str(L) + '_ALPHA_VEC_' + alpstring
        id_string += '_MUS_' + mus + '_NUM_CORES_' + str(NUM_CORES)
        id_string += '_THETA_DIM_' + str(THETA_DIM) + '_PARAMETRIC_'
        id_string += str(PARAMETRIC) + '_MULTIPLE_' + str(MULTIPLE)
        id_string += '_CT_SEEDS_' + str(CT_SEEDS) + '_start_seed_3_ELLS_'
        id_string += (','.join([str(i) for i in ELLS])).replace('.','-')
        dirstring = 'cd_results/' + game_id + '/' + id_string.replace('.','-') 
        if not os.path.isdir(dirstring):
            os.mkdir(dirstring)
        for i in range(N):
            rew_1, rew_2, ts, best = game_self_play(ucrl, thr, T, L,
                fullgame, game_id, met, ndraws, nburn=nburn, gam=gam, tol=tol,
                verbose=verbose, pi_function=pi_function,
                parametric=parametric)
            full_rew_1[i] += rew_1
            full_rew_2[i] += rew_2
            regr = best - rew_1
            plt.plot(np.cumsum(regr), color='blue', alpha=0.1,
              linestyle=lsty[thr])
            print('Round ' + str(i+1))
            full_best[i] = best
        mean = np.mean(full_rew_1, axis=0)
        best_mean = np.mean(full_best, axis=0)
        plt.plot(np.arange(T), np.cumsum(best_mean - mean), color='blue',
          linestyle=lsty[thr])
        regrets = full_best - full_rew_1
        df = pd.DataFrame(regrets)
        eyedee = 'self_play'
        datastring = 'cd_results/' + game_id + '/'
        datastring += id_string.replace('.','-') + '/THRESHOLD_'
        datastring += str(THRESHOLD).replace('.',"-") + '_UCRL_'
        datastring += str(UCRL) + '_' + eyedee + '.csv'
        df.to_csv(datastring)
        worst_regs = np.max(np.cumsum(np.array(regrets), axis=1), axis=0)
        plt.plot(worst_regs, color='black', linestyle=lsty[thr])
        rew_1_tens[z] = full_rew_1
        rew_2_tens[z] = full_rew_2
        df = pd.DataFrame(rew_1_tens[z])
        df.to_csv(datastring.replace('THRESHOLD', 'rew_1_THRESHOLD'))
        df = pd.DataFrame(rew_2_tens[z])
        df.to_csv(datastring.replace('THRESHOLD', 'rew_2_THRESHOLD'))

    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    labels_dict = {1.5: 'CD-TSMG', 3: 'TSMDP', 4: 'TSMDP-UCRL'}
    leg = [Line2D([0,1],[0,1],linestyle=lsty[thr],
      color='black') for thr in thresholds]
    plt.legend(leg, [labels_dict[thr] for thr in thresholds])
    figst = 'cd_results/' + game_id + '/' + id_string.replace('.','-') + '.png'
    plt.savefig(figst)
    plt.show()
    
    for z, thr in enumerate(thresholds):
        for i in range(N):
            plt.plot(np.cumsum(rew_1_tens[z,i]), color='blue', alpha=0.1,
              linestyle=lsty[thr])
            plt.plot(np.cumsum(rew_2_tens[z,i]), color='orange', alpha=0.1,
              linestyle=lsty[thr])
        plt.plot(np.cumsum(np.mean(rew_1_tens[z]), axis=0), color='blue',
          linestyle=lsty[thr])
        plt.plot(np.cumsum(np.mean(rew_2_tens[z]), axis=0), color='orange',
          linestyle=lsty[thr])
    plt.xlabel('Time')
    plt.ylabel('Cumulative Reward')
    
    leg = [Line2D([0,1],[0,1],linestyle=lsty[thr],
      color='black') for thr in thresholds]
    plt.legend(leg, [labels_dict[thr] for thr in thresholds])
    figst = 'cd_results/' + game_id + '/' + id_string.replace('.','-') 
    figst += '_rewardplot.png'
    plt.savefig(figst)
    plt.show()
    
#%%
    
'''
Plotting results
'''
    
    
def plot_comparison(game_id, folder_1, folder_2, n, T=int(1e5), save=False,
                    thresholds=[], num_taus=10, scaled=False):
    '''
    Comparing nonparametric and parametric TSMG with baselines
    
    folder_2 is parametric
    '''
    color_dict = {1.5: 'blue', 3: 'red', 4: 'green', 0: 'purple'}
    thr_strings = [str(i).replace('.','-') for i in thresholds]
    dstring_1 = 'cd_results/' + game_id + '/' + folder_1
    dstring_2 = 'cd_results/' + game_id + '/' + folder_2
    files_1 = os.listdir(dstring_1)
    files_2 = os.listdir(dstring_2)
    if scaled:
        # Optional, lets you check if regret scaling appears faster, slower,
        # or equal to O((T*log(T))**(1/2))
        Tees = np.arange(T) + 1
        Tees = np.sqrt(Tees*(np.log(Tees) + 0.01))
    else:
        Tees = np.ones(T)
    dfs = {}
    # Nonparametric first
    for i, thr_string in enumerate(thr_strings):
        dfs[thr_string] = []
        for fname in files_1:
            if fname.startswith('THRESHOLD_' + thr_string):
                data_arr = np.array(pd.read_csv(dstring_1 + '/' + fname,
                  index_col=0))
                dfs[thr_string].append(data_arr)
        dfs[thr_string] = dfs[thr_string][:num_taus]
        for j, data in enumerate(dfs[thr_string]):
            thr = thresholds[i]
            plt.plot(np.cumsum(np.mean(np.array(data), axis=0))/Tees,
              alpha=0.2, color=color_dict[thr])
        means = np.mean(np.array(dfs[thr_string]), axis=1)
        worst = np.max(np.cumsum(means, axis=1), axis=0)
        plt.plot(worst / Tees, color=color_dict[thr])
    dfs['0'] = []
    for fname in files_2:
        if fname.startswith('THRESHOLD_1-5'):
            data_arr = np.array(pd.read_csv(dstring_2 + '/' + fname,
              index_col=0))
            dfs['0'].append(data_arr)
    dfs['0'] = dfs['0'][:num_taus]        
    for j, data in enumerate(dfs['0']):
        plt.plot(np.cumsum(np.mean(np.array(data), axis=0))/Tees, alpha=0.2,
          color=color_dict[0])
    means = np.mean(np.array(dfs['0']), axis=1)
    worst = np.max(np.cumsum(means, axis=1), axis=0)
    plt.plot(worst / Tees, color=color_dict[0])
    labels_dict = {1.5: 'TSMG', 3: 'TSMDP', 4: 'R-TSMDP', 0: 'P-TSMG'}
    leg = [Line2D([0,1],[0,1],
      color=color_dict[thr]) for thr in thresholds + [0]]
    plt.legend(leg, [labels_dict[thr] for thr in thresholds + [0]])
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Cumulative Regret', fontsize=16)
    if save:
        plt.savefig(dstring_2 + '/' + save + '.png', bbox_inches='tight')
    plt.show()
    

def plot_selfplay(game_id, folder_def, folder_fav, n, save=False):
    '''
    Regret and rewards in self-play
    '''
    strings = ['def', 'fav']
    dstrings = {}
    files = {}
    dstrings['def'] = 'cd_results/' + game_id + '/' + folder_def
    dstrings['fav'] = 'cd_results/' + game_id + '/' + folder_fav
    files['def'] = os.listdir(dstrings['def'])
    files['fav'] = os.listdir(dstrings['fav'])
    lsty = {'def': 'solid', 'fav': 'dashed'}
    dfs = {}
    for i, dstring in enumerate(strings):
        dfs[dstring] = []
        for fname in files[dstring]:
            if fname.startswith('THRESHOLD'):
                data_arr = np.array(pd.read_csv(dstrings[dstring] + '/' + fname,
                  index_col=0))
                dfs[dstring].append(data_arr)
        dfs[dstring] = dfs[dstring][0]     
        for i in range(n):
            plt.plot(np.cumsum(dfs[dstring][i]), color='blue',
              linestyle=lsty[dstring], alpha=0.1)
        means = np.mean(dfs[dstring], axis=0)
        plt.plot(np.cumsum(means), color='blue', linestyle=lsty[dstring])
    labels_dict = {'def': 'Default Prior', 'fav': 'Alternative Prior'}
    leg = [Line2D([0,1],[0,1], linestyle=lsty[st],
      color='black') for st in strings]
    plt.legend(leg, [labels_dict[st] for st in strings])
    plt.xlabel('Time')
    plt.ylabel('Cumulative Regret')
    if save:
        plt.savefig('cd_results/' + game_id + '/' + game_id + '_sp_regret.png')
    plt.show()
    
    rew1 = {}
    rew2 = {}
    for i, dstring in enumerate(strings):
        rew1[dstring] = []
        rew2[dstring] = []
        for fname in files[dstring]:
            if fname.startswith('rew_1'):
                data_arr = np.array(pd.read_csv(dstrings[dstring] + '/' + fname,
                  index_col=0))
                rew1[dstring].append(data_arr)
            if fname.startswith('rew_2'):
                data_arr = np.array(pd.read_csv(dstrings[dstring] + '/' + fname,
                  index_col=0))
                rew2[dstring].append(data_arr)
        rew1[dstring] = rew1[dstring][0]
        rew2[dstring] = rew2[dstring][0]
        for i in range(n):
            plt.plot(np.cumsum(rew1[dstring][i]), color='blue',
              linestyle=lsty[dstring], alpha=0.1)
            plt.plot(np.cumsum(rew2[dstring][i]), color='orange',
              linestyle=lsty[dstring], alpha=0.1)
        means1 = np.mean(rew1[dstring], axis=0)
        means2 = np.mean(rew2[dstring], axis=0)
        plt.plot(np.cumsum(means1), color='blue', linestyle=lsty[dstring])
        plt.plot(np.cumsum(means2), color='orange', linestyle=lsty[dstring])
    leg1 = [Line2D([0,1],[0,1], linestyle=lsty[st],
      color='black') for st in strings]
    leg2 = [Line2D([0,1],[0,1], color='blue'), Line2D([0,1],[0,1],
      color='orange')]
    plt.legend(leg1 + leg2,
      [labels_dict[st] for st in strings] + ['Player 1', 'Player 2'])
    plt.xlabel('Time')
    plt.ylabel('Cumulative Reward')
    if save:
        plt.savefig('cd_results/' + game_id + '/' + game_id + '_sp_rewards.png')
    plt.show()    

    
#%%

if __name__ == "__main__":
    
    # Suppressing error messages
    logger = logging.getLogger('pymc3')
    logger.setLevel(logging.ERROR)

    fullgame = {'policies': {},
                'geom_states': {},
                'states': {},
                'actions': {},
                'rewards_1': {},
                'rewards_2': {},
                'base_1': {},
                'base_2': {},
                'TP': {},
                'init_state':{},
                's':{},
                's0':{}}
    
    pi_function = for_gen_pi2 if game_id == 'g' else lin_pi2
    # assumes THETA_DIM == 4
    
    # Parametric experiments were run after we made some edits to the
    # following workflow, so some blocks of code check for PARAMETRIC=True
    
    if not PARAMETRIC:
        fullgame['policies']['1'] = base_policies_1(D)[[0,2,3,5][:THETA_DIM]]
    
    # Grid
    fullgame['policies']['2'] = base_policies_gr(D)[[0,2,3,5][:THETA_DIM]]
    
    if not PARAMETRIC:
        (fullgame['geom_states']['1'], fullgame['states']['1'],
          fullgame['actions']['1'], fullgame['rewards_1']['1'],
          fullgame['rewards_2']['1'], fullgame['TP']['1'],
          fullgame['init_state']['1']) = grid_1(D)
    (fullgame['geom_states']['2'], fullgame['states']['2'],
     fullgame['actions']['2'], fullgame['rewards_1']['2'],
     fullgame['rewards_2']['2'], fullgame['TP']['2'],
     fullgame['init_state']['2']) = grid(D)
    
    np.random.seed(3)
    t = 0
    fullgame['s0']['2'] = 42
    
    # Prisoner's Dilemma
    fullgame['init_state']['p'] = 0
    pd_pols = base_policies_ipd(fullgame['init_state']['p'])
    fullgame['policies']['p'] = pd_pols[:THETA_DIM]
    base_1 = np.array([[3, 0],[4, 1]])
    base_2 = np.array([[3, 4],[0, 1]])
    (fullgame['geom_states']['p'], fullgame['states']['p'],
      fullgame['actions']['p'], fullgame['rewards_1']['p'],
      fullgame['rewards_2']['p'], fullgame['TP']['p'],
      fullgame['init_state']['p']) = iterated_game(base_1, base_2, 2, 0)
    fullgame['s0']['p'] = 0
 
    # Bach-or-Stravinsky
    fullgame['init_state']['b'] = 0
    bs_pols = base_policies_bos(fullgame['init_state']['b'])
    fullgame['policies']['b'] = bs_pols[:THETA_DIM]
    base_1 = np.array([[1, 0],[0, 2]])
    base_2 = np.array([[2, 0],[0, 1]])
    (fullgame['geom_states']['b'], fullgame['states']['b'],
      fullgame['actions']['b'], fullgame['rewards_1']['b'],
      fullgame['rewards_2']['b'], fullgame['TP']['b'],
      fullgame['init_state']['b']) = iterated_game(base_1, base_2, 2, 0)
    fullgame['s0']['b'] = 0 
    
    # BOS+PD
    bospd_1 = np.array([[3.5,0,-3],[0,1,-3],[2,2,-1]])
    bospd_2 = np.array([[1,0,2],[0,3,2],[-3,-3,-1]])
    (fullgame['geom_states']['g'], fullgame['states']['g'],
      fullgame['actions']['g'], fullgame['rewards_1']['g'],
      fullgame['rewards_2']['g'], fullgame['TP']['g'],
      fullgame['init_state']['g']) = iterated_game(bospd_1, bospd_2, 1, 0)
    fullgame['base_1']['g'] = bospd_1
    fullgame['base_2']['g'] = bospd_2
    fullgame['s0']['g'] = 0
    
    #%%
    
    '''
    Experiment
    '''
    
    if game_id == 'p':
        # This change was a path-dependency of the experiments
        np.random.seed(SEED_NUMBER - 2)
    M = 6 #ell in the paper
    
    # Reset schedule for UCRL2 variant
    RESET_LIST = []
    i = 0
    time = ceil((i**3)/(M**2))
    while time < T:
        RESET_LIST.append(time)
        i += 1
        time = ceil((i**3)/(M**2))
        
    # Constructing theta sequence
    true_thetas = []
    if game_id == 'g':
        th_list = [np.array([1, 0, 0, 0][:THETA_DIM]),
                       np.array([1, 10, 0, 0][:THETA_DIM]),
                       np.array([1, 10, 1, 0][:THETA_DIM]),
                       np.array([1, 10, 1, 5][:THETA_DIM]),
                       np.array([1, 5, 5, 0][:THETA_DIM]),
                       np.array([0, 10, 0, 0][:THETA_DIM])]
        true_thetas = []
        for i in range(M):
            true_thetas.append(th_list[i % 6])
    else:
        for i in range(M):
            th = np.ones(THETA_DIM)*0.05
            th[i % THETA_DIM] += 1 - THETA_DIM*0.05
            true_thetas.append(th)
    
    # Schedule of switch times
    nu_schedule_list = []
    for _ in range(CT_SEEDS):
        spaced = False
        while not spaced:
            pts = np.random.rand(M-1)
            pts.sort()
            arr = np.round(pts*T)
            if np.min(arr[1:] - arr[:-1]) > 2*L:
                spaced = True
        for ell in ELLS:
            nu_schedule = [0]
            nu_schedule += [list(arr)[int(round(i))] for i in np.linspace(0,
              len(arr)-1, ell-1)] + [T]
            nu_schedule_list.append(nu_schedule.copy())
    
    print('Running test...')
    if SELF_PLAY_TEST:
        self_play(N, T, L, fullgame, game_id, MET, ndraws, nburn, gam=0.99,
          tol=0.001, verbose=verbose, pi_function=pi_function,
          parametric=PARAMETRIC)
    else:
        experiment(N, T, L, fullgame, game_id, true_thetas, nu_schedule_list,
          MET, ndraws, nburn, gam=0.9, tol=0.001, verbose=verbose,
          pi_function=pi_function, parametric=PARAMETRIC)
