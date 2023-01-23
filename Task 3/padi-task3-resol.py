#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 3: Partially observable Markov decision problems
# 
# In the end of the lab, you should export the notebook to a Python script (File >> Download as >> Python (.py)). Your file should be named `padi-lab3-groupXX.py`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
# 
# Make sure...
# 
# * **... that the subject is of the form `[<group n.>] LAB <lab n.>`.** 
# 
# * **... to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.** 
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).
# 
# ### 1. The POMDP model
# 
# Consider once again the Pacman modeling problem described in the Homework and for which you wrote a Markov decision problem model. In this lab, you will consider a larger version of the Pacman problem, described by the diagram:
# 
# <img src="pacman-big.png">
# 
# Recall that the POMDP should describe the decision-making of a player. In the above domain,
# 
# * The ghost **moves randomly between cells 1-3**.
# * The player controls the movement of Pacman through four actions: `Up`, `Down`, `Left`, and `Right`. 
# * Each action moves the Pacman character one step in the corresponding direction, if an adjacent cell exists in that direction. Otherwise, Pacman remains in the same place.
# * The cell in the bottom left corner (cell `29`) is adjacent, to the left, to the cell in the bottom right corner (cell `35`). In other words, if Pacman "moves left" in cell `29` it will end up in cell `35` and vice-versa.
# * If Pacman lies in the same cell as the ghost (in either cell `1`, `2`, or `3`), the player loses the game. However, if Pacman "eats" the blue pellet (in cell `24`), it gains the ability to "eat" the ghost. In this case, if Pacman lies in the same cell as the ghost, it "eats" the ghost and wins the game. Assume that Pacman can never be in cell `24` without "eating" the pellet.
# * Pacman is unable to see the ghost unless if it stands in the same position as the ghost (however, it does know its own position and whether it ate the pellet or not).
# 
# In this lab you will use a POMDP based on the aforementioned domain and investigate how to simulate a partially observable Markov decision problem and track its state. You will also compare different MDP heuristics with the optimal POMDP solution.
# 
# **Throughout the lab, unless if stated otherwise, use $\gamma=0.9$.**
# 
# $$\diamond$$
# 
# In this first activity, you will implement an POMDP model in Python. You will start by loading the POMDP information from a `numpy` binary file, using the `numpy` function `load`. The file contains the list of states, actions, observations, transition probability matrices, observation probability matrices, and cost function.

# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_pomdp` that receives, as input, a string corresponding to the name of the file with the POMDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file contains 6 arrays:
# 
# * An array `X` that contains all the states in the POMDP, represented as strings. In the Pacman environment above, for example, there is a total of 209 states, each describing the position of Pacman in the environment, whether it has eaten the blue pellet, and the position of the ghost. Those states are either one of the strings `"V"` or `"D"`, corresponding to the absorbing "victory" and "defeat" states, or a string of the form `"(p, s, g)"`, where:
#     * `p` is a number between 1 and 35 indicating the position of Pacman;
#     * `s` is either `0` or `S`, where `0` indicates that Pacman has not yet eaten the pellet; `S` indicates that Pacman has eaten the pellet (and now has "superpowers");
#     * `g` is a number between 1 and 3, indicating the position of the ghost.
# * An array `A` that contains all the actions in the MDP, also represented as strings. In the Pacman environment above, for example, each action is represented as a string `"Up"`, `"Down"`, `"Left"` or `"Right"`.
# * An array `Z` that contains all the observations in the POMDP, also represented as strings. In the Pacman environment above, for example, there is a total of 77 observations, each describing the position of Pacman in the environment, whether it has eaten the blue pellet, and whether it sees the ghost. It also observes the victory and defeat states. This means that the strings are either `"V"` or `"D"`, corresponding to the "victory" and "defeat" states, or a string of the form `"(p, s, g)"`, where:
#     * `p` is a number between 1 and 35 indicating the position of Pacman;
#     * `s` is either `0` or `S`, where `0` indicates that Pacman has not yet eaten the pellet; `S` indicates that Pacman has eaten the pellet (and now has "superpowers");
#     * `g` is a number between 0 and 3, 0 indicating that the ghost is not seen, and the numbers between 1 and 3 indicates the position of the ghost (when visible).
# * An array `P` containing `len(A)` subarrays, each with dimension `len(X)` &times; `len(X)` and  corresponding to the transition probability matrix for one action.
# * An array `O` containing `len(A)` subarrays, each with dimension `len(X)` &times; `len(Z)` and  corresponding to the observation probability matrix for one action.
# * An array `c` containing the cost function for the POMDP.
# 
# Your function should create the POMDP as a tuple `(X, A, Z, (Pa, a = 0, ..., len(A)), (Oa, a = 0, ..., len(A)), c, g)`, where `X` is a tuple containing the states in the POMDP represented as strings (see above), `A` is a tuple containing the actions in the POMDP represented as strings (see above), `Z` is a tuple containing the observations in the POMDP represented as strings (see above), `P` is a tuple with `len(A)` elements, where `P[a]` is an `np.array` corresponding to the transition probability matrix for action `a`, `O` is a tuple with `len(A)` elements, where `O[a]` is an `np.array` corresponding to the observation probability matrix for action `a`, `c` is an `np.array` corresponding to the cost function for the POMDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the POMDP tuple.
# 
# ---

# In[105]:


import numpy as np

def load_pomdp(MDP, gamma):
    if gamma <0 or gamma >1:
        raise ValueError('The value of gamma is not between  0  and  1')
    M = np.load(MDP)
    X = tuple(M['X']) 
    A = tuple(M['A'])
    Z = tuple(M['Z'])
    P = tuple(M['P'])
    O = tuple(M['O'])
    c = M['c']
    return (X, A, Z, P, O, c, gamma)


# We provide below an example of application of the function with the file `pacman.npz` that you can use as a first "sanity check" for your code. Note that, even fixing the seed, the results you obtain may slightly differ.
# 
# ```python
# import numpy.random as rand
# 
# M = load_pomdp('pacman.npz', 0.9)
# 
# rand.seed(42)
# 
# # States
# print('= State space (%i states) =' % len(M[0]))
# print('\nStates:')
# for i in range(min(10, len(M[0]))):
#     print(M[0][i]) 
# 
# print('...')
# 
# # Random state
# s = rand.randint(len(M[0]))
# print('\nRandom state: s =', M[0][s])
# 
# # Last state
# print('\nLast state:', M[0][-1])
# 
# # Actions
# print('= Action space (%i actions) =' % len(M[1]))
# for i in range(len(M[1])):
#     print(M[1][i]) 
# 
# # Random action
# a = rand.randint(len(M[1]))
# print('\nRandom action: a =', M[1][a])
# 
# # Observations
# print('= Observation space (%i observations) =' % len(M[2]))
# print('\nObservations:')
# for i in range(min(10, len(M[2]))):
#     print(M[2][i]) 
# 
# print('...')
# 
# # Random observation
# z = rand.randint(len(M[2]))
# print('\nRandom observation: z =', M[2][z])
# 
# # Last state
# print('\nLast observation:', M[2][-1])
# 
# # Transition probabilities
# print('\n= Transition probabilities =')
# 
# for i in range(len(M[1])):
#     print('\nTransition probability matrix dimensions (action %s):' % M[1][i], M[3][i].shape)
#     print('Dimensions add up for action "%s"?' % M[1][i], np.isclose(np.sum(M[3][i]), len(M[0])))
#     
# print('\nState-action pair (%s, %s) transitions to state(s)' % (M[0][s], M[1][a]))
# print("s' in", np.array(M[0])[np.where(M[3][a][s, :] > 0)])
# 
# # Observation probabilities
# print('\n= Observation probabilities =')
# 
# for i in range(len(M[1])):
#     print('\nObservation probability matrix dimensions (action %s):' % M[1][i], M[4][i].shape)
#     print('Dimensions add up for action "%s"?' % M[1][i], np.isclose(np.sum(M[4][i]), len(M[0])))
#     
# print('\nState-action pair (%s, %s) yields observation(s)' % (M[0][s], M[1][a]))
# print("z in", np.array(M[0])[np.where(M[4][a][s, :] > 0)])
# 
# # Cost
# print('\n= Costs =')
# 
# print('\nSpecial states with cost different from 0.1:')
# print(np.array(M[0])[np.where(M[5][:, 0] != 0.1)])
# print('Associated costs:')
# print(M[5][np.where(M[5][:, 0] != 0.1), 0])
# 
# print('\nCost for the state-action pair (%s, %s):' % (M[0][s], M[1][a]))
# print('c(s, a) =', M[5][s, a])
# 
# 
# # Discount
# print('\n= Discount =')
# print('\ngamma =', M[6])
# ```
# 
# Output:
# 
# ```
# = State space (209 states) =
# 
# States:
# (1, S, 1)
# (1, S, 2)
# (1, S, 3)
# (1, 0, 1)
# (1, 0, 2)
# (1, 0, 3)
# (2, S, 1)
# (2, S, 2)
# (2, S, 3)
# (2, 0, 1)
# ...
# 
# Random state: s = (18, S, 1)
# 
# Last state: D
# = Action space (4 actions) =
# Up
# Down
# Left
# Right
# 
# Random action: a = Right
# = Observation space (77 observations) =
# 
# Observations:
# (1, S, 1)
# (1, S, 0)
# (1, 0, 1)
# (1, 0, 0)
# (2, S, 0)
# (2, S, 2)
# (2, 0, 0)
# (2, 0, 2)
# (3, S, 0)
# (3, S, 3)
# ...
# 
# Random observation: z = (5, S, 0)
# 
# Last observation: D
# 
# = Transition probabilities =
# 
# Transition probability matrix dimensions (action Up): (209, 209)
# Dimensions add up for action "Up"? True
# 
# Transition probability matrix dimensions (action Down): (209, 209)
# Dimensions add up for action "Down"? True
# 
# Transition probability matrix dimensions (action Left): (209, 209)
# Dimensions add up for action "Left"? True
# 
# Transition probability matrix dimensions (action Right): (209, 209)
# Dimensions add up for action "Right"? True
# 
# State-action pair ((18, S, 1), Right) transitions to state(s)
# s' in ['(19, S, 2)']
# 
# = Observation probabilities =
# 
# Observation probability matrix dimensions (action Up): (209, 77)
# Dimensions add up for action "Up"? True
# 
# Observation probability matrix dimensions (action Down): (209, 77)
# Dimensions add up for action "Down"? True
# 
# Observation probability matrix dimensions (action Left): (209, 77)
# Dimensions add up for action "Left"? True
# 
# Observation probability matrix dimensions (action Right): (209, 77)
# Dimensions add up for action "Right"? True
# 
# State-action pair ((18, S, 1), Right) yields observation(s)
# z in ['(7, 0, 2)']
# 
# = Costs =
# 
# Special states with cost different from 0.1:
# ['(1, S, 1)' '(1, 0, 1)' '(2, S, 2)' '(2, 0, 2)' '(3, S, 3)' '(3, 0, 3)'
#  'V' 'D']
# Associated costs:
# [[0. 1. 0. 1. 0. 1. 0. 0.]]
# 
# Cost for the state-action pair ((18, S, 1), Right):
# c(s, a) = 0.1
# 
# = Discount =
# 
# gamma = 0.9
# ```

# In[121]:


#test sanity

import numpy.random as rand

M = load_pomdp('pacman.npz', 0.9)

rand.seed(42)

# States
print('= State space (%i states) =' % len(M[0]))
print('\nStates:')
for i in range(min(10, len(M[0]))):
    print(M[0][i]) 

print('...')

# Random state
s = rand.randint(len(M[0]))
print('\nRandom state: s =', M[0][s])

# Last state
print('\nLast state:', M[0][-1])

# Actions
print('= Action space (%i actions) =' % len(M[1]))
for i in range(len(M[1])):
    print(M[1][i]) 

# Random action
a = rand.randint(len(M[1]))
print('\nRandom action: a =', M[1][a])

# Observations
print('= Observation space (%i observations) =' % len(M[2]))
print('\nObservations:')
for i in range(min(10, len(M[2]))):
    print(M[2][i]) 

print('...')

# Random observation
z = rand.randint(len(M[2]))
print('\nRandom observation: z =', M[2][z])

# Last state
print('\nLast observation:', M[2][-1])

# Transition probabilities
print('\n= Transition probabilities =')

for i in range(len(M[1])):
    print('\nTransition probability matrix dimensions (action %s):' % M[1][i], M[3][i].shape)
    print('Dimensions add up for action "%s"?' % M[1][i], np.isclose(np.sum(M[3][i]), len(M[0])))

print('\nState-action pair (%s, %s) transitions to state(s)' % (M[0][s], M[1][a]))
print("s' in", np.array(M[0])[np.where(M[3][a][s, :] > 0)])

# Observation probabilities
print('\n= Observation probabilities =')

for i in range(len(M[1])):
    print('\nObservation probability matrix dimensions (action %s):' % M[1][i], M[4][i].shape)
    print('Dimensions add up for action "%s"?' % M[1][i], np.isclose(np.sum(M[4][i]), len(M[0])))

print('\nState-action pair (%s, %s) yields observation(s)' % (M[0][s], M[1][a]))
print("z in", np.array(M[0])[np.where(M[4][a][s, :] > 0)])

# Cost
print('\n= Costs =')

print('\nSpecial states with cost different from 0.1:')
print(np.array(M[0])[np.where(M[5][:, 0] != 0.1)])
print('Associated costs:')
print(M[5][np.where(M[5][:, 0] != 0.1), 0])

print('\nCost for the state-action pair (%s, %s):' % (M[0][s], M[1][a]))
print('c(s, a) =', M[5][s, a])


# Discount
print('\n= Discount =')
print('\ngamma =', M[6])


# ### 2. Sampling
# 
# You are now going to sample random trajectories of your POMDP and observe the impact it has on the corresponding belief.

# ---
# 
# #### Activity 2.
# 
# Write a function called `gen_trajectory` that generates a random POMDP trajectory using a uniformly random policy. Your function should receive, as input, a POMDP described as a tuple like that from **Activity 1** and two integers, `x0` and `n` and return a tuple with 3 elements, where:
# 
# 1. The first element is a `numpy` array corresponding to a sequence of `n+1` state indices, $x_0,x_1,\ldots,x_n$, visited by the agent when following a uniform policy (i.e., a policy where actions are selected uniformly at random) from state with index `x0`. In other words, you should select $x_1$ from $x_0$ using a random action; then $x_2$ from $x_1$, etc.
# 2. The second element is a `numpy` array corresponding to the sequence of `n` action indices, $a_0,\ldots,a_{n-1}$, used in the generation of the trajectory in 1.;
# * The third element is a `numpy` array corresponding to the sequence of `n` observation indices, $z_1,\ldots,z_n$, experienced by the agent during the trajectory in 1.
# 
# The `numpy` array in 1. should have a shape `(n+1,)`; the `numpy` arrays from 2. and 3. should have a shape `(n,)`.
# 
# **Note:** Your function should work for **any** POMDP specified as above.
# 
# ---

# In[107]:


import numpy.random as rand

def gen_trajectory(M, x0, n):
    n_states = len(M[0]) 
    n_actions = len(M[1])
    n_observations = len(M[2]) 
    seq_states = np.array([x0]) 
    seq_actions = np.array([])  
    seq_observations = np.array([]) 
    observation_prob = np.array(M[4])
    prob = np.array(M[3])  
    state = x0
    for i in range(n):
        new_action = rand.choice(range(n_actions)) 
        seq_actions = np.append(seq_actions, new_action) 
        state = rand.choice(range(n_states), p = prob[new_action][state]) 
        seq_states = np.append(seq_states,state) 
        observation = rand.choice(range(n_observations), p = observation_prob[new_action][state]) 
        seq_observations = np.append(seq_observations,observation) 
    return (seq_states, seq_actions.astype(int), seq_observations.astype(int))


# For example, using the POMDP from **Activity 1** you could obtain the following interaction.
# 
# ```python
# rand.seed(42)
# 
# # Number of steps and initial state
# steps = 10
# s0    = 106 # State (18, 0, 2)
# 
# # Generate trajectory
# t = gen_trajectory(M, s0, steps)
# 
# # Check shapes
# print('Shape of state trajectory:', t[0].shape)
# print('Shape of state trajectory:', t[1].shape)
# print('Shape of state trajectory:', t[2].shape)
# 
# # Print trajectory
# for i in range(steps):
#     print('\n- Time step %i -' % i)
#     print('State:', M[0][t[0][i]], '(state %i)' % t[0][i])
#     print('Action selected:', M[1][t[1][i]], '(action %i)' % t[1][i])
#     print('Resulting state:', M[0][t[0][i+1]], '(state %i)' % t[0][i+1])
#     print('Observation:', M[2][t[2][i]], '(observation %i)' % t[2][i])
# ```
# 
# Output:
# 
# ```
# Shape of state trajectory: (11,)
# Shape of state trajectory: (10,)
# Shape of state trajectory: (10,)
# 
# - Time step 0 -
# State: (18, 0, 2) (state 106)
# Action selected: Left (action 2)
# Resulting state: (17, 0, 3) (state 101)
# Observation: (17, 0, 0) (observation 39)
# 
# - Time step 1 -
# State: (17, 0, 3) (state 101)
# Action selected: Right (action 3)
# Resulting state: (18, 0, 2) (state 106)
# Observation: (18, 0, 0) (observation 41)
# 
# - Time step 2 -
# State: (18, 0, 2) (state 106)
# Action selected: Left (action 2)
# Resulting state: (17, 0, 1) (state 99)
# Observation: (17, 0, 0) (observation 39)
# 
# - Time step 3 -
# State: (17, 0, 1) (state 99)
# Action selected: Up (action 0)
# Resulting state: (10, 0, 2) (state 58)
# Observation: (10, 0, 0) (observation 25)
# 
# - Time step 4 -
# State: (10, 0, 2) (state 58)
# Action selected: Down (action 1)
# Resulting state: (17, 0, 1) (state 99)
# Observation: (17, 0, 0) (observation 39)
# 
# - Time step 5 -
# State: (17, 0, 1) (state 99)
# Action selected: Down (action 1)
# Resulting state: (17, 0, 2) (state 100)
# Observation: (17, 0, 0) (observation 39)
# 
# - Time step 6 -
# State: (17, 0, 2) (state 100)
# Action selected: Up (action 0)
# Resulting state: (10, 0, 3) (state 59)
# Observation: (10, 0, 0) (observation 25)
# 
# - Time step 7 -
# State: (10, 0, 3) (state 59)
# Action selected: Up (action 0)
# Resulting state: (10, 0, 2) (state 58)
# Observation: (10, 0, 0) (observation 25)
# 
# - Time step 8 -
# State: (10, 0, 2) (state 58)
# Action selected: Left (action 2)
# Resulting state: (10, 0, 1) (state 57)
# Observation: (10, 0, 0) (observation 25)
# 
# - Time step 9 -
# State: (10, 0, 1) (state 57)
# Action selected: Right (action 3)
# Resulting state: (11, 0, 2) (state 64)
# Observation: (11, 0, 0) (observation 27)
# ```

# In[108]:


#test sanity
rand.seed(42)

# Number of steps and initial state
steps = 10
s0    = 106 # State (18, 0, 2)

# Generate trajectory
t = gen_trajectory(M, s0, steps)

# Check shapes
print('Shape of state trajectory:', t[0].shape)
print('Shape of state trajectory:', t[1].shape)
print('Shape of state trajectory:', t[2].shape)

# Print trajectory
for i in range(steps):
    print('\n- Time step %i -' % i)
    print('State:', M[0][t[0][i]], '(state %i)' % t[0][i])
    print('Action selected:', M[1][t[1][i]], '(action %i)' % t[1][i])
    print('Resulting state:', M[0][t[0][i+1]], '(state %i)' % t[0][i+1])
    print('Observation:', M[2][t[2][i]], '(observation %i)' % t[2][i])


# You will now write a function that samples a given number of possible belief points for a POMDP. To do that, you will use the function from **Activity 2**.
# 
# ---
# 
# #### Activity 3.
# 
# Write a function called `sample_beliefs` that receives, as input, a POMDP described as a tuple like that from **Activity 1** and an integer `n`, and return a tuple with `n+1` elements **or less**, each corresponding to a possible belief state (represented as a $1\times|\mathcal{X}|$ vector). To do so, your function should
# 
# * Generate a trajectory with `n` steps from a random initial state, using the function `gen_trajectory` from **Activity 2**.
# * For the generated trajectory, compute the corresponding sequence of beliefs, assuming that the agent does not know its initial state (i.e., the initial belief is the uniform belief, and should also be considered). 
# 
# Your function should return a tuple with the resulting beliefs, **ignoring duplicate beliefs or beliefs whose distance is smaller than $10^{-3}$.**
# 
# **Suggestion:** You may want to define an auxiliary function `belief_update` that receives a POMDP, a belief, an action and an observation and returns the updated belief.
# 
# **Note:** Your function should work for **any** POMDP specified as above. To compute the distance between vectors, you may find useful `numpy`'s function `linalg.norm`.
# 
# 
# ---

# In[109]:


def belief_update(belief, action, observation, P, O):
    a = np.dot(np.dot(belief, P[action]), np.diag(O[action][:,observation]))
    b = np.sum(np.dot(np.dot(belief, P[action]), np.diag(O[action][:,observation])))
    return a/b

def sample_beliefs(M,n):
    n_states = len(M[0])
    prob = np.array(M[3]) 
    observations_prob = np.array(M[4]) 
    init_state = int(np.random.choice(range(n_states)))
    belief = np.ones((1,n_states))/n_states
    belief_vec = [] 
    belief_vec.append(belief)
    traj = gen_trajectory(M, init_state, n) 
    for i in range(n):
        quit = True
        belief = belief_update(belief, traj[1][i], traj[2][i], prob, observations_prob) 
        for j in range(len(belief_vec)):
            if np.linalg.norm(belief - belief_vec[j]) < 10**(-3): 
                quit=False
        if quit: 
            belief_vec.append(belief)
    return tuple(belief_vec) 


# For example, using the POMDP from **Activity 1** you could obtain the following interaction.
# 
# ```python
# rand.seed(42)
# 
# # 3 sample beliefs + initial belief
# B = sample_beliefs(M, 3)
# print('%i beliefs sampled:' % len(B))
# for i in range(len(B)):
#     print(np.round(B[i], 3))
#     print('Belief adds to 1?', np.isclose(B[i].sum(), 1.))
# 
# # 100 sample beliefs
# B = sample_beliefs(M, 100)
# print('%i beliefs sampled.' % len(B))
# ```
# 
# Output:
# 
# ```
# 4 beliefs sampled:
# [[0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005
#   0.005 0.005 0.005 0.005 0.005]]
# Belief adds to 1? True
# [[0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.167 0.667 0.167 0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.   ]]
# Belief adds to 1? True
# [[0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.333 0.333 0.333 0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.   ]]
# Belief adds to 1? True
# [[0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.167 0.667 0.167 0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.    0.
#   0.    0.    0.    0.    0.   ]]
# Belief adds to 1? True
# 25 beliefs sampled.
# ```

# In[110]:


#test sanity
rand.seed(42)

# 3 sample beliefs + initial belief
B = sample_beliefs(M, 3)
print('%i beliefs sampled:' % len(B))
for i in range(len(B)):
    print(np.round(B[i], 3))
    print('Belief adds to 1?', np.isclose(B[i].sum(), 1.))

# 100 sample beliefs
B = sample_beliefs(M, 100)
print('%i beliefs sampled.' % len(B))


# ### 3. Solution methods
# 
# In this section you are going to compare different solution methods for POMDPs discussed in class.

# ---
# 
# #### Activity 4
# 
# Write a function `solve_mdp` that takes as input a POMDP represented as a tuple like that of **Activity 1** and returns a `numpy` array corresponding to the **optimal $Q$-function for the underlying MDP**. Stop the algorithm when the error between iterations is smaller than $10^{-8}$.
# 
# **Note:** Your function should work for **any** POMDP specified as above. You may reuse code from previous labs.
# 
# ---

# In[111]:


def solve_mdp(M): 
    n_states = len(M[0])
    n_actions = len(M[1]) 
    prob = np.array(M[3])
    c = np.array(M[5]) 
    gamma = M[6] 
    Q = np.ones((n_states,n_actions))
    Q_new = np.array(Q)
    err = 1   
    while err >= 10**(-8):
        for j in range(len(c[0])): 
            J = np.min(Q, axis=1) 
            cj=np.reshape(c[:,j],(n_states,1))
            Pj=np.reshape(prob[j,:].dot(J), (n_states,1))
            Q[:,j,None]= cj + gamma*Pj
        err = np.linalg.norm(Q - Q_new) 
        Q_new = np.array(Q) 
    return Q_new


# As an example, you can run the following code on the POMDP from **Activity 1**.
# 
# ```python
# Q = solve_mdp(M)
# 
# s = 106 # State (18, 0, 2)
# print('\nQ-values at state %s:' % M[0][s], np.round(Q[s, :], 3))
# print('Best action at state %s:' % M[0][s], M[1][np.argmin(Q[s, :])])
# 
# s = 12 # State (3, S, 1)
# print('\nQ-values at state %s:' % M[0][s], np.round(Q[s, :], 3))
# print('Best action at state %s:' % M[0][s], M[1][np.argmin(Q[s, :])])
# 
# s = 164 # State (28, 0, 3)
# print('\nQ-values at state %s:' % M[0][s], np.round(Q[s, :], 3))
# print('Best action at state %s:' % M[0][s], M[1][np.argmin(Q[s, :])])
# ```
# 
# Output:
# 
# ```
# Q-values at state (18, 0, 2): [0.804 0.804 0.804 0.759]
# Best action at state (18, 0, 2): Right
# 
# Q-values at state (3, S, 1): [0.231 0.231 0.1   0.231]
# Best action at state (3, S, 1): Left
# 
# Q-values at state (28, 0, 3): [0.669 0.732 0.732 0.732]
# Best action at state (28, 0, 3): Up
# ```

# In[112]:


#test sanity
Q = solve_mdp(M)

s = 106 # State (18, 0, 2)
print('\nQ-values at state %s:' % M[0][s], np.round(Q[s, :], 3))
print('Best action at state %s:' % M[0][s], M[1][np.argmin(Q[s, :])])

s = 12 # State (3, S, 1)
print('\nQ-values at state %s:' % M[0][s], np.round(Q[s, :], 3))
print('Best action at state %s:' % M[0][s], M[1][np.argmin(Q[s, :])])

s = 164 # State (28, 0, 3)
print('\nQ-values at state %s:' % M[0][s], np.round(Q[s, :], 3))
print('Best action at state %s:' % M[0][s], M[1][np.argmin(Q[s, :])])


# ---
# 
# #### Activity 5
# 
# You will now test the different MDP heuristics discussed in class. To that purpose, write down a function that, given a belief vector and the solution for the underlying MDP, computes the action prescribed by each of the three MDP heuristics. In particular, you should write down a function named `get_heuristic_action` that receives, as inputs:
# 
# * A belief state represented as a `numpy` array like those of **Activity 3**;
# * The optimal $Q$-function for an MDP (computed, for example, using the function `solve_mdp` from **Activity 4**);
# * A string that can be either `"mls"`, `"av"`, or `"q-mdp"`;
# 
# Your function should return an integer corresponding to the index of the action prescribed by the heuristic indicated by the corresponding string, i.e., the most likely state heuristic for `"mls"`, the action voting heuristic for `"av"`, and the $Q$-MDP heuristic for `"q-mdp"`. *In all heuristics, ties should be broken randomly, i.e., when maximizing/minimizing, you should randomly select between all maximizers/minimizers*.
# 
# ---

# In[117]:


import random

def get_heuristic_action(belief, Q_opt, string):
    n_states = Q_opt.shape[0]
    n_actions = Q_opt.shape[1]
    pi_opt = np.ones((n_states,1))
    actions_voting = {}
    weight_sum = []
    for i in range(n_states):
        minimum= np.min(Q_opt[i][:])
        min_actions= np.where(np.isclose(Q_opt[i][:], minimum, atol=1e-10,rtol=1e-10))[0]
        pi_opt[i] = random.choice(min_actions)
    if string == "mls":
        pi = int(pi_opt[np.argmax(belief)])
    elif string =="av":
        for j in range(len(belief.T)):
            if int(pi_opt[j]) not in actions_voting.keys():
                actions_voting[int(pi_opt[j])]=1
            else:
                actions_voting[int(pi_opt[j])]+=1
        pi = [key for key, value in actions_voting.items() if value == max(actions_voting.values())][0]
        
    elif string =="q-mdp":
        for a in range(n_actions):
            sum_states=0
            for x in range(n_states):
                sum_states = sum_states + belief.T[x]*Q_opt[x][a]
            weight_sum.append(sum_states[0])
        pi = weight_sum.index(min(weight_sum))
    return pi 
   


# For example, if you run your function in the examples from **Activity 3** using the $Q$-function from **Activity 4**, you can observe the following interaction.
# 
# ```python
# rand.seed(42)
# 
# for b in B[:10]:
#     
#     if np.all(b > 0):
#         print('Belief (approx.) uniform')
#     else:
#         initial = True
# 
#         for i in range(len(M[0])):
#             if b[0, i] > 0:
#                 if initial:
#                     initial = False
#                     print('Belief: [', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
#                 else:
#                     print(',', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
#         print(']')
# 
#     print('MLS action:', M[1][get_heuristic_action(b, Q, 'mls')], end='; ')
#     print('AV action:', M[1][get_heuristic_action(b, Q, 'av')], end='; ')
#     print('Q-MDP action:', M[1][get_heuristic_action(b, Q, 'q-mdp')])
# 
#     print()
# ```
# 
# Output:
# 
# ```
# Belief (approx.) uniform
# MLS action: Down; AV action: Up; Q-MDP action: Up
# 
# Belief: [ (18, 0, 1) : 0.167, (18, 0, 2) : 0.667, (18, 0, 3) : 0.167]
# MLS action: Right; AV action: Right; Q-MDP action: Right
# 
# Belief: [ (18, 0, 1) : 0.333, (18, 0, 2) : 0.333, (18, 0, 3) : 0.333]
# MLS action: Up; AV action: Right; Q-MDP action: Right
# 
# Belief: [ (19, 0, 1) : 0.333, (19, 0, 2) : 0.333, (19, 0, 3) : 0.333]
# MLS action: Down; AV action: Down; Q-MDP action: Down
# 
# Belief: [ (19, 0, 1) : 0.167, (19, 0, 2) : 0.667, (19, 0, 3) : 0.167]
# MLS action: Right; AV action: Down; Q-MDP action: Down
# 
# Belief: [ (26, 0, 1) : 0.333, (26, 0, 2) : 0.333, (26, 0, 3) : 0.333]
# MLS action: Down; AV action: Left; Q-MDP action: Left
# 
# Belief: [ (25, 0, 1) : 0.167, (25, 0, 2) : 0.667, (25, 0, 3) : 0.167]
# MLS action: Down; AV action: Left; Q-MDP action: Left
# 
# Belief: [ (24, S, 1) : 0.333, (24, S, 2) : 0.333, (24, S, 3) : 0.333]
# MLS action: Up; AV action: Right; Q-MDP action: Right
# 
# Belief: [ (24, S, 1) : 0.167, (24, S, 2) : 0.667, (24, S, 3) : 0.167]
# MLS action: Right; AV action: Right; Q-MDP action: Right
# 
# Belief: [ (25, S, 1) : 0.167, (25, S, 2) : 0.667, (25, S, 3) : 0.167]
# MLS action: Down; AV action: Right; Q-MDP action: Right
# ```

# In[118]:


#test sanity
rand.seed(42)

for b in B[:10]:

    if np.all(b > 0):
        print('Belief (approx.) uniform')
    else:
        initial = True

        for i in range(len(M[0])):
            if b[0, i] > 0:
                if initial:
                    initial = False
                    print('Belief: [', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
                else:
                    print(',', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
        print(']')

    print('MLS action:', M[1][get_heuristic_action(b, Q, 'mls')], end='; ')
    print('AV action:', M[1][get_heuristic_action(b, Q, 'av')], end='; ')
    print('Q-MDP action:', M[1][get_heuristic_action(b, Q, 'q-mdp')])

    print()


# Suppose that the optimal cost-to-go function for the POMDP can be represented using a set of $\alpha$-vectors that have been precomputed for you. 
# 
# ---
# 
# #### Activity 6
# 
# Write a function `get_optimal_action` that, given a belief vector and a set of pre-computed $\alpha$-vectors, computes the corresponding optimal action. Your function should receive, as inputs,
# 
# * A belief state represented as a `numpy` array like those of **Activity 3**;
# * The set of optimal $\alpha$-vectors, represented as a `numpy` array `av`; the $\alpha$-vectors correspond to the **columns** of `av`;
# * A list `ai` containing the **indices** (not the names) of the actions corresponding to each of the $\alpha$-vectors. In other words, the `ai[k]` is the action index of the $\alpha$-vector `av[:, k]`.
# 
# Your function should return an integer corresponding to the index of the optimal action. *Ties should be broken randomly, i.e., when selecting the minimizing action, you should randomly select between all minimizers*.
# 
# ---

# In[119]:


def get_optimal_action(belief, alpha_vec, ai):
    J = []
    for i in range(len(alpha_vec.T)):
        J.append(np.dot(belief, alpha_vec[:,i])[0])
    min_alpha_vec = int(np.argmin(np.asarray(J)))
    action = ai[min_alpha_vec]
    return action


# The binary file `alpha.npz` contains the $\alpha$-vectors and action indices for the Pacman environment in the figure above. If you compute the optimal actions for the beliefs in the example from **Activity 3** using the $\alpha$-vectors in `alpha.npz`, you can observe the following interaction.
# 
# ```python
# data = np.load('alpha.npz')
# 
# # Alpha vectors
# alph = data['avec']
# 
# # Corresponding actions
# act = list(map(lambda x : M[1].index(x), data['act']))
# 
# # Example alpha vector (n. 3) and action
# print('Alpha-vector n. 3:')
# print(np.round(alph[:, 3], 3))
# print('Associated action:', M[1][act[3]], '(action n. %i)' % act[3])
# print()
# 
# rand.seed(42)
# 
# # Computing the optimal actions
# for b in B[:10]:
#     
#     if np.all(b > 0):
#         print('Belief (approx.) uniform')
#     else:
#         initial = True
# 
#         for i in range(len(M[0])):
#             if b[0, i] > 0:
#                 if initial:
#                     initial = False
#                     print('Belief: [', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
#                 else:
#                     print(',', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
#         print(']')
# 
#     print('MLS action:', M[1][get_heuristic_action(b, Q, 'mls')], end='; ')
#     print('AV action:', M[1][get_heuristic_action(b, Q, 'av')], end='; ')
#     print('Q-MDP action:', M[1][get_heuristic_action(b, Q, 'q-mdp')], end='; ')
#     print('Optimal action:', M[1][get_optimal_action(b, alph, act)])
# 
#     print()
# ```
# 
# Output:
# 
# ```
# Alpha-vector n. 3:
# [0.007 0.278 0.366 1.007 1.143 1.187 0.107 0.007 0.107 1.057 1.007 1.057
#  0.53  0.155 0.007 1.269 1.081 1.007 0.248 0.378 0.248 1.127 1.306 1.127
#  0.46  0.336 0.46  1.234 1.171 1.234 0.495 0.499 0.495 1.251 1.253 1.251
#  0.625 0.71  0.625 1.316 1.359 1.316 0.366 0.278 0.366 1.187 1.143 1.187
#  0.53  0.155 0.53  1.269 1.081 1.269 0.981 0.898 0.981 1.518 1.519 1.518
#  0.981 0.898 0.981 1.518 1.519 1.518 0.625 0.71  0.625 1.316 1.359 1.316
#  0.681 0.605 0.681 1.344 1.306 1.344 0.419 0.537 0.419 1.416 1.379 1.416
#  0.611 0.499 0.611 1.309 1.253 1.309 0.825 0.75  0.825 1.416 1.379 1.416
#  0.981 0.898 0.981 1.518 1.519 1.518 0.714 0.71  0.714 1.361 1.359 1.361
#  0.714 0.81  0.714 1.361 1.409 1.361 0.419 0.537 0.419 1.416 1.379 1.416
#  0.825 0.89  0.825 1.416 1.449 1.416 0.825 0.75  0.825 1.416 1.379 1.416
#  0.825 0.89  0.825 1.416 1.449 1.416 1.823 1.837 1.823 1.823 1.837 1.823
#  1.823 1.837 1.823 0.714 0.81  0.714 1.361 1.409 1.361 0.825 0.89  0.825
#  1.416 1.449 1.416 1.823 1.837 1.823 1.823 1.837 1.823 1.823 1.837 1.823
#  1.823 1.837 1.823 1.823 1.837 1.823 1.823 1.837 1.823 1.823 1.837 1.823
#  1.823 1.837 1.823 0.84  1.197 0.84  1.291 1.344 1.291 1.823 1.837 1.823
#  1.823 1.837 1.823 1.823 1.837 1.823 1.823 1.837 1.823 1.823 1.837 1.823
#  1.823 1.837 1.823 0.007 0.007]
# Associated action: Up (action n. 0)
# 
# Belief (approx.) uniform
# MLS action: Down; AV action: Up; Q-MDP action: Up; Optimal action: Left
# 
# Belief: [ (18, 0, 1) : 0.167, (18, 0, 2) : 0.667, (18, 0, 3) : 0.167]
# MLS action: Right; AV action: Right; Q-MDP action: Right; Optimal action: Right
# 
# Belief: [ (18, 0, 1) : 0.333, (18, 0, 2) : 0.333, (18, 0, 3) : 0.333]
# MLS action: Right; AV action: Right; Q-MDP action: Right; Optimal action: Right
# 
# Belief: [ (19, 0, 1) : 0.333, (19, 0, 2) : 0.333, (19, 0, 3) : 0.333]
# MLS action: Right; AV action: Down; Q-MDP action: Down; Optimal action: Down
# 
# Belief: [ (19, 0, 1) : 0.167, (19, 0, 2) : 0.667, (19, 0, 3) : 0.167]
# MLS action: Up; AV action: Down; Q-MDP action: Down; Optimal action: Down
# 
# Belief: [ (26, 0, 1) : 0.333, (26, 0, 2) : 0.333, (26, 0, 3) : 0.333]
# MLS action: Left; AV action: Left; Q-MDP action: Left; Optimal action: Left
# 
# Belief: [ (25, 0, 1) : 0.167, (25, 0, 2) : 0.667, (25, 0, 3) : 0.167]
# MLS action: Down; AV action: Left; Q-MDP action: Left; Optimal action: Left
# 
# Belief: [ (24, S, 1) : 0.333, (24, S, 2) : 0.333, (24, S, 3) : 0.333]
# MLS action: Right; AV action: Right; Q-MDP action: Right; Optimal action: Right
# 
# Belief: [ (24, S, 1) : 0.167, (24, S, 2) : 0.667, (24, S, 3) : 0.167]
# MLS action: Right; AV action: Right; Q-MDP action: Right; Optimal action: Right
# 
# Belief: [ (25, S, 1) : 0.167, (25, S, 2) : 0.667, (25, S, 3) : 0.167]
# MLS action: Up; AV action: Right; Q-MDP action: Right; Optimal action: Right
# ```

# In[120]:


#test sanity
data = np.load('alpha.npz')

# Alpha vectors
alph = data['avec']

# Corresponding actions
act = list(map(lambda x : M[1].index(x), data['act']))

# Example alpha vector (n. 3) and action
print('Alpha-vector n. 3:')
print(np.round(alph[:, 3], 3))
print('Associated action:', M[1][act[3]], '(action n. %i)' % act[3])
print()

rand.seed(42)

# Computing the optimal actions
for b in B[:10]:

    if np.all(b > 0):
        print('Belief (approx.) uniform')
    else:
        initial = True

        for i in range(len(M[0])):
            if b[0, i] > 0:
                if initial:
                    initial = False
                    print('Belief: [', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
                else:
                    print(',', M[0][i], ': %.3f' % np.round(b[0, i], 3), end='')
        print(']')

    print('MLS action:', M[1][get_heuristic_action(b, Q, 'mls')], end='; ')
    print('AV action:', M[1][get_heuristic_action(b, Q, 'av')], end='; ')
    print('Q-MDP action:', M[1][get_heuristic_action(b, Q, 'q-mdp')], end='; ')
    print('Optimal action:', M[1][get_optimal_action(b, alph, act)])

    print()


# In[ ]:




