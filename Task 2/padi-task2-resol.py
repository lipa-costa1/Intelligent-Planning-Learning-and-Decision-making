#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 2: Markov decision problems
# 
# In the end of the lab, you should export the notebook to a Python script (File >> Download as >> Python (.py)). Your file should be named `padi-lab2-groupXX.py`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
# 
# Make sure...
# 
# * **... that the subject is of the form `[<group n.>] LAB <lab n.>`.** 
# 
# * **... to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.** 
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).
# 
# ### 1. The MDP Model
# 
# Consider once again the Pacman modeling problem described in the Homework and for which you wrote a Markov decision problem model. In this lab, you will consider a larger version of the Pacman problem, described by the diagram:
# 
# <img src="pacman-big.png">
# 
# Recall that the MDP should describe the decision-making of a player. In the above domain,
# 
# * The ghost **moves randomly between cells 1-3**.
# * The player controls the movement of Pacman through four actions: `Up`, `Down`, `Left`, and `Right`. 
# * Each action moves the Pacman character one step in the corresponding direction, if an adjacent cell exists in that direction. Otherwise, Pacman remains in the same place.
# * The cell in the bottom left corner (cell `29`) is adjacent, to the left, to the cell in the bottom right corner (cell `35`). In other words, if Pacman "moves left" in cell `29` it will end up in cell `35` and vice-versa.
# * If Pacman lies in the same cell as the ghost (in either cell `1`, `2`, or `3`), the player loses the game. However, if Pacman "eats" the blue pellet (in cell `24`), it gains the ability to "eat" the ghost. In this case, if Pacman lies in the same cell as the ghost, it "eats" the ghost and wins the game. Assume that Pacman can never be in cell `24` without "eating" the pellet.
# 
# In this lab you will use an MDP based on the aforementioned domain and investigate how to evaluate, solve and simulate a Markov decision problem.
# 
# **Throughout the lab, unless if stated otherwise, use $\gamma=0.9$.**
# 
# $$\diamond$$
# 
# In this first activity, you will implement an MDP model in Python. You will start by loading the MDP information from a `numpy` binary file, using the `numpy` function `load`. The file contains the list of states, actions, the transition probability matrices and cost function.

# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_mdp` that receives, as input, a string corresponding to the name of the file with the MDP information, and a real number $\gamma$ between $0$ and $1$. The loaded file contains 4 arrays:
# 
# * An array `X` that contains all the states in the MDP represented as strings. In the Pacman environment above, for example, there is a total of 209 states, each describing the position of Pacman in the environment, whether it has eaten the blue pellet, and the position of the ghost. Those states are either one of the strings `"V"` or `"D"`, corresponding to the absorbing "victory" and "defeat" states, or a string of the form `"(p, s, g)"`, where:
#     * `p` is a number between 1 and 35 indicating the position of Pacman;
#     * `s` is either `0` or `S`, where `0` indicates that Pacman has not yet eaten the pellet; `S` indicates that Pacman has eaten the pellet (and now has "superpowers");
#     * `g` is a number between 1 and 3, indicating the position of the ghost.
# * An array `A` that contains all the actions in the MDP represented as strings. In the Pacman environment above, for example, each action is represented as a string `"Up"`, `"Down"`, `"Left"` or `"Right"`.
# * An array `P` containing `len(A)` subarrays, each with dimension `len(X)` &times; `len(X)` and  corresponding to the transition probability matrix for one action.$^1$
# * An array `c` containing the cost function for the MDP.
# 
# Your function should create the MDP as a tuple `(X, A, (Pa, a = 0, ..., len(A)), c, g)`, where `X` is a tuple containing the states in the MDP represented as strings (see above), `A` is a tuple containing the actions in the MDP represented as strings (see above), `P` is a tuple with `len(A)` elements, where `P[a]` is an `np.array` corresponding to the transition probability matrix for action `a`, `c` is an np.array corresponding to the cost function for the MDP, and `g` is a float, corresponding to the discount and provided as the argument $\gamma$ of your function. Your function should return the MDP tuple.
# 
# ---

# In[123]:


from numpy import load

data = load('pacman.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])


# In[124]:


import numpy as np

def load_mdp(MDP, gamma):
    M = load('pacman.npz')
    X = tuple(M['X'])
    A = tuple(M['A']) 
    P = tuple(M['P'])
    c= M['c']
    return (X, A, P, c, gamma)


# We provide below an example of application of the function with the file `pacman.npz` that you can use as a first "sanity check" for your code. Note that, even fixing the seed, the results you obtain may slightly differ.
# 
# ```python
# import numpy.random as rand
# 
# M = load_mdp('pacman.npz', 0.9)
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
# # Transition probabilities
# print('\n= Transition probabilities =')
# 
# for i in range(len(M[1])):
#     print('\nTransition probability matrix dimensions (action %s):' % M[1][i], M[2][i].shape)
#     print('Dimensions add up for action "%s"?' % M[1][i], np.isclose(np.sum(M[2][i]), len(M[0])))
#     
# print('\nState-action pair (%s, %s) transitions to state(s)' % (M[0][s], M[1][a]))
# print("s' in", np.array(M[0])[np.where(M[2][a][s, :] > 0)])
# 
# # Cost
# print('\n= Costs =')
# 
# print('\nSpecial states with cost different from 0.1:')
# print(np.array(M[0])[np.where(M[3][:, 0] != 0.1)])
# print('Associated costs:')
# print(M[3][np.where(M[3][:, 0] != 0.1), 0])
# 
# print('\nCost for the state-action pair (%s, %s):' % (M[0][s], M[1][a]))
# print('c(s, a) =', M[3][s, a])
# 
# 
# # Discount
# print('\n= Discount =')
# print('\ngamma =', M[4])
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

# In[125]:


# Test sanity

import numpy.random as rand

M = load_mdp('pacman.npz', 0.9)

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

# Transition probabilities
print('\n= Transition probabilities =')

for i in range(len(M[1])):
    print('\nTransition probability matrix dimensions (action %s):' % M[1][i], M[2][i].shape)
    print('Dimensions add up for action "%s"?' % M[1][i], np.isclose(np.sum(M[2][i]), len(M[0])))

print('\nState-action pair (%s, %s) transitions to state(s)' % (M[0][s], M[1][a]))
print("s' in", np.array(M[0])[np.where(M[2][a][s, :] > 0)])

# Cost
print('\n= Costs =')

print('\nSpecial states with cost different from 0.1:')
print(np.array(M[0])[np.where(M[3][:, 0] != 0.1)])
print('Associated costs:')
print(M[3][np.where(M[3][:, 0] != 0.1), 0])

print('\nCost for the state-action pair (%s, %s):' % (M[0][s], M[1][a]))
print('c(s, a) =', M[3][s, a])


# Discount
print('\n= Discount =')
print('\ngamma =', M[4])


# ### 2. Prediction
# 
# You are now going to evaluate a given policy, computing the corresponding cost-to-go.

# ---
# 
# #### Activity 2.
# 
# Write a function `noisy_policy` that builds a noisy policy "around" a provided action. Your function should receive, as input, an MDP described as a tuple like that of **Activity 1**, an integer `a`, corresponding to the index of an action in the MDP, and a real number `eps`. The function should return, as output, a policy for the provided MDP that selects action with index `a` with a probability `1-eps` and, with probability `eps`, selects another action uniformly at random. The policy should be a `numpy` array with as many rows as states and as many columns as actions, where the element in position `[s,a]` should contain the probability of action `a` in state `s` according to the desired policy.
# 
# **Note:** The examples provided correspond for the MDP in the previous Pacman environment. However, your code should be tested with MDPs of different sizes, so **make sure not to hard-code any of the MDP elements into your code**.
# 
# ---

# In[126]:


def noisy_policy(M, a, eps):
    n_states = len(M[0])
    n_actions = len(M[1])
    pi = np.zeros((n_states,n_actions))
    for i in range(n_states):
        for j in range(n_actions):
            if j==a: 
                pi[i][j]= 1-eps 
            else:
                pi[i][j] = eps / (n_actions-1)
    return pi


# We provide below an example of application of the function with MDP from the example in **Activity 1**, that you can use as a first "sanity check" for your code. Note that, even fixing the seed, the results you obtain may slightly differ. Note also that, as emphasized above, your function should work with **any** MDP that is specified as a tuple with the structure of the one from **Activity 1**.
# 
# ```python
# rand.seed(42)
# 
# # Noiseless policy for action "Left" (action index: 2)
# pol_noiseless = noisy_policy(M, 2, 0.)
# 
# # Random state
# s = 106 # State (18, 0, 2)
# 
# # Policy at selected state
# print('Random state:', M[0][s])
# print('Noiseless policy at selected state:', pol_noiseless[s, :])
# 
# # Noisy policy for action "Left" (action index: 2)
# pol_noisy = noisy_policy(M, 2, 0.1)
# 
# # Policy at selected state
# print('Noisy policy at selected state:', np.round(pol_noisy[s, :], 2))
# 
# # Random policy for action "Left" (action index: 2)
# pol_random = noisy_policy(M, 2, 0.75)
# 
# # Policy at selected state
# print('Random policy at selected state:', np.round(pol_random[s, :], 2))
# ```
# 
# Output:
# 
# ```
# Random state: (18, 0, 2)
# Noiseless policy at selected state: [0. 0. 1. 0.]
# Noisy policy at selected state: [0.03 0.03 0.9  0.03]
# Random policy at selected state: [0.25 0.25 0.25 0.25]
# ```

# In[127]:


#Test sanity

rand.seed(42)

# Noiseless policy for action "Left" (action index: 2)
pol_noiseless = noisy_policy(M, 2, 0.)

# Random state
s = 106 # State (18, 0, 2)

# Policy at selected state
print('Random state:', M[0][s])
print('Noiseless policy at selected state:', pol_noiseless[s, :])

# Noisy policy for action "Left" (action index: 2)
pol_noisy = noisy_policy(M, 2, 0.1)

# Policy at selected state
print('Noisy policy at selected state:', np.round(pol_noisy[s, :], 2))

# Random policy for action "Left" (action index: 2)
pol_random = noisy_policy(M, 2, 0.75)

# Policy at selected state
print('Random policy at selected state:', np.round(pol_random[s, :], 2))


# ---
# 
# #### Activity 3.
# 
# You will now write a function called `evaluate_pol` that evaluates a given policy. Your function should receive, as an input, an MDP described as a tuple like that of **Activity 1** and a policy described as an array like that of **Activity 2** and return a `numpy` array corresponding to the cost-to-go function associated with the given policy. 
# 
# **Note:** The array returned by your function should have as many rows as the number of states in the received MDP, and exactly one column. Note also that, as before, your function should work with **any** MDP that is specified as a tuple with the same structure as the one from **Activity 1**.
# 
# ---

# In[128]:


def evaluate_pol(M,pi):
    n_states = len(M[0])
    n_actions = len(M[1])
    c = M[3]
    gamma = M[4]
    prob = np.array(M[2])
    Ppi = np.zeros((n_states, n_states))
    for i in range(n_states):
        Ppi[i] = pi[i,:].dot(prob[:,i,:])     
    A = np.linalg.inv(np.eye(n_states) - gamma*Ppi)
    cpi = np.diag(pi.dot(c.T))
    Jpi = A.dot(cpi).reshape((-1,1))
    return Jpi


# As an example, you can evaluate the random policy from **Activity 2** in the MDP from **Activity 1**.
# 
# ```python
# Jact2 = evaluate_pol(M, pol_noisy)
# 
# rand.seed(42)
# 
# print('Dimensions of cost-to-go:', Jact2.shape)
# 
# print('\nExample values of the computed cost-to-go:')
# 
# s = 106 # State (18, 0, 2)
# print('\nCost-to-go at state %s:' % M[0][s], np.round(Jact2[s], 3))
# 
# s = 12 # State (3, S, 1)
# print('Cost-to-go at state %s:' % M[0][s], np.round(Jact2[s], 3))
# 
# s = 164 # State (28, 0, 3)
# print('Cost-to-go at state %s:' % M[0][s], np.round(Jact2[s], 3))
# 
# # Example with random policy
# 
# rand_pol = rand.randint(2, size=(len(M[0]), len(M[1]))) + 0.01 # We add 0.01 to avoid all-zero rows
# rand_pol = rand_pol / rand_pol.sum(axis = 1, keepdims = True)
# 
# Jrand = evaluate_pol(M, rand_pol)
# 
# print('\nExample values of the computed cost-to-go:')
# 
# s = 106 # State (18, 0, 2)
# print('\nCost-to-go at state %s:' % M[0][s], np.round(Jrand[s], 3))
# 
# s = 12 # State (3, S, 1)
# print('Cost-to-go at state %s:' % M[0][s], np.round(Jrand[s], 3))
# 
# s = 164 # State (28, 0, 3)
# print('Cost-to-go at state %s:' % M[0][s], np.round(Jrand[s], 3))
# ```
# 
# Output: 
# ```
# Dimensions of cost-to-go: (209, 1)
# 
# Example values of the computed cost-to-go:
# 
# Cost-to-go at state (18, 0, 2): [1.]
# Cost-to-go at state (3, S, 1): [0.144]
# Cost-to-go at state (28, 0, 3): [1.]
# 
# Example values of the computed cost-to-go:
# 
# Cost-to-go at state (18, 0, 2): [1.]
# Cost-to-go at state (3, S, 1): [0.905]
# Cost-to-go at state (28, 0, 3): [1.]
# ```

# In[129]:


#Test sanity

Jact2 = evaluate_pol(M, pol_noisy)

rand.seed(42)

print('Dimensions of cost-to-go:', Jact2.shape)

print('\nExample values of the computed cost-to-go:')

s = 106 # State (18, 0, 2)
print('\nCost-to-go at state %s:' % M[0][s], np.round(Jact2[s], 3))

s = 12 # State (3, S, 1)
print('Cost-to-go at state %s:' % M[0][s], np.round(Jact2[s], 3))

s = 164 # State (28, 0, 3)
print('Cost-to-go at state %s:' % M[0][s], np.round(Jact2[s], 3))

# Example with random policy

rand_pol = rand.randint(2, size=(len(M[0]), len(M[1]))) + 0.01 # We add 0.01 to avoid all-zero rows
rand_pol = rand_pol / rand_pol.sum(axis = 1, keepdims = True)

Jrand = evaluate_pol(M, rand_pol)

print('\nExample values of the computed cost-to-go:')

s = 106 # State (18, 0, 2)
print('\nCost-to-go at state %s:' % M[0][s], np.round(Jrand[s], 3))

s = 12 # State (3, S, 1)
print('Cost-to-go at state %s:' % M[0][s], np.round(Jrand[s], 3))

s = 164 # State (28, 0, 3)
print('Cost-to-go at state %s:' % M[0][s], np.round(Jrand[s], 3))


# ### 3. Control
# 
# In this section you are going to compare value and policy iteration, both in terms of time and number of iterations.

# ---
# 
# #### Activity 4
# 
# In this activity you will show that the policy in Activity 3 is _not_ optimal. For that purpose, you will use value iteration to compute the optimal cost-to-go, $J^*$, and show that $J^*\neq J^\pi$. 
# 
# Write a function called `value_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal cost-to-go function associated with that MDP. Before returning, your function should print:
# 
# * The time it took to run, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# * The number of iterations, in the format `N. iterations: xxx`, where `xxx` represents the number of iterations.
# 
# **Note 1:** Stop the algorithm when the error between iterations is smaller than $10^{-8}$.
# 
# **Note 2:** You may find useful the function ``time()`` from the module ``time``.
# 
# **Note 3:** The array returned by your function should have as many rows as the number of states in the received MDP, and exactly one column. As before, your function should work with **any** MDP that is specified as a tuple with the same structure as the one from **Activity 1**.
# 
# 
# ---

# In[130]:


import time

def value_iteration(M):
    init_time = time.time()
    n_states = len(M[0])
    n_actions = len(M[1])
    prob = np.array(M[2])
    c = M[3]
    gamma = M[4]
    J = np.zeros((n_states,1))
    err = 1
    i = 0
    while err >= 10**(-8):
        Q_state = []
        for j in range(len(c[0])):
            cj=np.reshape(c[:,j],(n_states,1))
            Pj=np.reshape(prob[j,:].dot(J), (n_states,1))
            Q_state.append(cj + gamma*Pj)
        Jnew = np.min(Q_state,axis=0)
        err = np.linalg.norm(Jnew - J)
        i += 1
        J = Jnew
    print('Execution time: %s seconds'  % round((time.time() - init_time),3))
    print('N. iterations:', i)
    return J
    


# For example, using the MDP from **Activity 1** you could obtain the following interaction.
# 
# ```python
# Jopt = value_iteration(M)
# 
# print('\nDimensions of cost-to-go:', Jopt.shape)
# 
# rand.seed(42)
# 
# print('\nExample values of the optimal cost-to-go:')
# 
# s = 106 # State (18, 0, 2)
# print('\nCost to go at state %s:' % M[0][s], Jopt[s])
# 
# s = 12 # State (3, S, 1)
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# s = 164 # State (28, 0, 3)
# print('Cost to go at state %s:' % M[0][s], Jopt[s])
# 
# print('\nIs the policy from Activity 2 optimal?', np.all(np.isclose(Jopt, Jact2)))
# ```
# 
# Output:
# ```
# Execution time: 0.007 seconds
# N. iterations: 23
# 
# Dimensions of cost-to-go: (209, 1)
# 
# Example values of the optimal cost-to-go:
# 
# Cost to go at state (18, 0, 2): [0.75852275]
# Cost to go at state (3, S, 1): [0.1]
# Cost to go at state (28, 0, 3): [0.66875548]
# 
# Is the policy from Activity 2 optimal? False
# ```

# In[131]:


# Test sanity

Jopt = value_iteration(M)

print('\nDimensions of cost-to-go:', Jopt.shape)

rand.seed(42)

print('\nExample values of the optimal cost-to-go:')

s = 106 # State (18, 0, 2)
print('\nCost to go at state %s:' % M[0][s], Jopt[s])

s = 12 # State (3, S, 1)
print('Cost to go at state %s:' % M[0][s], Jopt[s])

s = 164 # State (28, 0, 3)
print('Cost to go at state %s:' % M[0][s], Jopt[s])

print('\nIs the policy from Activity 2 optimal?', np.all(np.isclose(Jopt, Jact2)))


# ---
# 
# #### Activity 5
# 
# You will now compute the optimal policy using policy iteration. Write a function called `policy_iteration` that receives as input an MDP represented as a tuple like that of **Activity 1** and returns an `numpy` array corresponding to the optimal policy associated with that MDP. Before returning, your function should print:
# * The time it took to run, in the format `Execution time: xxx seconds`, where `xxx` represents the number of seconds rounded up to $3$ decimal places.
# * The number of iterations, in the format `N. iterations: xxx`, where `xxx` represents the number of iterations.
# 
# **Note:** If you find that numerical errors affect your computations (especially when comparing two values/arrays) you may use the `numpy` function `isclose` with adequately set absolute and relative tolerance parameters (e.g., $10^{-8}$).
# 
# ---

# In[132]:


def policy_iteration(M):
    init_time = time.time()
    n_states = len(M[0])
    n_actions = len(M[1])
    prob = np.array(M[2])
    c = M[3]
    gamma = M[4]
    pi = np.ones((n_states, n_actions))/n_actions
    quit = False
    Ppi = np.zeros((n_states, n_states))
    i = 0
    while not quit: 
        for j in range(n_states):
            Ppi[j] = pi[j,:].dot(prob[:,j,:])
        cpi = np.diag(pi.dot(c.T)) 
        J = np.linalg.inv(np.eye(n_states) - gamma*Ppi).dot(cpi)
        Q_state = []
        for w in range(len(c[0])):
            cw=np.reshape(c[:,w],(n_states,1))
            Pw=np.reshape(prob[w,:].dot(J), (n_states,1))
            Q_state.append(cw + gamma*Pw)
        pinew = np.zeros((n_states, n_actions))
        for z in range(n_actions):
            pinew[:, z, None] = np.isclose(Q_state[z],np.min(Q_state,axis=0),atol=10**(-8),rtol=10**(-8)).astype(int)
        pinew = pinew/np.sum(pinew,axis=1,keepdims=True)
        quit = (pi==pinew).all()
        pi = pinew
        i += 1
    print('Execution time: %s seconds'  % round((time.time() - init_time),3))
    print('N. iterations:', i)
    return pi


# For example, using the MDP from **Activity 1** you could obtain the following interaction.
# 
# ```python
# popt = policy_iteration(M)
# 
# print('\nDimension of the policy matrix:', popt.shape)
# 
# rand.seed(42)
# 
# print('\nExamples of actions according to the optimal policy:')
# 
# # Select random state, and action using the policy computed
# s = 106 # State (18, 0, 2)
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# 
# # Select random state, and action using the policy computed
# s = 12 # State (3, S, 1)
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# 
# # Select random state, and action using the policy computed
# s = 164 # State (28, 0, 3)
# a = rand.choice(len(M[1]), p=popt[s, :])
# print('Policy at state %s: %s' % (M[0][s], M[1][a]))
# 
# # Verify optimality of the computed policy
# 
# print('\nOptimality of the computed policy:')
# 
# Jpi = evaluate_pol(M, popt)
# print('- Is the new policy optimal?', np.all(np.isclose(Jopt, Jpi)))
# ```
# 
# Output:
# ```
# Execution time: 0.006 seconds
# N. iterations: 3
# 
# Dimension of the policy matrix: (209, 4)
# 
# Examples of actions according to the optimal policy:
# Policy at state (18, 0, 2): Right
# Policy at state (3, S, 1): Left
# Policy at state (28, 0, 3): Up
# 
# Optimality of the computed policy:
# - Is the new policy optimal? True
# ```

# In[133]:


# Test sanity

popt = policy_iteration(M)

print('\nDimension of the policy matrix:', popt.shape)

rand.seed(42)

print('\nExamples of actions according to the optimal policy:')

# Select random state, and action using the policy computed
s = 106 # State (18, 0, 2)
a = rand.choice(len(M[1]), p=popt[s, :])
print('Policy at state %s: %s' % (M[0][s], M[1][a]))

# Select random state, and action using the policy computed
s = 12 # State (3, S, 1)
a = rand.choice(len(M[1]), p=popt[s, :])
print('Policy at state %s: %s' % (M[0][s], M[1][a]))

# Select random state, and action using the policy computed
s = 164 # State (28, 0, 3)
a = rand.choice(len(M[1]), p=popt[s, :])
print('Policy at state %s: %s' % (M[0][s], M[1][a]))

# Verify optimality of the computed policy

print('\nOptimality of the computed policy:')

Jpi = evaluate_pol(M, popt)
print('- Is the new policy optimal?', np.all(np.isclose(Jopt, Jpi)))


# ### 4. Simulation
# 
# Finally, in this section you will check whether the theoretical computations of the cost-to-go actually correspond to the cost incurred by an agent following a policy.

# ---
# 
# #### Activity 6
# 
# Write a function `simulate` that receives, as inputs
# 
# * An MDP represented as a tuple like that of **Activity 1**;
# * A policy, represented as an `numpy` array like that of **Activity 2**;
# * An integer, `x0`, corresponding to a state index
# * A second integer, `length`
# 
# Your function should return, as an output, a float corresponding to the estimated cost-to-go associated with the provided policy at the provided state. To estimate such cost-to-go, your function should:
# 
# * Generate **`NRUNS`** trajectories of `length` steps each, starting in the provided state and following the provided policy. 
# * For each trajectory, compute the accumulated (discounted) cost. 
# * Compute the average cost over the 100 trajectories.
# 
# **Note 1:** You may find useful to import the numpy module `numpy.random`.
# 
# **Note 2:** Each simulation may take a bit of time, don't despair ☺️.
# 
# ---

# In[134]:


NRUNS = 100 # Do not delete this
import numpy.random as rand 


# In[135]:


def simulate(M, pi, x0, lenght):
    n_states = len(M[0])
    n_actions = len(M[1]) 
    prob = np.array(M[2])
    c = M[3]
    gamma = M[4]
    cost = 0 
    for j in range(NRUNS): 
        s = x0
        run_cost = 0 
        for i in range(lenght): 
            a = rand.choice(range(n_actions), p = pi[s])
            run_cost = run_cost + (gamma**i) * c[:,a][s] 
            s = rand.choice(range(n_states), p = prob[a][s]) 
        cost = cost + run_cost 
    return cost/NRUNS 


# For example, we can use this function to estimate the values of some random states and compare them with those from **Activity 4**.
# 
# ```python
# rand.seed(42)
# 
# # Select random state, and evaluate for the optimal policy
# s = 106 # State (18, 0, 2)
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', np.round(Jopt[s], 4))
# print('\tEmpirical:', np.round(simulate(M, popt, s, 1000), 4))
# 
# # Select random state, and evaluate for the optimal policy
# s = 12 # State (3, S, 1)
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', np.round(Jopt[s], 4))
# print('\tEmpirical:', np.round(simulate(M, popt, s, 1000), 4))
# 
# # Select random state, and evaluate for the optimal policy
# s = 164 # State (28, 0, 3)
# print('Cost-to-go for state %s:' % M[0][s])
# print('\tTheoretical:', np.round(Jopt[s], 4))
# print('\tEmpirical:', np.round(simulate(M, popt, s, 1000), 4))
# ```
# 
# Output:
# ````
# Cost-to-go for state (18, 0, 2):
# 	Theoretical: [0.7585]
# 	Empirical: 0.7588
# Cost-to-go for state (3, S, 1):
# 	Theoretical: [0.1]
# 	Empirical: 0.1
# Cost-to-go for state (28, 0, 3):
# 	Theoretical: [0.6688]
# 	Empirical: 0.6677
# ```

# In[136]:


# Test sanity

rand.seed(42)

# Select random state, and evaluate for the optimal policy
s = 106 # State (18, 0, 2)
print('Cost-to-go for state %s:' % M[0][s])
print('\tTheoretical:', np.round(Jopt[s], 4))
print('\tEmpirical:', np.round(simulate(M, popt, s, 1000), 4))

# Select random state, and evaluate for the optimal policy
s = 12 # State (3, S, 1)
print('Cost-to-go for state %s:' % M[0][s])
print('\tTheoretical:', np.round(Jopt[s], 4))
print('\tEmpirical:', np.round(simulate(M, popt, s, 1000), 4))

# Select random state, and evaluate for the optimal policy
s = 164 # State (28, 0, 3)
print('Cost-to-go for state %s:' % M[0][s])
print('\tTheoretical:', np.round(Jopt[s], 4))
print('\tEmpirical:', np.round(simulate(M, popt, s, 1000), 4))


# In[ ]:




