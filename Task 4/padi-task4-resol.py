#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 4: Reinforcement learning
# 
# In the end of the lab, you should export the notebook to a Python script (File >> Download as >> Python (.py)). Your file should be named `padi-lab4-groupXX.py`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
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
# In this lab you will implement several reinforcement learning algorithms, and use the "Pacman" domain, from Lab 2, to test and compare these algorithms. Don't forget, however, that your functions should work for **any MDP** and not just the one provided. 
# 
# The "Pacman" domain to be used is represented in the diagram below.
# 
# <img src="pacman-big.png">
# 
# In the Pacman domain above,
# 
# * The ghost moves randomly between cells 1-3.
# * The player controls the movement of Pacman through four actions: `Up`, `Down`, `Left`, and `Right`. 
# * Each action moves the Pacman character one step in the corresponding direction, if an adjacent cell exists in that direction. Otherwise, Pacman remains in the same place.
# * The cell in the bottom left corner (cell `29`) is adjacent, to the left, to the cell in the bottom right corner (cell `35`). In other words, if Pacman "moves left" in cell `29` it will end up in cell `35` and vice-versa.
# * If Pacman lies in the same cell as the ghost (in either cell `1`, `2`, or `3`), the player loses the game. However, if Pacman "eats" the blue pellet (in cell `24`), it gains the ability to "eat" the ghost. In this case, if Pacman lies in the same cell as the ghost, it "eats" the ghost and wins the game. Assume that Pacman can never be in cell `24` without "eating" the pellet.
# 
# **Throughout the lab, unless if stated otherwise, use $\gamma=0.9$.**
# 
# $$\diamond$$
# 
# We start by loading the MDP for the "Pacman" domain from the file `pacman.npz`. We will use this domain as an example to illustrate the different functions/algorithms you are expected to deploy. The file contains both the MDP, described as a tuple like those from Lab 2, and the corresponding optimal $Q$-function.
# 
# To do so, you can run the code
# ```python
# import numpy as np
# 
# mdp_info = np.load('pacman.npz', allow_pickle=True)
# 
# # The MDP is a tuple (X, A, P, c, gamma)
# M = mdp_info['M']
# 
# # We also load the optimal Q-function for the MDP
# Qopt = mdp_info['Q']
# ```
# 
# ---
# 
# In the first activity, you will implement a "simulator of the world". The simulator consists of a function that enables you to sample a transition from a given MDP. You will then use this function, in subsequent activities, to generate the data that your agent will use to learn.

# In[99]:


import numpy as np

mdp_info = np.load('pacman.npz', allow_pickle=True)

# The MDP is a tuple (X, A, P, c, gamma)
M = mdp_info['M']

# We also load the optimal Q-function for the MDP
Qopt = mdp_info['Q']

    


# ---
# 
# #### Activity 1.        
# 
# Write a function named `sample_transition` that receives, as input, a tuple representing an arbitrary MDP as well as two integers, `s` and `a`, corresponding to a state and an action. The function should return a tuple `(s, a, c, s')`, where `c` is the cost associated with performing action `a` in state `s` and `s'` is a state generated from `s` upon selecting action `a`, according to the transition probabilities for the MDP.
# 
# ---

# In[100]:


import numpy as np

def sample_transition(mdp, s, a):
    c = mdp[3][s][a]
    nextstate = np.random.choice(len(mdp[0]), p=mdp[2][a][s])
    return (s, a, c, nextstate)   


# All reinforcement learning algorithms that you will implement can only access the MDP through the function `sample_transition` which, in a sense, simulates an "interaction" of the agent with the environment.
# 
# For example, using the "Pacman" MDP, you could run:
# 
# ```python
# import numpy.random as rnd
# 
# rnd.seed(42)
# 
# # Select random state and action
# s = 106 # State (18, 0, 2)
# a = rnd.randint(len(M[1]))
# 
# s, a, cnew, snew = sample_transition(M, s, a)
# 
# print('Observed transition:\n(', end='')
# print(M[0][s], end=', ')
# print(M[1][a], end=', ')
# print(cnew, end=', ')
# print(M[0][snew], end=')\n')
# 
# # Select random state and action
# s = 12 # State (3, S, 1)
# a = rnd.randint(len(M[1]))
# 
# s, a, cnew, snew = sample_transition(M, s, a)
# 
# print('\nObserved transition:\n(', end='')
# print(M[0][s], end=', ')
# print(M[1][a], end=', ')
# print(cnew, end=', ')
# print(M[0][snew], end=')\n')
# 
# # Select random state and action
# s = 164 # State (28, 0, 3)
# a = rnd.randint(len(M[1]))
# 
# s, a, cnew, snew = sample_transition(M, s, a)
# 
# print('\nObserved transition:\n(', end='')
# print(M[0][s], end=', ')
# print(M[1][a], end=', ')
# print(cnew, end=', ')
# print(M[0][snew], end=')\n')
# ```
# 
# and get, as output:
# 
# ```
# Observed transition:
# ((18, 0, 2), Left, 0.1, (17, 0, 3))
# 
# Observed transition:
# ((3, S, 1), Left, 0.1, (2, S, 2))
# 
# Observed transition:
# ((28, 0, 3), Up, 0.1, (24, S, 2))
# ```

# In[101]:


#test sanity

import numpy.random as rnd

rnd.seed(42)

# Select random state and action
s = 106 # State (18, 0, 2)
a = rnd.randint(len(M[1]))

s, a, cnew, snew = sample_transition(M, s, a)

print('Observed transition:\n(', end='')
print(M[0][s], end=', ')
print(M[1][a], end=', ')
print(cnew, end=', ')
print(M[0][snew], end=')\n')

# Select random state and action
s = 12 # State (3, S, 1)
a = rnd.randint(len(M[1]))

s, a, cnew, snew = sample_transition(M, s, a)

print('\nObserved transition:\n(', end='')
print(M[0][s], end=', ')
print(M[1][a], end=', ')
print(cnew, end=', ')
print(M[0][snew], end=')\n')

# Select random state and action
s = 164 # State (28, 0, 3)
a = rnd.randint(len(M[1]))

s, a, cnew, snew = sample_transition(M, s, a)

print('\nObserved transition:\n(', end='')
print(M[0][s], end=', ')
print(M[1][a], end=', ')
print(cnew, end=', ')
print(M[0][snew], end=')\n')


# ---
# 
# #### Activity 2.        
# 
# Write down a function named `egreedy` that implements an $\epsilon$-greedy policy. Your function should receive, as input, a `numpy` array `Q` with shape `(N,)`, for some integer `N`, and, as an optional argument, a floating point number `eps` with a default value `eps=0.1`. Your function should return... 
# 
# * ... with a probability $\epsilon$, a random index between $0$ and $N-1$.
# * ... with a probability $1-\epsilon$, the index between $0$ and $N-1$ corresponding to the minimum value of `Q`. If more than one such index exists, the function should select among such indices **uniformly at random**.
# 
# **Note:** In the upcoming activities, the array `Q` received by the function `egreedy` will correspond to a row of a $Q$-function, and `N` will correspond to the number of actions.

# In[102]:


def egreedy(Q, eps=0.1):
    n = len(Q)
    rand_number = np.random.rand()
    if rand_number <= eps:
        target = np.random.choice(n)
    else:
        Q_min = np.amin(Q)
        aux = np.abs(Q-Q_min)<1e-3
        target = np.random.choice(n, p = aux/(np.sum(aux)))
    return target


# For example, using the function `Qopt` loaded from the "Pacman" file, you can run:
# 
# ```python
# rnd.seed(42)
# 
# s = 106 # State (18, 0, 2)
# a = egreedy(Qopt[s, :], eps=0)
# print('State:', M[0][s], '- action (eps=0.0):', M[1][a])
# a = egreedy(Qopt[s, :], eps=0.5)
# print('State:', M[0][s], '- action (eps=0.5):', M[1][a])
# a = egreedy(Qopt[s, :], eps=1.0)
# print('State:', M[0][s], '- action (eps=1.0):', M[1][a])
# 
# s = 12 # State (3, S, 1)
# a = egreedy(Qopt[s, :], eps=0)
# print('\nState:', M[0][s], '- action (eps=0.0):', M[1][a])
# a = egreedy(Qopt[s, :], eps=0.5)
# print('State:', M[0][s], '- action (eps=0.5):', M[1][a])
# a = egreedy(Qopt[s, :], eps=1.0)
# print('State:', M[0][s], '- action (eps=1.0):', M[1][a])
# 
# s = 164 # State (28, 0, 3)
# a = egreedy(Qopt[s, :], eps=0)
# print('\nState:', M[0][s], '- action (eps=0.0):', M[1][a])
# a = egreedy(Qopt[s, :], eps=0.5)
# print('State:', M[0][s], '- action (eps=0.5):', M[1][a])
# a = egreedy(Qopt[s, :], eps=1.0)
# print('State:', M[0][s], '- action (eps=1.0):', M[1][a])
# ```
# 
# you will get the output
# 
# ```
# State: (18, 0, 2) - action (eps=0.0): Right
# State: (18, 0, 2) - action (eps=0.5): Right
# State: (18, 0, 2) - action (eps=1.0): Left
# 
# State: (3, S, 1) - action (eps=0.0): Left
# State: (3, S, 1) - action (eps=0.5): Right
# State: (3, S, 1) - action (eps=1.0): Down
# 
# State: (28, 0, 3) - action (eps=0.0): Up
# State: (28, 0, 3) - action (eps=0.5): Up
# State: (28, 0, 3) - action (eps=1.0): Up
# ```
# 
# Note that, depending on the order and number of calls to functions in the random library you may get slightly different results.

# In[103]:


#test sanity

rnd.seed(42)

s = 106 # State (18, 0, 2)
a = egreedy(Qopt[s, :], eps=0)
print('State:', M[0][s], '- action (eps=0.0):', M[1][a])
a = egreedy(Qopt[s, :], eps=0.5)
print('State:', M[0][s], '- action (eps=0.5):', M[1][a])
a = egreedy(Qopt[s, :], eps=1.0)
print('State:', M[0][s], '- action (eps=1.0):', M[1][a])

s = 12 # State (3, S, 1)
a = egreedy(Qopt[s, :], eps=0)
print('\nState:', M[0][s], '- action (eps=0.0):', M[1][a])
a = egreedy(Qopt[s, :], eps=0.5)
print('State:', M[0][s], '- action (eps=0.5):', M[1][a])
a = egreedy(Qopt[s, :], eps=1.0)
print('State:', M[0][s], '- action (eps=1.0):', M[1][a])

s = 164 # State (28, 0, 3)
a = egreedy(Qopt[s, :], eps=0)
print('\nState:', M[0][s], '- action (eps=0.0):', M[1][a])
a = egreedy(Qopt[s, :], eps=0.5)
print('State:', M[0][s], '- action (eps=0.5):', M[1][a])
a = egreedy(Qopt[s, :], eps=1.0)
print('State:', M[0][s], '- action (eps=1.0):', M[1][a])


# ---
# 
# #### Activity 3. 
# 
# Write a function `mb_learning` that implements the model-based reinforcement learning algorithm discussed in class. Your function should receive as input arguments 
# 
# * A tuple, `mdp`, containing the description of an **arbitrary** MDP. The structure of the tuple is similar to that provided in the example above. 
# * An integer, `n`, corresponding the number of steps that your algorithm should run.
# *  A numpy array `qinit` with as many rows as the number of states in `mdp` and as many columns as the number of actions in `mdp`. The matrix `qinit` should be used to initialize the $Q$-function being learned by your function.
# * A tuple, `Pinit`, with as many elements as the number of actions in `mdp`. Each element of `Pinit` corresponds to square numpy arrays with as many rows/columns as the number of states in `mdp` and can be **any** transition probability matrix. The matrices in `Pinit` should be used to initialize the transition probability matrices of the model being learned by your function.
# * A numpy array `cinit` with as many rows as the number of states in `mdp` and as many columns as the number of actions in `mdp`. The matrix `cinit` should be used to initialize the cost function of the model being learned by your function.
# 
# Your function should simulate an interaction of `n` steps between the agent and the environment, during which it should perform `n` iterations of the model-based RL algorithm seen in class. In particular, it should learn the transition probabilities and cost function from the interaction between the agent and the environment, and use these to compute the optimal $Q$-function. The transition probabilities, cost and $Q$-functions to be learned should be initialized using `Pinit`, `cinit` and `qinit`, respectively. 
# 
# Note that, at each step of the interaction,
# 
# * The agent should observe the current state, and select an action using an $\epsilon$-greedy policy with respect to its current estimate of the optimal $Q$-values. You should use the function `egreedy` from Activity 2, with $\epsilon=0.15$. 
# * Given the state and action, you must then compute the cost and generate the next state, using `mdp` and the function `sample_transition` from Activity 1.
# * With this transition information (state, action, cost, next-state), you can now perform an update to the transition probabilities, cost function, and $Q$-function.
# * When updating the components $(x,a)$ of the model, use the step-size
# 
# $$\alpha_t=\frac{1}{N_t(x,a)+1},$$
# 
# where $N_t(x,a)$ is the number of visits to the pair $(x,a)$ up to time step $t$.
# 
# Your function should return a tuple containing:
# 
# *  A numpy array with as many rows as the number of states in `mdp` and as many columns as the number of actions in `mdp`, corresponding to the learned $Q$-function.
# * A tuple with as many elements as the number of actions in `mdp`. The element $a$ of the tuple corresponds to a square numpy array with as many rows/columns as the number of states in `mdp`, corresponding to the learned transition probabilities for action $a$.
# * A numpy array with as many rows as the number of states in `mdp` and as many columns as the number of actions in `mdp`, corresponding to the learned cost function.
# 
# ---

# In[104]:


import copy
def mb_learning(mdp, n, qinit, Pinit, cinit):
    n_states = len(mdp[0])
    n_actions = len(mdp[1])
    gamma = mdp[4]
    Q = qinit
    P = Pinit
    c = cinit
    Nt = {(x,a): 0 for x in range(n_states) for a in range(n_actions)}
    S = np.random.choice(n_states)
    for i in range(n):
        A = egreedy(Q[S,:], eps=0.15)
        transition = sample_transition(mdp, S, A)
        cost = transition[2]
        next_state = transition[3]
        Nt[(S, A)] = Nt[(S, A)] + 1
        alpha = 1/(Nt[(S, A)] + 1)
        c[S][A] = c[S][A] + alpha* (cost - c[S][A])
        for x in range(n_states):
            if x == next_state:
                P[A][S][x] = P[A][S][x] + alpha* (1 - P[A][S][x])
            else:
                P[A][S][x] = P[A][S][x] + alpha* (0 - P[A][S][x])
        sum_Q = 0
        for x in range(n_states):
            sum_Q = sum_Q + P[A][S][x]* np.min(Q[x,:])
        Q[S][A] = c[S][A] + gamma*sum_Q
        S = copy.deepcopy(next_state)
    return (Q,P,c)
        


# As an example using the "Pacman" MDP, we could run:
# 
# ```python
# rnd.seed(42)
# 
# # Initialize transition probabilities
# pinit = ()
# 
# for a in range(len(M[1])):
#     pinit += (np.eye(len(M[0])),)
# 
# # Initialize cost function
# cinit = np.zeros((len(M[0]), len(M[1])))
# 
# # Initialize Q-function
# qinit = np.zeros((len(M[0]), len(M[1])))
# 
# # Run 1000 steps of model-based learning
# qnew, pnew, cnew = mb_learning(M, 1000, qinit, pinit, cinit)
# 
# # Compare the learned Q with the optimal Q
# print('Error in Q after 1000 steps:', np.linalg.norm(qnew - Qopt))
# 
# # Run 1000 additional steps of model-based learning
# qnew, pnew, cnew = mb_learning(M, 1000, qnew, pnew, cnew)
# 
# # Compare once again the learned Q with the optimal Q
# print('Error in Q after 2000 steps:', np.linalg.norm(qnew - Qopt))
# ```
# 
# to get
# 
# ```
# Error in Q after 1000 steps: 19.916238521031588
# Error in Q after 2000 steps: 19.86435667980359
# ```
# 
# Note that, even if the seed is fixed, the numerical values may differ somewhat from those above.

# In[105]:


#test sanity

rnd.seed(42)

# Initialize transition probabilities
pinit = ()

for a in range(len(M[1])):
    pinit += (np.eye(len(M[0])),)

# Initialize cost function
cinit = np.zeros((len(M[0]), len(M[1])))

# Initialize Q-function
qinit = np.zeros((len(M[0]), len(M[1])))

# Run 1000 steps of model-based learning
qnew, pnew, cnew = mb_learning(M, 1000, qinit, pinit, cinit)

# Compare the learned Q with the optimal Q
print('Error in Q after 1000 steps:', np.linalg.norm(qnew - Qopt))

# Run 1000 additional steps of model-based learning
qnew, pnew, cnew = mb_learning(M, 1000, qnew, pnew, cnew)

# Compare once again the learned Q with the optimal Q
print('Error in Q after 2000 steps:', np.linalg.norm(qnew - Qopt))


# ### 3. Model-free learning
# 
# You will now implement both $Q$-learning and SARSA.

# ---
# 
# #### Activity 4. 
# 
# Write a function `qlearning` that implements the $Q$-learning algorithm discussed in class. Your function should receive as input arguments 
# 
# * A tuple, `mdp`, containing the description of an **arbitrary** MDP. The structure of the tuple is similar to that provided in the examples above. 
# * An integer, `n`, corresponding he number of steps that your algorithm should run.
# *  A `numpy` array `qinit` with as many rows as the number of states in `mdp` and as many columns as the number of actions in `mdp`. The matrix `qinit` should be used to initialize the $Q$-function being learned by your function.
# 
# Your function should simulate an interaction of `n` steps between the agent and the environment, during which it should perform `n` iterations of the $Q$-learning algorithm seen in class. In particular, it should learn optimal $Q$-function. The $Q$-function to be learned should be initialized using `qinit`. 
# 
# Note that, at each step of the interaction,
# 
# * The agent should observe the current state, and select an action using an $\epsilon$-greedy policy with respect to its current estimate of the optimal $Q$-values. You should use the function `egreedy` from Activity 2, with $\epsilon=0.15$. 
# * Given the state and action, you must then compute the cost and generate the next state, using `mdp` and the function `sample_transition` from Activity 1.
# * With this transition information (state, action, cost, next-state), you can now perform an update to the $Q$-function.
# * When updating the components $(x,a)$ of the model, use the step-size $\alpha=0.3$.
# 
# Your function should return a `numpy` array with as many rows as the number of states in `mdp` and as many columns as the number of actions in `mdp`, corresponding to the learned $Q$-function.
# 
# ---

# In[106]:


def qlearning(mdp, n, qinit):
    n_states = len(mdp[0])
    gamma = mdp[4]
    Q = qinit
    alpha = 0.3
    S = np.random.choice(n_states) 
    for i in range(n):
        A = egreedy(Q[S,:], eps=0.15)
        transition = sample_transition(mdp, S, A)
        cost = transition[2]
        next_state = transition[3]
        Q[S][A] = Q[S][A] + alpha* (cost + gamma*min(Q[next_state,:]) - Q[S][A])
        S = copy.deepcopy(next_state)
    return Q


# As an example using the "Pacman" MDP, we could run:
# 
# ```python
# rnd.seed(42)
# 
# # Initialize Q-function
# qinit = np.zeros((len(M[0]), len(M[1])))
# 
# # Run 1000 steps of model-based learning
# qnew = qlearning(M, 1000, qinit)
# 
# # Compare the learned Q with the optimal Q
# print('Error in Q after 1000 steps:', np.linalg.norm(qnew - Qopt))
# 
# # Run 1000 additional steps of model-based learning
# qnew = qlearning(M, 1000, qnew)
# 
# # Compare once again the learned Q with the optimal Q
# print('Error in Q after 2000 steps:', np.linalg.norm(qnew - Qopt))
# ```
# 
# to get
# 
# ```
# Error in Q after 1000 steps: 19.944334092242844
# Error in Q after 2000 steps: 19.91105731381223
# ```
# 
# Once again, even if the seed is fixed, the numerical values may differ somewhat from those above.

# In[107]:


#test sanity 
rnd.seed(42)

# Initialize Q-function
qinit = np.zeros((len(M[0]), len(M[1])))

# Run 1000 steps of model-based learning
qnew = qlearning(M, 1000, qinit)

# Compare the learned Q with the optimal Q
print('Error in Q after 1000 steps:', np.linalg.norm(qnew - Qopt))

# Run 1000 additional steps of model-based learning
qnew = qlearning(M, 1000, qnew)

# Compare once again the learned Q with the optimal Q
print('Error in Q after 2000 steps:', np.linalg.norm(qnew - Qopt))


# ---
# 
# #### Activity 5. 
# 
# Write a function `sarsa` that implements the SARSA algorithm discussed in class. Your function should receive as input arguments 
# 
# * A tuple, `mdp`, containing the description of an **arbitrary** MDP. The structure of the tuple is similar to that provided in the examples above. 
# * An integer, `n`, corresponding he number of steps that your algorithm should run.
# *  A `numpy` array `qinit` with as many rows as the number of states in `mdp` and as many columns as the number of actions in `mdp`. The matrix `qinit` should be used to initialize the $Q$-function being learned by your function.
# 
# Your function should simulate an interaction of `n` steps between the agent and the environment, during which it should perform `n` iterations of the SARSA algorithm seen in class. The $Q$-function to be learned should be initialized using `qinit`. 
# 
# Note that, at each step of the interaction,
# 
# * The agent should observe the current state, and select an action using an $\epsilon$-greedy policy with respect to its current estimate of the optimal $Q$-values. You should use the function `egreedy` from Activity 2, with $\epsilon=0.15$. **Do not adjust the value of $\epsilon$ during learning.**
# * Given the state and action, you must then compute the cost and generate the next state, using `mdp` and the function `sample_transition` from Activity 1.
# * With this transition information (state, action, cost, next-state), you can now perform an update to the $Q$-function.
# * When updating the components $(x,a)$ of the model, use the step-size $\alpha=0.3$.
# 
# Your function should return a `numpy` array with as many rows as the number of states in `mdp` and as many columns as the number of actions in `mdp`, corresponding to the learned $Q$-function.
# 
# ---

# In[108]:


def sarsa(mdp, n, qinit):
    n_states = len(mdp[0])
    gamma = mdp[4]
    Q = qinit
    alpha = 0.3
    S = np.random.choice(n_states) 
    A = egreedy(Q[S,:], eps=0.15)
    for i in range(n):
        transition = sample_transition(mdp, S, A)
        cost = transition[2]
        next_state = transition[3]
        next_action = egreedy(Q[next_state,:], eps=0.15)
        Q[S][A] = Q[S][A] + alpha*(cost + gamma*Q[next_state][next_action] - Q[S][A])
        S = copy.deepcopy(next_state)
        A = copy.deepcopy(next_action)
    return Q


# As an example using the "Pacman" MDP, we could run:
# 
# ```python
# rnd.seed(42)
# 
# # Initialize Q-function
# qinit = np.zeros((len(M[0]), len(M[1])))
# 
# # Run 1000 steps of model-based learning
# qnew = sarsa(M, 1000, qinit)
# 
# # Compare the learned Q with the optimal Q
# print('Error in Q after 1000 steps:', np.linalg.norm(qnew - Qopt))
# 
# # Run 1000 additional steps of model-based learning
# qnew = sarsa(M, 1000, qnew)
# 
# # Compare once again the learned Q with the optimal Q
# print('Error in Q after 2000 steps:', np.linalg.norm(qnew - Qopt))
# ```
# 
# to get
# 
# ```
# Error in Q after 1000 steps: 19.944134856701385
# Error in Q after 2000 steps: 19.91302892958602
# ```

# In[113]:


#test sanity 
rnd.seed(42)

# Initialize Q-function
qinit = np.zeros((len(M[0]), len(M[1])))

# Run 1000 steps of model-based learning
qnew = sarsa(M, 1000, qinit)

# Compare the learned Q with the optimal Q
print('Error in Q after 1000 steps:', np.linalg.norm(qnew - Qopt))

# Run 1000 additional steps of model-based learning
qnew = sarsa(M, 1000, qnew)

# Compare once again the learned Q with the optimal Q
print('Error in Q after 2000 steps:', np.linalg.norm(qnew - Qopt))


# You can also run the following code, to compare the performance of the three methods.
# 
# ```python
# %matplotlib inline
# 
# import matplotlib.pyplot as plt
# from tqdm import trange
# 
# STEPS = 10
# ITERS = 1000
# RUNS  = 10
# 
# iters = range(0, STEPS * ITERS + 1, STEPS)
# 
# # Error matrices
# Emb = np.zeros(ITERS + 1)
# Eql = np.zeros(ITERS + 1)
# Ess = np.zeros(ITERS + 1)
# 
# Emb[0] = np.linalg.norm(Qopt) * RUNS
# Eql[0] = Emb[0]
# Ess[0] = Emb[0]
# 
# rnd.seed(42)
# 
# for n in trange(RUNS):
# 
#     # Initialization
#     pmb = ()
#     for a in range(len(M[1])):
#         pmb += (np.eye(len(M[0])),)
#     cmb = np.zeros((len(M[0]), len(M[1])))
#     qmb = np.zeros((len(M[0]), len(M[1])))
# 
#     qql = np.zeros((len(M[0]), len(M[1])))
# 
#     qss = np.zeros((len(M[0]), len(M[1])))
# 
#     # Run evaluation
#     for t in range(ITERS):
#         qmb, pmb, cmb = mb_learning(M, STEPS, qmb, pmb, cmb)
#         Emb[t + 1] += np.linalg.norm(Qopt - qmb)
# 
#         qql = qlearning(M, STEPS, qql)
#         Eql[t + 1] += np.linalg.norm(Qopt - qql)
# 
#         qss = sarsa(M, STEPS, qss)
#         Ess[t + 1] += np.linalg.norm(Qopt - qss)
#         
# Emb /= RUNS
# Eql /= RUNS
# Ess /= RUNS
# 
# plt.figure()
# plt.plot(iters, Emb, label='Model based learning')
# plt.plot(iters, Eql, label='Q-learning')
# plt.plot(iters, Ess, label='SARSA')
# plt.legend()
# plt.xlabel('N. iterations')
# plt.ylabel('Error in $Q$-function')
# ```
# 
# As the output, you should observe a plot similar to the one below.
# 
# <img src="plot.png" align="left">

# In[110]:


#test sanity 2
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
from tqdm import trange

STEPS = 10
ITERS = 1000
RUNS  = 10

iters = range(0, STEPS * ITERS + 1, STEPS)

# Error matrices
Emb = np.zeros(ITERS + 1)
Eql = np.zeros(ITERS + 1)
Ess = np.zeros(ITERS + 1)

Emb[0] = np.linalg.norm(Qopt) * RUNS
Eql[0] = Emb[0]
Ess[0] = Emb[0]

rnd.seed(42)

for n in trange(RUNS):

    # Initialization
    pmb = ()
    for a in range(len(M[1])):
        pmb += (np.eye(len(M[0])),)
    cmb = np.zeros((len(M[0]), len(M[1])))
    qmb = np.zeros((len(M[0]), len(M[1])))

    qql = np.zeros((len(M[0]), len(M[1])))

    qss = np.zeros((len(M[0]), len(M[1])))

    # Run evaluation
    for t in range(ITERS):
        qmb, pmb, cmb = mb_learning(M, STEPS, qmb, pmb, cmb)
        Emb[t + 1] += np.linalg.norm(Qopt - qmb)

        qql = qlearning(M, STEPS, qql)
        Eql[t + 1] += np.linalg.norm(Qopt - qql)

        qss = sarsa(M, STEPS, qss)
        Ess[t + 1] += np.linalg.norm(Qopt - qss)

Emb /= RUNS
Eql /= RUNS
Ess /= RUNS

plt.figure()
plt.plot(iters, Emb, label='Model based learning')
plt.plot(iters, Eql, label='Q-learning')
plt.plot(iters, Ess, label='SARSA')
plt.legend()
plt.xlabel('N. iterations')
plt.ylabel('Error in $Q$-function')


# ---
# 
# #### Activity 6.
# 
# **Based on the results you obtained when running the above code with your algorithms**, discuss the differences observed between the performance of the three methods.
# 
# ---

# <span style="color:blue">
# Based on the study of the figure above, we can conclude that the Model Based Learning has the best performance, as the error in the Q-function decreases faster in less iterations than the other two models. 
#     
# Because the curves have a similar evolution, the performance of SARSA and Q-learning is relatively similar. The error in the Q-function drops from 20 to 12 (with 10000 iterations) for both models.
#     
# Model Based Learning requires the use of initial transition probabilities, an initial cost function, and an initial Q-function. SARSA and Q-learning, on the other hand, only require an initial Q-function.    
#     
# Besides, Q-learning is an off-policy algorithm. In computing the target, $c_t + \gamma min_{a\in A} Q(x_{t+1}, a)$, it uses as the next action that of a policy different from the one that the agent is currently using (the greedy action). 
# 
# In contrast, SARSA uses the actual policy all the time, hence it is an on-policy algorithm.  In
# computing the target, $c_t + \gamma Q(x_{t+1}, a_{t+1})$, the next action is chosen from the agent’s current policy. 
#     
# It should be highlighted that because the SARSA model is an on-policy model, it will not be able to converge to an optimal value because it needs an improvement in policy.
#     
# Also, it is important to note that the error never reaches 0, because $\epsilon$ is fixed at 0.15, which means that there is always a 15% probability of doing a sub-optimal action, as the agent is encouraged to explore the space of actions. 
#     
#   
#     
# </span>
