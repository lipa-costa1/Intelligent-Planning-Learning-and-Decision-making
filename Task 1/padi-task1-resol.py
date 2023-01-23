#!/usr/bin/env python
# coding: utf-8

# # Learning and Decision Making

# ## Laboratory 1: Markov chains
# 
# In the end of the lab, you should export the notebook to a Python script (File >> Download as >> Python (.py)). Your file should be named `padi-lab1-groupXX.py`, where the `XX` corresponds to your group number and should be submitted to the e-mail <adi.tecnico@gmail.com>. 
# 
# Make sure...
# 
# * **... that the subject is of the form `[<group n.>] LAB <lab n.>`.** 
# 
# * **... to strictly respect the specifications in each activity, in terms of the intended inputs, outputs and naming conventions.** 
# 
# In particular, after completing the activities you should be able to replicate the examples provided (although this, in itself, is no guarantee that the activities are correctly completed).
# 
# ### 1. The Markov chain model
# 
# Consider once again the Pacman modeling problem described in the Homework and for which you wrote a Markov chain model. In this lab, you will consider a larger version of the Pacman problem, described by the diagram:
# 
# <img src="pacman-big.png">
# 
# Recall that your chain should describe the motion of the single ghost moving in the environment, where: 
# 
# * The cells are numbered from 1 to 35, as indicated by the blue numbers;
# * At each moment, the ghost is in one of the 35 cells; at the next time step, it will move to one of the adjacent cells with equal probability;
# * The cell in the bottom left corner (cell `29`) is adjacent, to the left, to the cell in the bottom right corner (cell `35`). In other words, if the ghost "moves left" in cell `29` it will end up in cell `35` and vice-versa.
# 
# In this first activity, you will implement your Markov chain model in Python. You will start by loading the transition probability matrix from a `numpy` binary file, using the `numpy` function `load`. You will then consider the state space to consist of all possible cells in the environment, each represented as a string. For example, if the environment has 10 cells, the states should include the strings `'1'` to `'10'`. 

# ---
# 
# #### Activity 1.        
# 
# Write a function named `load_chain` that receives, as input, a string corresponding to the name of a file containing a transition probability matrix for some Pacman maze to be loaded and returns, as output, a two-element tuple corresponding to the Markov chain that describes the movement of the ghost, where:
# 
# * ... the first element is a tuple containing an enumeration of the state-space (i.e., each element of the tuple corresponds to a state of the chain, represented as a string).
# * ... the second element is a `numpy` array corresponding to the transition probability matrix for the chain.
# 
# **Note:** The file provided contains the transition probability matrix for the maze in the diagram above. However, your code will be tested with mazes of different sizes, so **make sure not to hard-code the size of the environment into your code**.
# 
# ---

# In[46]:


import numpy as np
tpm = np.load('pacman-big.npy')
print(tpm)


# In[121]:


def load_chain(data):
    tpm=np.load(data)
    states = []
    for i in range(1,len(tpm)+1):
        states.append(str(i))
    M = (tuple(states), tpm)
    return M


# We provide below an example of application of the function, that you can use as a first "sanity check" for your code. Note, however, that the fact that you can replicate the examples below is not indicative that your code is correct.
# 
# ```python
# M = load_chain('pacman-big.npy')
# 
# print('Number of states:', len(M[0]))
# print('Type of states:', type(M[0][0]))
# print('First state:', M[0][0])
# print('\nTransition probability matrix (type):', type(M[1]))
# print('Transition probability matrix (dimension):', M[1].shape)```
# 
# Output:
# ```
# Number of states: 35
# Type of states: <class 'str'>
# First state: 1
# 
# Transition probability matrix (type): <class 'numpy.ndarray'>
# Transition probability matrix (dimension): (35, 35)
# ```

# In[120]:


#teste sanity
M = load_chain('pacman-big.npy')

print('Number of states:', len(M[0]))
print('Type of states:', type(M[0][0]))
print('First state:', M[0][0])
print('\nTransition probability matrix (type):', type(M[1]))
print('Transition probability matrix (dimension):', M[1].shape)


# In the next activity, you will use the Markov chain model to evaluate the likelihood of any given path for the ghost.
# 
# ---
# 
# #### Activity 2.
# 
# Write a function `prob_trajectory` that receives, as inputs, 
# 
# * ... a Markov chain in the form of a tuple like the one returned by the function in Activity 1;
# * ... a trajectory, corresponding to a sequence of states (i.e., a tuple or list of strings, each string corresponding to a state).
# 
# Your function should return, as output, a floating point number corresponding to the probability of observing the provided trajectory, taking the first state in the trajectory as initial state. 
# 
# ---

# In[122]:


def prob_trajectory(M, traj):
    prod = 1
    prob_matrix = M[1]
    for i in range(len(traj)-1):
        trajec = (int(traj[i])-1, int(traj[i+1])-1)
        prod = prod * prob_matrix[trajec[0]][trajec[1]]
    return prod


# Example of application of the function with the chain $M$ from Activity 1.
# 
# ```python
# print("Prob. of trajectory ('3', '9', '15'):", prob_trajectory(M, ('3', '9', '15')))
# print("Prob. of trajectory ('6', '7', '12', '11', '10'):", prob_trajectory(M, ('6', '7', '12', '11', '10')))
# print("Prob. of trajectory ('10', '11', '17'):", prob_trajectory(M, ('10', '11', '17')))
# print("Prob. of trajectory ('34', '35', '29'):", prob_trajectory(M, ('34', '35', '29')))
# ```
# 
# Output:
# ```
# Prob. of trajectory ('3', '9', '15'): 0.16666666666666666
# Prob. of trajectory ('6', '7', '12', '11', '10'): 0.0625
# Prob. of trajectory ('10', '11', '17'): 0.0
# Prob. of trajectory ('34', '35', '29'): 0.25
# ```
# 
# Note that your function should work with **any** Markov chain that is specified as a tuple like the one from Activity 1.

# In[123]:


#teste sanity
print("Prob. of trajectory ('3', '9', '15'):", prob_trajectory(M, ('3', '9', '15')))
print("Prob. of trajectory ('6', '7', '12', '11', '10'):", prob_trajectory(M, ('6', '7', '12', '11', '10')))
print("Prob. of trajectory ('10', '11', '17'):", prob_trajectory(M, ('10', '11', '17')))
print("Prob. of trajectory ('34', '35', '29'):", prob_trajectory(M, ('34', '35', '29')))


# ### 2. Stability

# The next activities explore the notion of *stationary distribution* for the chain, a central concept in the the PageRank algorithm.
# 
# ---
# 
# #### Activity 3
# 
# Write a function `stationary_dist` that receives, as input, a Markov chain in the form of a tuple like the one returned by the function in Activity 1. Your function should return, as output, a `numpy` array corresponding to a row vector containing the stationary distribution for the chain.
# 
# **Note:** The stationary distribution is a *left* eigenvector of the transition probability matrix associated to the eigenvalue 1. As such, you may find useful the numpy function `numpy.linalg.eig`. Also, recall that the stationary distribution is *a distribution*.
# 
# ---

# In[124]:


def stationary_dist(M):
    eigenval = np.linalg.eig(M[1].T)[0].real
    eigenvec = np.linalg.eig(M[1].T)[1]
    targetindex = np.where(np.isclose(eigenval,1))
    targetvector = eigenvec.T[targetindex[0][0]].real
    statdist = targetvector / np.sum(targetvector)
    return statdist


# Example of application of the function with the chain $M$ from Activity 1.
# 
# ```python
# u_star = stationary_dist(M)
# 
# print('Stationary distribution:')
# print(np.round(u_star, 2))
# 
# u_prime = u_star.dot(M[1])
# 
# print('\nIs u* * P = u*?', np.all(np.isclose(u_prime, u_star)))
# ```
# 
# Output:
# ```
# Stationary distribution:
# [0.03 0.03 0.05 0.03 0.03 0.03 0.03 0.04 0.03 0.03 0.03 0.03 0.05 0.03
#  0.07 0.03 0.04 0.03 0.03 0.04 0.04 0.05 0.03 0.02 0.02 0.02 0.02 0.02
#  0.01 0.   0.   0.01 0.01 0.01 0.01]
# 
# Is u* * P = u*? True
# ```

# In[125]:


#teste sanity
u_star = stationary_dist(M)

print('Stationary distribution:')
print(np.round(u_star, 2))

u_prime = u_star.dot(M[1])

print('\nIs u* * P = u*?', np.all(np.isclose(u_prime, u_star)))


# To complement Activity 3, you will now empirically establish that the chain is ergodic, i.e., no matter where the ghost starts, its visitation frequency will eventually converge to the stationary distribution.
# 
# ---
# 
# #### Activity 4.
# 
# Write a function `compute_dist` that receives, as inputs, 
# 
# * ... a Markov chain in the form of a tuple like the one returned by the function in Activity 1;
# * ... a row vector (a numpy array) corresponding to the initial distribution for the chain;
# * ... an integer $N$, corresponding to the number of steps that the bot is expected to take.
# 
# Your function should return, as output, a row vector (a `numpy` array) containing the distribution after $N$ steps of the chain.
# 
# ---

# In[126]:


def compute_dist(M,init_dist,n):
    power=np.linalg.matrix_power(M[1], n)
    return(np.dot(init_dist,power))
    


# In[127]:


#teste sanity
import numpy.random as rnd

# Number of states
nS = len(M[0])

rnd.seed(42)

# Initial random distribution
u = rnd.random((1, nS))
u = u / np.sum(u)

# Distrbution after 100 steps
v = compute_dist(M, u, 100)
print('\nIs u * P^100 = u*?', np.all(np.isclose(v, u_star)))

# Distrbution after 2000 steps
v = compute_dist(M, u, 2000)
print('\nIs u * P^2000 = u*?', np.all(np.isclose(v, u_star)))


# Example of application of the function with the chain $M$ from Activity 1.
# 
# ```python
# import numpy.random as rnd
# 
# # Number of states
# nS = len(M[0])
# 
# rnd.seed(42)
# 
# # Initial random distribution
# u = rnd.random((1, nS))
# u = u / np.sum(u)
# 
# # Distrbution after 100 steps
# v = compute_dist(M, u, 100)
# print('\nIs u * P^100 = u*?', np.all(np.isclose(v, u_star)))
# 
# # Distrbution after 2000 steps
# v = compute_dist(M, u, 2000)
# print('\nIs u * P^2000 = u*?', np.all(np.isclose(v, u_star)))
# ```
# 
# Output:
# ```
# Is u * P^100 = u*? False
# 
# Is u * P^2000 = u*? True
# ```

# Is the chain ergodic? Justify, based on the results above.

# #### Para provar que é ergódica, precisamos primeiro de provar que é irredutível e aperiódica. 
# #### A Markov Chain é irredutível, pois se olharmos para o diagrama do Pacman, todos os estados comunicam uns com os outros, ou seja, para qualquer estado, é possível alcançar qualquer outro estado num número finito de passos. 
# #### A Markov Chain é aperiódica, pois todos os estados têm o periodo igual a 1, isto é, o máximo divisor comum do número possível de passos entre 2 visitas consecutivas a um certo estado é 1. Por exemplo, se o ghost está no estado 1, consegue regressar ao estado 1 em 2 passos, com a trajetória ('1','2','1'). Mas, também é possível regressar em 27 passos, com a trajetória ('1','2','3','9','15','22','23','27','32','31','30','29','35','34','33','28','24','25','26','19','18','17','16','15','9','3','2','1'). O máximo divisor comum entre 2 e 27 é 1. O mesmo acontece com os restantes estados. 
# #### Então, é possível dizer que a Markov chain tem um distribuição estacionária u*.
# 
# #### Se para qualquer distribuição inicial u0, lim t -> inf (u0 * P^t) = u*, a Markov Chain é ergódica. 
# #### Escolhendo uma distribuição inicial qualquer, a condição u0 * P^2000= u* é verificada, o que significa que após 2000 passos, a distribuição inicial converge para a estacionária, sendo a markov chain ergódica. 

# ### 3. Simulation
# 
# In this part of the lab, you will *simulate* the actual bot, and empirically compute the visitation frequency of each state.

# ---
# 
# #### Activity 5
# 
# Write down a function `simulate` that receives, as inputs, 
# 
# * ... a Markov chain in the form of a tuple like the one returned by the function in Activity 1;
# * ... a row vector (a `numpy` array) corresponding to the initial distribution for the chain;
# * ... an integer $N$, corresponding to the number of steps that the bot is expected to take.
# 
# Your function should return, as output, a tuple containing a trajectory containing $N$ states, obtained from the initial distribution provided. Each element in the tuple should be a string corresponding to a state index.
# 
# ---
# 
# **Note:** You may find useful to import the numpy module `numpy.random`.

# In[128]:


def simulate(M, init_dist, n):
    traj = []
    initialstate = np.random.choice(M[0], p = init_dist[0])
    traj.append(str(initialstate))
    for i in range(1,n):
        target=int(initialstate)-1
        nextstate = int(np.random.choice(M[0],p=M[1][target]))
        traj.append(str(nextstate))
        initialstate = nextstate
    return tuple(traj)


# Example of application of the function with the chain $M$ from Activity 1.
# 
# ```python
# # Number of states
# nS = len(M[0])
# 
# # Initial, uniform distribution
# u = np.ones((1, nS)) / nS
# 
# np.random.seed(42)
# 
# # Simulate short trajectory
# traj = simulate(M, u, 10)
# print('Small trajectory:', traj)
# 
# # Simulate a long trajectory
# traj = simulate(M, u, 10000)
# print('End of large trajectory:', traj[-10:])
# ```
# 
# Output:
# ```
# Small trajectory: ('14', '15', '16', '17', '10', '11', '10', '17', '16', '17')
# End of large trajectory: ('13', '8', '1', '8', '1', '8', '1', '8', '13', '20')
# ```
# 
# Note that, even if the seed is fixed, it is possible that your trajectories are slightly different.

# In[129]:


#teste sanity
# Number of states
nS = len(M[0])

# Initial, uniform distribution
u = np.ones((1, nS)) / nS

np.random.seed(42)

# Simulate short trajectory
traj = simulate(M, u, 10)
print('Small trajectory:', traj)

# Simulate a long trajectory
traj = simulate(M, u, 10000)
print('End of large trajectory:', traj[-10:])


# ---
# 
# #### Activity 6
# 
# Use the function `simulate` from Activity #5 to generate a 20,000-step trajectory. Plot the histogram of the obtained trajectory using the function `hist` from the module `matplotlib.pyplot`. Make sure that the histogram has one bin for each state. Compare the relative frequencies with the result of Activity #3.
# 
# **Note**: Don't forget to load `matplotlib`. 
# 
# **Note 2**: Recall that the states in the trajectory obtained from the function `simulate` consist of *strings*, which should be converted to state indices to match the entries in the distribution computed in Activity #3.
# 
# ---

# In[131]:


import matplotlib.pyplot as plt


# In[130]:


np.random.seed(2022)
traj = simulate(M, u, 20000)


# In[133]:


trajectory = list(traj)
for i in range(len(trajectory)): 
    trajectory[i] = int(trajectory[i]) 


# In[174]:


d = np.diff(np.unique(trajectory)).min()
left_bin = min(trajectory) - float(d)/2
right_bin = max(trajectory) + float(d)/2


# In[175]:


plt.figure(figsize=(15,6))
plt.hist(trajectory, bins=np.arange(left_bin, right_bin + d, d), density=True, color = "orchid", label = 'Empirical distribution', edgecolor='black')
plt.plot(range(1,36), u_star, 'o', color='black', label = 'Theoretical distribution')
plt.xlabel('States')
plt.ylabel('Probability')
plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
plt.show()


# #### Compare the relative frequencies with the result of Activity #3. 
# #### Ao comparar a distribuição estacionária teórica obtida na atividade 3, e a distribuição estacionária estimada, obtida na atividade 6, podemos concluir que os resultados são muito próximos. 
# #### Estes resultados são expectáveis pois o número de passos considerado (20 000) é muito alto, e como a Markov Chain é ergódica, in the long run, a proporção de tempo que o ghost está em cada estado vai convergir para a distribuição estacionária, qualquer que seja o estado inicial.
