import nltk
import random, string
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict
import HMM
from HMM import HiddenMarkovModel
from HMM import unsupervised_HMM, semisupervised_HMM
from syllable_test2 import sylco
import numpy as np
import pickle

# Load transition matrix, observation matrix, and maps from state to part of speech, 
# part of speech to state, observation to word, word to observation from saved 
# HMM model and poem generation run
A = np.loadtxt("transition_matrix.txt")
O = np.loadtxt("observation_matrix.txt")
state_POS_map = pickle.load(open("state_POS_map.p", "rb"))
POS_state_map = pickle.load(open("POS_state_map.p", "rb"))
observation_word_map = pickle.load(open("observation_word_map.p", "rb"))
word_observation_map = pickle.load(open("word_observation_map.p", "rb"))

# Print out the ten words most likely to be generated when in each state.
for i in range(O.shape[0]):
    row = O[i]
    row_index_list = []
    for j in range(len(row)):
        row_index_list.append((row[j], j))
    row_index_list.sort(reverse=True)
    for k in range(10):
        print i, observation_word_map[row_index_list[k][1]]
    print "\n"

# Print out the transition matrix rounded so that transitions with less than 30%
# probability are output as having 0 probability. Also, the printed values of the 
# transition matrix are rounded as well.
np.set_printoptions(precision=4)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if A[i, j] < 0.3:
            A[i, j] = 0
np.savetxt("transition_matrix_rounded.txt", A, fmt='%.3f')
    
# Print out most likely state transition (from state is first, to state is second,
# probability of transition is last)
np.set_printoptions(precision=4)
for i in range(A.shape[0]):
    print i, np.argmax(A[i]), np.max(A[i])
