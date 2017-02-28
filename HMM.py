########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5 solutions
########################################

import random
import numpy as np
from syllable_test2 import sylco

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models. Can generate lines of poems with 
    a given number of syllables via unsupervised, supervised, or semi-supervised 
    learning.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state. 

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Note that alpha_j(0) is already correct for all j's.
        # Calculate alpha_j(1) for all j's.
        for curr in range(self.L):
            alphas[1][curr] = self.A_start[curr] * self.O[curr][x[0]]

        # Calculate alphas throughout sequence.
        for t in range(1, M):
            # Iterate over all possible current states.
            for curr in range(self.L):
                prob = 0

                # Iterate over all possible previous states to accumulate
                # the probabilities of all paths from the start state to
                # the current state.
                for prev in range(self.L):
                    prob += alphas[t][prev] \
                            * self.A[prev][curr] \
                            * self.O[curr][x[t]]

                # Store the accumulated probability.
                alphas[t + 1][curr] = prob

            if normalize:
                norm = sum(alphas[t + 1])
                for curr in range(self.L):
                    alphas[t + 1][curr] /= norm

        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Initialize initial betas.
        for curr in range(self.L):
            betas[-1][curr] = 1

        # Calculate betas throughout sequence.
        for t in range(-1, -M - 1, -1):
            # Iterate over all possible current states.
            for curr in range(self.L):
                prob = 0

                # Iterate over all possible next states to accumulate
                # the probabilities of all paths from the end state to
                # the current state.
                for nxt in range(self.L):
                    if t == -M:
                        prob += betas[t][nxt] \
                                * self.A_start[nxt] \
                                * self.O[nxt][x[t]]

                    else:
                        prob += betas[t][nxt] \
                                * self.A[curr][nxt] \
                                * self.O[nxt][x[t]]

                # Store the accumulated probability.
                betas[t - 1][curr] = prob

            if normalize:
                norm = sum(betas[t - 1])
                for curr in range(self.L):
                    betas[t - 1][curr] /= norm

        return betas

    def unsupervised_learning(self, X, iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        dataset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.
        '''

        # Note that a comment starting with 'E' refers to the fact that
        # the code under the comment is part of the E-step.

        # Similarly, a comment starting with 'M' refers to the fact that
        # the code under the comment is part of the M-step.

        for iteration in range(iters):
            print("Iteration: " + str(iteration))

            # Numerator and denominator for the update terms of A and O.
            A_num = [[0. for i in range(self.L)] for j in range(self.L)]
            O_num = [[0. for i in range(self.D)] for j in range(self.L)]
            A_den = [0. for i in range(self.L)]
            O_den = [0. for i in range(self.L)]

            # For each input sequence:
            for x in X:
                M = len(x)
                # Compute the alpha and beta probability vectors.
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)

                # E: Update the expected observation probabilities for a
                # given (x, y).
                # The i^th index is P(y^t = i, x).
                for t in range(1, M + 1):
                    P_curr = [0. for _ in range(self.L)]
                    
                    for curr in range(self.L):
                        P_curr[curr] = alphas[t][curr] * betas[t][curr]

                    # Normalize the probabilities.
                    norm = sum(P_curr)
                    for curr in range(len(P_curr)):
                        P_curr[curr] /= norm

                    for curr in range(self.L):
                        if t != M:
                            A_den[curr] += P_curr[curr]
                        O_den[curr] += P_curr[curr]
                        O_num[curr][x[t - 1]] += P_curr[curr]

                # E: Update the expectedP(y^j = a, y^j+1 = b, x) for given (x, y)
                for t in range(1, M):
                    P_curr_nxt = [[0. for _ in range(self.L)] for _ in range(self.L)]

                    for curr in range(self.L):
                        for nxt in range(self.L):
                            P_curr_nxt[curr][nxt] = alphas[t][curr] \
                                                    * self.A[curr][nxt] \
                                                    * self.O[nxt][x[t]] \
                                                    * betas[t + 1][nxt]

                    # Normalize:
                    norm = 0
                    for lst in P_curr_nxt:
                        norm += sum(lst)
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            P_curr_nxt[curr][nxt] /= norm

                    # Update A_num
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            A_num[curr][nxt] += P_curr_nxt[curr][nxt]

            for curr in range(self.L):
                for nxt in range(self.L):
                    self.A[curr][nxt] = A_num[curr][nxt] / A_den[curr]

            for curr in range(self.L):
                for xt in range(self.D):
                    self.O[curr][xt] = O_num[curr][xt] / O_den[curr]

    def preethi_generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a string.
        '''
        emission = []

        # Choose the starting state
        curr_state = int(np.random.uniform(0, self.L))
        for i in range(M):
            obs_random = np.random.uniform(0, 1)
            index = self.random_selector(obs_random, self.O[curr_state])
            emission.append(index)
            state_random = np.random.uniform(0, 1)
            index = self.random_selector(state_random, self.A[curr_state])
            curr_state = index
        # My favorite sequence is the very last one, because it signals the end of this homework
        # set 
        return emission

    def generate_line_syl(self, line, num_syl, observation_word_map):
        '''
        Generates a line of a poem with a given number of syllables, assuming 
        that the starting state is chosen uniformly at random. The line will start 
        with a word from the rhyming dictionary.

        Arguments:
            line:       Contains the rhyme word to be in the line

            num_syl:    The number of syllables to be in the line

            observation_word_map: the dictionary containing observation:corresponding
                                  word as the key and value
            

        Returns:
            emission:   The randomly generated line as a string.
        '''

        # Choose the starting state
        curr_state = int(np.random.uniform(0, self.L))
        # the current number of syllables is the number of syllables of the 
        # first word. This is calculated using the sylco() function
        num_syls = sylco(line) 
        # Continue looping until line with exactly 10 syllables is generated.
        while True:
            obs_random = np.random.uniform(0, 1)
            index = self.random_selector(obs_random, self.O[curr_state])
            chosen_word = observation_word_map[index] 
            # update num_syls with the number of syllables of the chosen word
            num_syls = num_syls + sylco(chosen_word)
            # if the number of syllables is above 10, dont add the word and sub
            # syllable count from num_syls. Try to resample a word using current 
            # hidden state.
            if num_syls > 10:
                num_syls -= sylco(chosen_word)
                continue
            # else add the word to the line
            else:
                line = line + chosen_word + " "
                #if the number of syllables is 10, finish and retun line
                if num_syls == 10:
                    return line
            # Sample new state given current state.
            state_random = np.random.uniform(0, 1)
            index = self.random_selector(state_random, self.A[curr_state])
            curr_state = index
        return line
    
    def random_selector(self, number, probs):
        total_probs = 0.0
        for prob_index, prob in enumerate(probs):
            total_probs += prob
            if (number <= total_probs):
                return prob_index

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        '''
        # Calculate each element of A using the M-step formulas.
        
        self.A_start = [0 for i in range(self.L)]
        self.A = [[0. for j in range(self.L)] for i in range(self.L)]
        self.O = [[0. for j in range(self.D)] for i in range(self.L)] # COLUMNS VS. ROWS?

        # Fill up transition matrix
        prev_state = 9000 # bc why not
        state_count = [0.0 for i in range(self.L)]
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if (j == 0):
                    first = Y[i][j]
                    self.A_start[first] += 1.0
                    prev_state = first
                else:
                    curr_state = Y[i][j]
                    self.A[prev_state][curr_state] += 1.0
                    prev_state = curr_state
                state_count[Y[i][j]] += 1.0
                
        # Sum up stuffs in the array
        A_np = np.array(self.A)
        A_np = A_np/A_np.sum(axis=1, keepdims=True)
        self.A = A_np.tolist()
             
        # Fill up observation matrix
        prev_state = 9000 # bc why not
        for i in range(len(X)):
            for j in range(len(X[i])):
                col = X[i][j]
                row = Y[i][j]
                self.O[row][col] += 1
        O_np = np.array(self.O)
        O_np = O_np/O_np.sum(axis=1, keepdims=True)
        self.O = O_np.tolist()

    def supervised_learning_counts(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). We then return the counts associated with the transition 
        and observation matrices as lists; these counts are used when calculating 
        the numerators in the M-step formulas. This method is a helper method 
        used for semi-supervised learning.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.

        Returns:
            the counts associated with the transition and observation matrices 
            as lists
        '''
        # Calculate each element of A using the M-step formulas.
        
        self.A_start = [0 for i in range(self.L)]
        self.A = [[0. for j in range(self.L)] for i in range(self.L)]
        self.O = [[0. for j in range(self.D)] for i in range(self.L)] # COLUMNS VS. ROWS?

        # Fill up transition matrix
        prev_state = 9000 # bc why not
        state_count = [0.0 for i in range(self.L)]
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if (j == 0):
                    first = Y[i][j]
                    self.A_start[first] += 1.0
                    prev_state = first
                else:
                    curr_state = Y[i][j]
                    self.A[prev_state][curr_state] += 1.0
                    prev_state = curr_state
                state_count[Y[i][j]] += 1.0
                
        # Sum up stuffs in the array
        A_np = np.array(self.A)
        self.A = (A_np/A_np.sum(axis=1, keepdims=True)).tolist()
             
        # Fill up observation matrix
        prev_state = 9000 # bc why not
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                col = X[i][j]
                row = Y[i][j]
                self.O[row][col] += 1
        O_np = np.array(self.O)
        self.O = (O_np/O_np.sum(axis=1, keepdims=True)).tolist()
        return A_np.tolist(), O_np.tolist()

    def semisupervised_learning(self, X, n_states, iters, n_labeled, A_num_counts, O_num_counts):
        '''
        Trains the HMM using semi-supervised learning by using the Baum-Welch algorithm 
        on the partially labeled dataset X. The Baum-Welch algorithm uses counts 
        for the labeled part of the dataset and probabilities for the 
        unlabeled part of the dataset. Note that this method does not return 
        anything, but instead updates the attributes of the HMM object. Also, note that
        this method does not need the labels corresponding to the labeled portion of X
        as we pass in the matrices of labeled counts used for supervised learning with 
        the labeled portion of X.

        Arguments:
            X:            A dataset consisting of input sequences in the form
                          of lists of length M, consisting of integers ranging
                          from 0 to D - 1. In other words, a list of lists.

            n_states:     number of hidden states of the HMM

            iters:        number of transitions to run Baum-Welch algorithm

            n_labeled:    number of labeled sequences in X

            A_num_counts: matrix of labeled counts associated with the transition matrix.
                          Matrix is used when calculating the numerators in the M-step 
                          formulas for supervised learning applied to the labeled portion
                          of X.

            O_num_counts: matrix of labeled counts associated with the observation matrix.
                          Matrix is used when calculating the numerators in the M-step 
                          formulas for supervised learning applied to the labeled portion
                          of X.
        '''

        # Note that a comment starting with 'E' refers to the fact that
        # the code under the comment is part of the E-step.

        # Similarly, a comment starting with 'M' refers to the fact that
        # the code under the comment is part of the M-step.

        # Calculate terms in denominators of M-step when performing supervised learning
        # on the labeled part of X
        A_denom_counts = [sum(A_num_counts[i]) for i in range(len(A_num_counts))]
        O_denom_counts = [sum(O_num_counts[i]) for i in range(len(O_num_counts))]
        L_sup = len(A_num_counts)
        D_sup = len(O_num_counts[0])

        for iteration in range(iters):
            print("Iteration: " + str(iteration))

            # Numerator and denominator for the update terms of A and O.
            A_num = [[0. for i in range(self.L)] for j in range(self.L)]
            O_num = [[0. for i in range(self.D)] for j in range(self.L)]
            A_den = [0. for i in range(self.L)]
            O_den = [0. for i in range(self.L)]

            # Add up the counts for the terms corresponding to the labeled part of 
            # X to the numerators and denominators of the M-step update. 
            for curr in range(L_sup):
                A_den[curr] += A_denom_counts[curr]
                for nxt in range(L_sup):
                    A_num[curr][nxt] += A_num_counts[curr][nxt]
                O_den[curr] += O_denom_counts[curr]
                for xt in range(D_sup):
                    O_num[curr][xt] += O_num_counts[curr][xt]

            # Add probabilities to the numerators and denominators of the M-step update.
            # These probabilities correspond to terms that do not correspond to the 
            # labeled part of the data.
            # For each input sequence that was not labeled:
            for x in X[n_labeled:]: 
                M = len(x)
                # Compute the alpha and beta probability vectors.
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)

                # E: Update the expected observation probabilities for a
                # given (x, y).
                # The i^th index is P(y^t = i, x).
                for t in range(1, M + 1):
                    P_curr = [0. for _ in range(self.L)]
                    
                    for curr in range(self.L):
                        P_curr[curr] = alphas[t][curr] * betas[t][curr]

                    # Normalize the probabilities.
                    norm = sum(P_curr)
                    for curr in range(len(P_curr)):
                        P_curr[curr] /= norm

                    for curr in range(self.L):
                        if t != M:
                            A_den[curr] += P_curr[curr]
                        O_den[curr] += P_curr[curr]
                        O_num[curr][x[t - 1]] += P_curr[curr]

                # E: Update the expectedP(y^j = a, y^j+1 = b, x) for given (x, y)
                for t in range(1, M):
                    P_curr_nxt = [[0. for _ in range(self.L)] for _ in range(self.L)]

                    for curr in range(self.L):
                        for nxt in range(self.L):
                            P_curr_nxt[curr][nxt] = alphas[t][curr] \
                                                    * self.A[curr][nxt] \
                                                    * self.O[nxt][x[t]] \
                                                    * betas[t + 1][nxt]

                    # Normalize:
                    norm = 0
                    for lst in P_curr_nxt:
                        norm += sum(lst)
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            P_curr_nxt[curr][nxt] /= norm

                    # Update A_num
                    for curr in range(self.L):
                        for nxt in range(self.L):
                            A_num[curr][nxt] += P_curr_nxt[curr][nxt]

            for curr in range(self.L):
                for nxt in range(self.L):
                    self.A[curr][nxt] = A_num[curr][nxt] / A_den[curr]

            for curr in range(self.L):
                for xt in range(self.D):
                    self.O[curr][xt] = O_num[curr][xt] / O_den[curr]

def unsupervised_HMM(X, n_states, n_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, n_iters)

    return HMM

def semisupervised_HMM(X, Y, n_states, n_iters):
    '''
    Helper function to train an semi-supervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then trains it 
    using semi-supervised learning. For semi-supervised learning, the function 
    first trains a HMM with the labeled part of the data using supervised learning to 
    obtain counts useful for semi-supervised learning and then runs the function for 
    semi-supervised training.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.
        
        Y:          Labels of hidden states for the labeled part of the dataset in the 
                    form of lists of variable length. In other words, a list of lists.
                    These labels correspond to the first several sequences in X.

        n_states:   Number of hidden states to use in training. Should be at least as many
                    hidden states as the number of hidden states specified in Y.
        
        n_iters:    Number of iterations to train the HMM for
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of observations.
    observations_sup = set()
    for x in X[0:len(Y)]:
        observations_sup |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Compute L_sup and D_sup, the number of hidden states and observations corresponding
    # to the labeled part of the dataset. Note that hidden states are 0-indexed.
    L_sup = 0
    for i in range(len(Y)):
        for j in range(len(Y[i])):
            L_sup = max(L_sup, Y[i][j])
    L_sup += 1
    D_sup = len(observations_sup)

    # Initialize and normalize matrices A_sup and O_sup.
    A_sup = [[0. for i in range(L_sup)] for j in range(L_sup)]
    
    O_sup = [[0. for i in range(D_sup)] for j in range(L_sup)]
        
    # Create a HMM and perform supervised learning using the labeled part of X
    # (supervised_learning_counts takes in all of X and the labels Y corresponding to 
    # a part of X). Obtain the counts necessary to apply Baum-Welch for semi-supervised 
    # learning.
    HMM_sup = HiddenMarkovModel(A_sup, O_sup)
    A_num_counts, O_num_counts = HMM_sup.supervised_learning_counts(X, Y)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with partially labeled data using the counts obtained before by training
    # on the labeled part of the data.
    HMM = HiddenMarkovModel(A, O)
    HMM.semisupervised_learning(X, n_states, n_iters, len(Y), A_num_counts, O_num_counts)

    return HMM
    
    
