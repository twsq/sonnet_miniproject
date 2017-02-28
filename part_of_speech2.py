'''
Created on Feb 22, 2017

@author: PreethiKP
'''

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

def preprocess(file):
    '''
    This function processes the file of Shakespeare sonnets by taking each line
    to be a sequence. The lines will be tokenized by the words in them. In
    addition, this function is responsible to creating the rhyming dictionary to
    be used when emitting the lines according to the sonnet rhyme structure. 
    
    Input: name of the file of sonnets
    
    Output: tokenized_lines (a list of lists), rhyme_dict (rhyming dictionary)
    '''
    tokenized_lines = []
    rhyme_dict = {}
    # Retrieve the lines from the file
    with open(file) as f:
        lines = f.readlines()
    for i in range(len(lines)):
        words = lines[i].split()
        sonnet = []
        # 1 word indicates the # of the sonnet, followed by the 14 line sonnet
        if len(words) == 1:
            b = i + 1
            # Loop through the lines in the sonnet
            for index in range(b, b+14):
                words = lines[index].split()
                # Strip the words of punctuation and appeend them to tokenized_lines
                for k in range(len(words)):
                    words[k] = words[k].strip(string.punctuation)
                tokenized_lines.append(words)
                sonnet.append(words)
            # Build the rhyme dictionary. Rhyme pairs are decided by form 
            # ABAB CDCD EFEF GG
            rhyme_dict[sonnet[0][-1]] = sonnet[2][-1]
            rhyme_dict[sonnet[1][-1]] = sonnet[3][-1]
            rhyme_dict[sonnet[2][-1]] = sonnet[0][-1]
            rhyme_dict[sonnet[3][-1]] = sonnet[1][-1]
            rhyme_dict[sonnet[4][-1]] = sonnet[6][-1]
            rhyme_dict[sonnet[5][-1]] = sonnet[7][-1]
            rhyme_dict[sonnet[6][-1]] = sonnet[4][-1]
            rhyme_dict[sonnet[7][-1]] = sonnet[5][-1]
            rhyme_dict[sonnet[8][-1]] = sonnet[10][-1]
            rhyme_dict[sonnet[9][-1]] = sonnet[11][-1]
            rhyme_dict[sonnet[10][-1]] = sonnet[8][-1]
            rhyme_dict[sonnet[11][-1]] = sonnet[9][-1]
            rhyme_dict[sonnet[12][-1]] = sonnet[13][-1]
            rhyme_dict[sonnet[13][-1]] = sonnet[12][-1]
    return tokenized_lines, rhyme_dict    

def tag_POS(line):
    """ Given a list of words, find the part of speech of each word """
    return nltk.pos_tag(line) # Will return a set of tuples (word, POS)

def convert_lines_observations(tokenized_lines):
    '''
    Given a list of lines with words tokenized, assign each unique word an integer 
    observation. Return a list of sequences of observations (i.e. a list of lists) 
    and the maps mapping observations to words and words to observations.

    Note that capital and lowercase words are treated as different words.
    '''
    num_unique_words = 0
    word_observations_map = {}
    observations_word_map = {}
    observations = []
    # For each tokenized line...
    for line in tokenized_lines:
        sequence = []
        # For each word 
        for word in line:
            # Add each unique word to the word observations map
            # Have a second map that returns the word given the observation
            if word not in word_observations_map.keys():
                word_observations_map[word] = num_unique_words
                observations_word_map[num_unique_words] = word
                num_unique_words += 1
            sequence.append(word_observations_map[word])
        observations.append(sequence)
    return observations, observations_word_map, word_observations_map

def convert_POS_to_states(tokenized_lines):
    '''
    Given a list of lines with words tokenized, assign each unique part of speech 
    an integer state. Return a list of sequences of states (i.e. a list of lists) 
    and the maps mapping part of speech to states and states to part of speech.
    '''
    num_of_POS = 0
    POS_state_map = {}
    states_POS_map = {}
    states = []
    # For each tokenized line...
    for line in tokenized_lines:
        # Find the part of speech of each word in the line
        tagged_tokens = tag_POS(line)
        sequence = []
        # For each word and its corresponding part of speech
        for word, POS in tagged_tokens:
            # Add each unique POS to the POS states map
            # Have a second map that returns the POS given the state number
            if POS not in POS_state_map.keys(): 
                POS_state_map[POS] = num_of_POS
                states_POS_map[num_of_POS] = POS
                num_of_POS += 1
            sequence.append(POS_state_map[POS])
        states.append(sequence)
    return states, states_POS_map, POS_state_map
        
def modify_line(line):
    '''
    This function modifies the line by taking the first word (the rhyme word), 
    and moving it to the end of the sentence. 
    
    Input: line
    
    Output: modified line with rhyme word at end
    '''
    words = line.split()
    out_line = ''
    for i in range(1, len(words)):
        out_line = out_line + words[i] + ' '
    out_line = out_line + words[0]
    return out_line

def supervised_learning(tokenized_lines):
    '''
    Generate a sonnet by training a HMM using supervised learning and then using 
    the HMM to generate a line with 10 words. The dataset is labeled with part of 
    speech tags as states for supervised learning.

    Arguments:
        tokenized_lines: a list of lines tokenized as words
    '''

    # Come up with all of the maps needed (states to part-of-speech and vice versa)
    # These maps will be used for the training portion of the supervised model and for 
    # generating the poem.
    states, state_POS_map, POS_state_map = convert_POS_to_states(tokenized_lines)
    observations, observation_word_map, word_observation_map = convert_lines_observations(tokenized_lines)
    
    # Initialize transition and observation matrices.
    A = [[0. for j in range(len(state_POS_map))] for i in range(len(state_POS_map))]
    O = [[0. for j in range(len(observation_word_map))] for i in range(len(state_POS_map))]

    # Create HMM that will be trained. X is a list of lines tokenized as words, and 
    # Y is the corresponding part of speech tag labels for every word.
    hmm = HiddenMarkovModel(A, O)
    X = [[]]
    Y = [[]]

    # For each tokenized line that we are using for the training data, find the part of
    # speech of each word and add the corresponding states to Y. Also, fill in 
    # the X with words from the lines.
    for line in tokenized_lines:
        words_and_tags = tag_POS(line)
        x = []
        y = []
        for word, POS in words_and_tags:
            x.append(word_observation_map[word])
            y.append(POS_state_map[POS])
        X.append(x)
        Y.append(y)

    # Train HMM using supervised learning with X and Y, where Y contains the part of speech
    # labels.
    hmm.supervised_learning(X, Y)

    # Generate 14 lines with 10 words each and print them out.
    for i in range(14):
        obs = hmm.preethi_generate_emission(10)
        line = ''
        for j in obs:
            line += observation_word_map[j]
            line += " "
        print(line)

def unsupervised_learning(tokenized_lines):
    '''
    Generate a sonnet by training a HMM using unsupervised learning and then using 
    the HMM to generate a line with 10 words. The number of hidden states used is the 
    same as the number of part of speech labels for the words in the sonnet dataset 
    (i.e. the same number of hidden states as supervised learning would use).

    Arguments:
        tokenized_lines: a list of lines tokenized as words
    '''

    # Come up with all of the maps needed (states to part-of-speech and vice versa)
    # These maps will be used for determining how many hidden states the HMM will use and
    # for generating the poem. The list of observation sequences will be used to train the 
    # unsupervised HMM.
    states, state_POS_map, POS_state_map = convert_POS_to_states(tokenized_lines)
    observations, observation_word_map, word_observation_map = convert_lines_observations(tokenized_lines)
    
    # Initialize transition and observation matrices.
    A = [[0. for j in range(len(state_POS_map))] for i in range(len(state_POS_map))]
    O = [[0. for j in range(len(observation_word_map))] for i in range(len(state_POS_map))]

    # Create HMM and train it with unsupervised learning.
    hmm = unsupervised_HMM(observations, len(state_POS_map), 50)

    # Generate 14 lines with 10 words each and print them out.
    for i in range(14):
        obs = hmm.preethi_generate_emission(10)
        line = ''
        for j in obs:
            line += observation_word_map[j]
            line += " "
        print(line)

def semisupervised_learning(tokenized_lines, rhyme_dict, prop_labeled, num_extra_states, n_iters):
    '''
    Generate a sonnet by training a HMM using semisupervised learning and then using the HMM's 
    generate_line_syl function to generate the lines of the poem with the correct
    number of syllables. This function also outputs the lines in the ABAB CDCD EFEF GG rhyme 
    structure using the rhyme dictionary. 

    Return the trained HMM, the maps of part of speech to state and state to part of speech, 
    the maps of observations to words and words to observations, and the poem. Everything 
    returned besides the poem is used for visualization and interpretation. 

    Arguments:
        tokenized_lines: a list of lines tokenized as words

        rhyme_dict: a rhyme dictionary used for enforcing rhyme

        prop_labeled: the proportion of data to label with part of speech hidden states

        num_extra_states: extra hidden states (hidden states not corresponding to 
            part of speech labels in the labeled part of the data)

        n_iters: number of EM iterations.

    Returns:
        hmm: trained HMM

        state_POS_map: map of state to part of speech

        POS_state_map: map of part of speech to state

        observation_word_map: map of observations to words
 
        word_observation_map: map of words to observations

        poem: generated sonnet in the form of a list of lines
    '''

    # Come up with all of the maps needed (states to part-of-speech and vice versa)
    # These maps will be used for the training portion of the semi-supervised model.
    states, state_POS_map, POS_state_map = convert_POS_to_states(tokenized_lines)
    observations, observation_word_map, word_observation_map = convert_lines_observations(tokenized_lines)
    poem = []
    Y = []
    num_states_labeled = 0
    # For each tokenized line that we are using for the training data, find the part of
    # speech of each word and add the corresponding states to Y
    for line in tokenized_lines[:int(len(tokenized_lines) * prop_labeled)]:
        words_and_tags = tag_POS(line)
        y = []
        for word, POS in words_and_tags:
            num_states_labeled = max(POS_state_map[POS], num_states_labeled)
            y.append(POS_state_map[POS])
        Y.append(y)
    num_states_labeled += 1
    # Use the training data states in the semi-supervised HMM for the specified number of
    # iterations. Add in the extra states (as wiggle-room for the HMM)
    hmm = semisupervised_HMM(observations, Y, num_states_labeled + num_extra_states, n_iters)
    # Generate 3 stanzas of four lines, each with 10 syllables. The stanzas have the correct 
    # rhyme scheme.
    for i in range(3):
        # first line: get first word in first rhyme pair
        first_word = list(rhyme_dict)[random.randint(0, len(list(rhyme_dict)))]
        # generate the line with rhyme word as input, and 10 syllables
        line = first_word + ' '
        line = hmm.generate_line_syl(line, 10, observation_word_map)
        poem.append(modify_line(line))
        # second line: get the first word in the second rhyme pair
        second_word = list(rhyme_dict)[random.randint(0, len(list(rhyme_dict)))]
        line = second_word + ' '
        line = hmm.generate_line_syl(line, 10, observation_word_map)
        poem.append(modify_line(line))
         # third line: get the second word in first rhyme pair
        first_rhyme = rhyme_dict[first_word]
        line = first_rhyme + ' '
        line = hmm.generate_line_syl(line, 10, observation_word_map)
        poem.append(modify_line(line))
        # fourth line: get the second word in second rhyme pair
        sec_rhyme = rhyme_dict[second_word]
        line = sec_rhyme + ' '
        line = hmm.generate_line_syl(line, 10, observation_word_map)
        poem.append(modify_line(line))
    # Generate last 2 lines of 10 syllables each. The two lines rhyme with each other.
    # get the first word in the last rhyme pair
    last_word = list(rhyme_dict)[random.randint(0, len(list(rhyme_dict)))]
    line = last_word + ' '
    line = hmm.generate_line_syl(line, 10, observation_word_map)
    poem.append(modify_line(line))
    # get the second word in the last rhyme pair
    last_rhyme = rhyme_dict[last_word]
    line = last_rhyme + ' '
    line = hmm.generate_line_syl(line, 10, observation_word_map)
    poem.append(modify_line(line))
    return hmm, state_POS_map, POS_state_map, observation_word_map, word_observation_map, poem

def make_haiku(tokenized_lines, prop_labeled, num_extra_states, n_iters):
    '''
    Generate a haiku by training a HMM using semisupervised learning.

    Arguments:
        tokenized_lines: a list of lines tokenized as words

        prop_labeled: the proportion of data to label with part of speech hidden states

        num_extra_states: extra hidden states (hidden states not corresponding to 
            part of speech labels)

        n_iters: number of EM iterations.

    Returns:
        poem: generated sonnet in the form of a list of lines
    '''
    # Come up with all of the maps needed (states to part-of-speech and vice versa)
    # These maps will be used for the training portion of the semi-supervised model.
    states, state_POS_map, POS_state_map = convert_POS_to_states(tokenized_lines)
    observations, observation_word_map, word_observation_map = convert_lines_observations(tokenized_lines)
    poem = []
    Y = []
    num_states_labeled = 0
    # For each tokenized line that we are using for the training data, find the part of
    # speech of each word and add the corresponding states to the Y matrix
    for line in tokenized_lines[:int(len(tokenized_lines) * prop_labeled)]:
        words_and_tags = tag_POS(line)
        y = []
        for word, POS in words_and_tags:
            num_states_labeled = max(POS_state_map[POS], num_states_labeled)
            y.append(POS_state_map[POS])
        Y.append(y)
    num_states_labeled += 1
    # Use the training data states in the semi-supervised HMM for the specified number of
    # iterations. Add in the extra states (as wiggle-room for the HMM)
    hmm = semisupervised_HMM(observations, Y, num_states_labeled + num_extra_states, n_iters)
    sylcount = [5, 7, 5] # An array specifying the syllable count for each line
    haiku = []
    for i in range(3): # Because a haiku is 3 lines long
        # Generate the first word in each line
        first_word = hmm.preethi_generate_emission(1)
        # Look up the word in the observation-to-word map
        line = observation_word_map[first_word[0]]
        # Generate a line based on the first word, with the necessary syllable count
        line = hmm.generate_line_syl(line, sylcount[i], observation_word_map)
        haiku.append(line) # Add it to the haiku
    return haiku

def make_limerick(tokenized_lines, rhyme_dict, prop_labeled, num_extra_states, n_iters):
    '''
    Generate a limerick by training a HMM using semisupervised learning.

    Arguments:
        tokenized_lines: a list of lines tokenized as words

        rhyme_dict: a rhyme dictionary used for enforcing rhyme

        prop_labeled: the proportion of data to label with part of speech hidden states

        num_extra_states: extra hidden states (hidden states not corresponding to 
            part of speech labels)

        n_iters: number of EM iterations.

    Returns:
        poem: generated sonnet in the form of a list of lines
    '''

    # Come up with all of the maps needed (states to part-of-speech and vice versa)
    # These maps will be used for the training portion of the semi-supervised model.
    states, state_POS_map, POS_state_map = convert_POS_to_states(tokenized_lines)
    observations, observation_word_map, word_observation_map = convert_lines_observations(tokenized_lines)
    Y = []
    num_states_labeled = 0
    # For each tokenized line that we are using for the training data, find the part of
    # speech of each word and add the corresponding states to the Y matrix
    for line in tokenized_lines[:int(len(tokenized_lines) * prop_labeled)]:
        words_and_tags = tag_POS(line)
        y = []
        for word, POS in words_and_tags:
            # Look up the corresponding states of the parts of speech for the words
            # in the training data set, append them to the Y matrix
            num_states_labeled = max(POS_state_map[POS], num_states_labeled)
            y.append(POS_state_map[POS])
        Y.append(y)
    num_states_labeled += 1
    # Use the training data states in the semi-supervised HMM for the specified number of
    # iterations. Add in the extra states (as wiggle-room for the HMM)
    hmm = semisupervised_HMM(observations, Y, num_states_labeled + num_extra_states, n_iters)
    poem = []
    # Use the rhyming code in the semisupervised_learning function to generate
    # the lines of the limerick.
    # Limericks have rhyme scheme AABBA.
    # First, generate the first 2 lines and the last line (the A's)
    first_word = list(rhyme_dict)[random.randint(0, len(list(rhyme_dict)))]
    line = first_word + ' '
    line = hmm.generate_line_syl(line, 8, observation_word_map)
    poem.append(modify_line(line))
    for i in range(2):
        first_rhyme = rhyme_dict[first_word]
        line = first_rhyme + ' '
        line = hmm.generate_line_syl(line, 8, observation_word_map)
        poem.append(modify_line(line))
    # Then, generate the middle two lines (the B's)
    second_word = list(rhyme_dict)[random.randint(0, len(list(rhyme_dict)))]
    line = second_word + ' '
    line = hmm.generate_line_syl(line, 5, observation_word_map)
    poem.append(modify_line(line))
    sec_rhyme = rhyme_dict[second_word]
    line = sec_rhyme + ' '
    line = hmm.generate_line_syl(line, 5, observation_word_map)
    poem.append(modify_line(line))
    
    # Need to move the third A to the end (otherwise it's AAABB,
    # and we need AABBA)
    third_line = poem.pop(2) # remove the third line and return it
    poem.append(third_line) # move the third line to the end
    return poem

def postprocess_print(poem):
    '''
    Capitalize the first letter of each line, add punctuation, and print the poem.
    '''
    # Capitalize the right letters and add punctuation
    count = 0
    fixed_poem = []
    for line in poem:
        line_list = line.split(" ")
        for i in range(len(line_list)): # Convert everything to lower case, except the pronoun "I"
            if (line_list[i] != "I"):
                line_list[i] = (line_list[i]).lower()
        line_list[0] = (line_list[0]).title() # Capitalize the first letter of each line
        new_line = ""
        for word in line_list:
            new_line += " "
            new_line += word
        new_line = new_line.lstrip(" ")
        new_line = new_line.rstrip(" ")
        if (count < len(poem) - 1): # Add a comma at the end of each line, except the last one
            new_line += ","
            count += 1
        else:
            new_line += "." # Put a period after the end of the last line
        print(new_line) # Remove any leading whitespaces and print

# Preprocess the Shakespeare text
# One of the Shakespearean sonnets was removed because it had 12 lines
# instead of 14, and it was causing indexing errors. 
tokenized_lines1, rhyme_dict1 = preprocess("shakespeare.txt")

# Train HMM with semi-supervised learning using Shakespeare text
hmm, state_POS_map, POS_state_map, observation_word_map, word_observation_map, poem = \
    semisupervised_learning(tokenized_lines1, rhyme_dict1, 0.2, 1, 50)
postprocess_print(poem)

# Save transition and observation matrices of HMM into text files. Save maps of state to
# part of speech, part of speech to state, observations to words, and words to observations 
# using pickle. We can then retrieve the matrices and maps for further analysis later.
A = np.array(hmm.A)
O = np.array(hmm.O)
np.savetxt("transition_matrix.txt", A)
np.savetxt("observation_matrix.txt", O)
pickle.dump(state_POS_map, open("state_POS_map.p", "wb"))
pickle.dump(POS_state_map, open("POS_state_map.p", "wb"))
pickle.dump(observation_word_map, open("observation_word_map.p", "wb"))
pickle.dump(word_observation_map, open("word_observation_map.p", "wb"))

# Preprocess the Spenser text
# The Spenser text was modified: one poem was removed because it had a 
# different rhyme scheme, and the epigraph at the beginning of Sonnet 58 was
# removed (because it then made the sonnet have 13 lines).
tokenized_lines2, rhyme_dict2 = preprocess("spenser.txt")

# Combine the tokenized lines into one list and random-shuffle them
# (because we are using 20% of the data for training, we want to make
# sure that some of the training data are also from Spenser's sonnets).
tokenized_lines = tokenized_lines1 + tokenized_lines2
random.shuffle(tokenized_lines)

# Update the rhyme dictionary to incorporate Spenser's work.
rhyme_dict = rhyme_dict1.copy()
rhyme_dict.update(rhyme_dict2)

# Generate haiku and limerick
poem = make_haiku(tokenized_lines, 0.2, 3, 10)
postprocess_print(poem)

poem = make_limerick(tokenized_lines, rhyme_dict, 0.2, 3, 10)
postprocess_print(poem)


