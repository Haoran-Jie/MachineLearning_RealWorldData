from utils.markov_models import load_dice_data, print_matrices
import os
from typing import List, Dict, Tuple


def get_transition_probs(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden dice types using maximum likelihood estimation. Counts the number of times each state sequence appears and divides it by the count of all transitions going from that state. The table must include proability values for all state-state pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """
    res = dict()
    tmp = dict() # count the number of occurence of a state
    hidden_states = set()
    for sample in hidden_sequences:
        now = sample[0]
        hidden_states.add(now)
        for timestep in sample[1:]:
            hidden_states.add(timestep)
            if now in tmp:
                tmp[now]+=1
            else:
                tmp[now]=1
            future = timestep
            if (now,future) in res:
                res[(now,future)]+=1
            else:
                res[(now,future)]=1
            now = timestep
    for key in res:
        res[key]/=tmp[key[0]]
    hidden_states = list(hidden_states)
    for i in range(len(hidden_states)):
        for j in range(len(hidden_states)):
            if (hidden_states[i],hidden_states[j]) not in res:
                res[(hidden_states[i],hidden_states[j])]=0

    return res



def get_emission_probs(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden dice states to observed dice rolls, using maximum likelihood estimation. Counts the number of times each dice roll appears for the given state (fair or loaded) and divides it by the count of that state. The table must include proability values for all state-observation pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
    @param observed_sequences: A list of dice roll sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """
    res = dict()
    tmp = dict()
    hidden_states = set()
    observed_states = set()
    for i in range(len(hidden_sequences)):
        for hidden,observed in zip(hidden_sequences[i],observed_sequences[i]):
            hidden_states.add(hidden)
            observed_states.add(observed)
            if (hidden,observed) in res:
                res[(hidden,observed)] += 1
            else:
                res[(hidden,observed)] = 1
            if hidden in tmp:
                tmp[hidden]+=1
            else:
                tmp[hidden]=1
    for key in res:
        res[key]/=tmp[key[0]]
    hidden_states = list(hidden_states)
    observed_states = list(observed_states)
    for i in hidden_states:
        for j in observed_states:
            if (i,j) not in res:
                res[(i,j)]=0
    return res



def estimate_hmm(training_data: List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities. We use 'B' for the start state and 'Z' for the end state, both for emissions and transitions.

    @param training_data: The dice roll sequence data (visible dice rolls and hidden dice types), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs(hidden_sequences)
    emission_probs = get_emission_probs(hidden_sequences, observed_sequences)
    return [transition_probs, emission_probs]


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))
    transition_probs, emission_probs = estimate_hmm(dice_data)
    print(f"The transition probabilities of the HMM:")
    print_matrices(transition_probs)
    print(f"The emission probabilities of the HMM:")
    print_matrices(emission_probs)

if __name__ == '__main__':
    main()