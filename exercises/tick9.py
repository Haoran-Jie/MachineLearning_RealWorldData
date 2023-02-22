import numpy as np

from utils.markov_models import load_bio_data
import os
import random
import math
from exercises.tick8 import recall_score, precision_score, f1_score

from typing import List, Dict, Tuple


def get_transition_probs_bio(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden feature types using maximum likelihood estimation.

    @param hidden_sequences: A list of feature sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """
    res = dict()
    tmp = dict()
    hidden_states = set()
    for sample in hidden_sequences:
        now = sample[0]
        hidden_states.add(now)
        for timestep in sample[1:]:
            hidden_states.add(timestep)
            if now in tmp:
                tmp[now] += 1
            else:
                tmp[now] = 1
            future = timestep
            if (now, future) in res:
                res[(now, future)] += 1
            else:
                res[(now, future)] = 1
            now = timestep
    for key in res:
        res[key] /= tmp[key[0]]
    hidden_states = list(hidden_states)
    for i in range(len(hidden_states)):
        for j in range(len(hidden_states)):
            if (hidden_states[i], hidden_states[j]) not in res:
                res[(hidden_states[i], hidden_states[j])] = 0

    return res


def get_emission_probs_bio(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden feature states to visible amino acids, using maximum likelihood estimation.
    @param hidden_sequences: A list of feature sequences
    @param observed_sequences: A list of amino acids sequences
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


def estimate_hmm_bio(training_data:List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities.

    @param training_data: The biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    transition_probs = get_transition_probs_bio(hidden_sequences)
    emission_probs = get_emission_probs_bio(hidden_sequences, observed_sequences)
    return [transition_probs, emission_probs]


def viterbi_bio(observed_sequence, transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model.

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """
    observed_sequence = ['B'] + observed_sequence + ['Z']
    mostprob_prev_hidden = []
    path_probs = []
    hidden_states = list(set([key[0] for key in emission_probs]))
    total_time = len(observed_sequence)
    for i in range(total_time):
        nowprobs = dict()
        if i ==0:
            for j in hidden_states:
                if emission_probs[(j,observed_sequence[i])]==0:
                    nowprobs[j]=-math.inf
                else:
                    nowprobs[j]=math.log(emission_probs[(j,observed_sequence[i])])
        else:
            now_prev_hidden = dict()
            for j in hidden_states:
                maxprob = -math.inf
                for k in hidden_states:
                    if transition_probs[(k,j)]==0 or emission_probs[(j,observed_sequence[i])]==0 or path_probs[i-1][k]<-100000:
                        continue
                    origmaxprob = maxprob
                    maxprob = max(maxprob,path_probs[i-1][k]+math.log(transition_probs[(k,j)]*emission_probs[(j,observed_sequence[i])]))
                    if maxprob>origmaxprob:
                        now_prev_hidden[j]=k
                nowprobs[j]=maxprob
            mostprob_prev_hidden.append(now_prev_hidden)
        path_probs.append(nowprobs)
    final_prob = -math.inf
    final_state = ''
    for key in path_probs[-1]:
        if path_probs[-1][key]>final_prob:
            final_state=key
            final_prob=path_probs[-1][key]
    rev_sequence = [final_state]
    now_state = final_state
    for i in mostprob_prev_hidden[::-1]:
        now_state = i[now_state]
        rev_sequence.append(now_state)

    return rev_sequence[::-1][1:-1]




def self_training_hmm(training_data: List[Dict[str, List[str]]], dev_data: List[Dict[str, List[str]]],
    unlabeled_data: List[List[str]], num_iterations: int) -> List[Dict[str, float]]:
    """
    The self-learning algorithm for your HMM for a given number of iterations, using a training, development, and unlabeled dataset (no cross-validation to be used here since only very limited computational resources are available.)

    @param training_data: The training set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param dev_data: The development set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param unlabeled_data: Unlabeled sequence data of amino acids, encoded as a list of sequences.
    @num_iterations: The number of iterations of the self_training algorithm, with the initial HMM being the first iteration.
    @return: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    """
    dev_observed_sequences = [x['observed'] for x in dev_data]
    dev_hidden_sequences = [x['hidden'] for x in dev_data]
    res = []
    now_training_data = training_data
    for i in range(num_iterations):
        transition_probs, emission_probs = estimate_hmm_bio(now_training_data)
        predictions = []
        for sample in dev_observed_sequences:
            prediction = viterbi_bio(sample, transition_probs, emission_probs)
            predictions.append(prediction)
        predictions_binarized = [[1 if x == 'M' else 0 for x in pred] for pred in predictions]
        dev_hidden_sequences_binarized = [[1 if x == 'M' else 0 for x in dev] for dev in dev_hidden_sequences]

        p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
        r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
        f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

        nowscore = dict()
        nowscore['recall']=r
        nowscore['precision']=p
        nowscore['f1']=f1
        res.append(nowscore)

        tmp_pseudolabelled_data = []
        for sample in unlabeled_data:
            sample_labelled_dict = dict()
            prediction = viterbi_bio(sample,transition_probs,emission_probs)
            sample_labelled_dict['observed']=sample
            sample_labelled_dict['hidden']=prediction
            tmp_pseudolabelled_data.append(sample_labelled_dict)

        now_training_data = training_data + tmp_pseudolabelled_data

    return res





def visualize_scores(score_list:List[Dict[str,float]]) -> None:
    """
    Visualize scores of the self-learning algorithm by plotting iteration vs scores.

    @param score_list: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    @return: The most likely single sequence of hidden states
    """
    from utils.sentiment_detection import clean_plot, chart_plot
    import numpy as np

    iter_num = len(score_list)
    x = 1+np.arange(iter_num)
    recalls = [item['recall'] for item in score_list]
    precisions = [item['precision'] for item in score_list]
    f1s = [item['f1'] for item in score_list]
    data = [(x_coor,y_coor) for x_coor,y_coor in zip(x,recalls)]
    chart_plot(data,title="iteration vs. recall",x_label='iterations',y_label='recall')
    data = [(x_coor, y_coor) for x_coor, y_coor in zip(x, precisions)]
    chart_plot(data, title="iteration vs. precision", x_label='iterations', y_label='precision')
    data = [(x_coor, y_coor) for x_coor, y_coor in zip(x, f1s)]
    chart_plot(data, title="iteration vs. f1", x_label='iterations', y_label='f1')


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    bio_data = load_bio_data(os.path.join('data', 'markov_models', 'bio_dataset.txt'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    bio_data_shuffled = random.sample(bio_data, len(bio_data))
    dev_size = int(len(bio_data_shuffled) / 10)
    train = bio_data_shuffled[dev_size:]
    dev = bio_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm_bio(train)

    for sample in dev_observed_sequences:
        prediction = viterbi_bio(sample, transition_probs, emission_probs)
        predictions.append(prediction)
    predictions_binarized = [[1 if x=='M' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x=='M' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    unlabeled_data = []
    with open(os.path.join('data', 'markov_models', 'bio_dataset_unlabeled.txt'), encoding='utf-8') as f:
        content = f.readlines()
        for i in range(0, len(content), 2):
            unlabeled_data.append(list(content[i].strip())[1:])

    scores_each_iteration = self_training_hmm(train, dev, unlabeled_data, 5)

    visualize_scores(scores_each_iteration)



if __name__ == '__main__':
    main()
