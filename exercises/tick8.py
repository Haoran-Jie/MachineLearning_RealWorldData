from utils.markov_models import load_dice_data
import os
from exercises.tick7 import estimate_hmm
import random
import math

from typing import List, Dict, Tuple


def viterbi(observed_sequence: List[str], transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model. Use the same symbols for the start and end observations as in tick 7 ('B' for the start observation and 'Z' for the end observation).

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





def precision_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the precision of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of predicted weighted states that were actually weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The precision of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    TP = 0
    denom = 0

    for pre,tru in zip(pred,true):
        for i,j in zip(pre,tru):
            if i==j==1:
                TP+=1
                denom+=1
            elif i==1:
                denom+=1
    return TP/denom


def recall_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the recall of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of actual weighted states that were predicted weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The recall of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    TP = 0
    denom = 0
    for pre,tru in zip(pred,true):
        for i,j in zip(pre,tru):
            if i==j==1:
                TP+=1
                denom+=1
            elif j==1:
                denom+=1
    return TP/denom


def f1_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the F1 measure of the estimated sequence with respect to the positive class (weighted state), i.e. the harmonic mean of precision and recall.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The F1 measure of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    prec = precision_score(pred,true)
    rec = recall_score(pred,true)
    return 2* (prec*rec)/(prec+rec)


def cross_validation_sequence_labeling(data: List[Dict[str, List[str]]]) -> Dict[str, float]:
    """
    Run 10-fold cross-validation for evaluating the HMM's prediction with Viterbi decoding. Calculate precision, recall, and F1 for each fold and return the average over the folds.

    @param data: the sequence data encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'
    @return: a dictionary with keys 'recall', 'precision', and 'f1' and its associated averaged score.
    """

    def divide_list(lst, n):
        # Calculate the length of each part
        length = len(lst) // n
        # Calculate the remainder
        remainder = len(lst) % n
        # Initialize the start index
        start = 0
        # Divide the list into n parts
        result = []
        for i in range(n):
            # Calculate the end index
            end = start + length + (i < remainder)
            # Add the part to the result list
            result.append(lst[start:end])
            # Update the start index
            start = end
        return result

    total_length = len(data)
    batch = total_length/10
    ps = []
    rs = []
    f1s = []
    groups = divide_list(data,10)
    for i in range(10):
        train=[]
        for j in range(10):
            if j!=i:
                train+=groups[j]
        dev = groups[i];
        dev_observed_sequences = [x['observed'] for x in dev]
        dev_hidden_sequences = [x['hidden'] for x in dev]
        predictions = []
        transition_probs, emission_probs = estimate_hmm(train)

        for sample in dev_observed_sequences:
            prediction = viterbi(sample, transition_probs, emission_probs)
            predictions.append(prediction)

        predictions_binarized = [[1 if x == 'W' else 0 for x in pred] for pred in predictions]
        dev_hidden_sequences_binarized = [[1 if x == 'W' else 0 for x in dev] for dev in dev_hidden_sequences]

        p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
        r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
        f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)

    res = dict()
    res['recall'] = sum(rs)/len(rs)
    res['precision']= sum(ps)/len(ps)
    res['f1']=sum(f1s)/len(f1s)
    return res


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    dice_data_shuffled = random.sample(dice_data, len(dice_data))
    dev_size = int(len(dice_data) / 10)
    train = dice_data_shuffled[dev_size:]
    dev = dice_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm(train)

    for sample in dev_observed_sequences:
        prediction = viterbi(sample, transition_probs, emission_probs)
        predictions.append(prediction)

    predictions_binarized = [[1 if x == 'W' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x == 'W' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    print(f"Evaluating HMM using cross-validation with 10 folds.")

    cv_scores = cross_validation_sequence_labeling(dice_data)

    print(f" Your cv average precision using the HMM: {cv_scores['precision']}")
    print(f" Your cv average recall using the HMM: {cv_scores['recall']}")
    print(f" Your cv average F1 using the HMM: {cv_scores['f1']}")



if __name__ == '__main__':
    main()
