import os
from typing import List, Dict, Union

from utils.sentiment_detection import load_reviews, read_tokens, read_student_review_predictions, print_agreement_table

from exercises.tick5 import generate_random_cross_folds, cross_validation_accuracy

import math


def nuanced_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c) for nuanced sentiments.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to prior probability
    """
    res = dict()
    res[1]=0
    res[0]=0
    res[-1]=0
    total = len(training_data)
    for instance in training_data:
        res[instance['sentiment']]+=1
    res[1] = math.log(res[1]/total)
    res[0] = math.log(res[0]/total)
    res[-1] = math.log(res[-1]/total)
    return res


def nuanced_conditional_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a nuanced sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    res = dict()
    res[1] = dict()
    res[0] = dict()
    res[-1] = dict()
    wordset = set()
    possum = negsum = neusum = 0
    for instance in training_data:
        if instance['sentiment']==1:
            possum+=len(instance['text'])
            for token in instance['text']:
                wordset.add(token)
                if token in res[1]:
                    res[1][token]+=1
                else:
                    res[1][token]=1
        elif instance['sentiment']==0:
            neusum+=len(instance['text'])
            for token in instance['text']:
                wordset.add(token)
                if token in res[0]:
                    res[0][token]+=1
                else:
                    res[0][token]=1
        else:
            negsum+=len(instance['text'])
            for token in instance['text']:
                wordset.add(token)
                if token in res[-1]:
                    res[-1][token]+=1
                else:
                    res[-1][token]=1
    for word in wordset:
        if word not in res[1]:
            res[1][word]=0
        if word not in res[0]:
            res[0][word]=0
        if word not in res[-1]:
            res[-1][word]=0
    for i,nowsum in zip([-1,0,1],[negsum,neusum,possum]):
        for key in res[i]:
            res[i][key]=math.log((res[i][key]+1)/(nowsum+len(wordset)))
    return res



def nuanced_accuracy(pred: List[int], true: List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    correctnum = 0
    for pre,tru in zip(pred,true):
        if pre==tru:
            correctnum+=1
    return correctnum/len(pred)


def predict_nuanced_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                                  class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior probability
    @return: predicted sentiment [-1, 0, 1] for the given review
    """
    negprob = class_log_probabilities[-1]
    posprob = class_log_probabilities[1]
    neuprob = class_log_probabilities[0]
    for word in review:
        if word in log_probabilities[1]:
            posprob+=log_probabilities[1][word]
        if word in log_probabilities[0]:
            neuprob += log_probabilities[0][word]
        if word in log_probabilities[-1]:
            negprob += log_probabilities[-1][word]
    if posprob> negprob and posprob> neuprob:
        return 1
    elif neuprob>posprob and neuprob>negprob:
        return 0
    else:
        return -1



def calculate_kappa(agreement_table: Dict[int, Dict[int,int]]) -> float:
    """
    Using your agreement table, calculate the kappa value for how much agreement there was; 1 should mean total agreement and -1 should mean total disagreement.

    @param agreement_table:  For each review (1, 2, 3, 4) the number of predictions that predicted each sentiment
    @return: The kappa value, between -1 and 1
    """
    width = 0
    for i in agreement_table:
        width = len(agreement_table[i])
    if width==3:
        S_i = []
        C_i = dict()
        C_i[1]=0
        C_i[0]=0
        C_i[-1]=0
        k = 0
        for i in agreement_table:
            k = agreement_table[i][0]+agreement_table[i][1]+agreement_table[i][-1]
            break
        N = len(agreement_table)
        n = 3
        for i in agreement_table:
            ans = 0
            for categories in agreement_table[i]:
                ans+=agreement_table[i][categories]*(agreement_table[i][categories]-1)/2
                C_i[categories]+=agreement_table[i][categories]
            ans = ans/(k*(k-1)/2)
            S_i.append(ans)
        P_a = sum(S_i)/len(S_i)
        P_e =0
        for i in [0,-1,1]:
            P_e+=math.pow(C_i[i]/(k*N),2)

        k = (P_a - P_e)/(1-P_e)
        return k
    else:
        S_i = []
        C_i = dict()
        C_i[1] = 0
        C_i[-1] = 0
        k = 0
        for i in agreement_table:
            k = agreement_table[i][1] + agreement_table[i][-1]
            break
        N = len(agreement_table)
        n = 2
        for i in agreement_table:
            ans = 0
            for categories in agreement_table[i]:
                ans += agreement_table[i][categories] * (agreement_table[i][categories] - 1) / 2
                C_i[categories] += agreement_table[i][categories]
            ans = ans / (k * (k - 1) / 2)
            S_i.append(ans)
        P_a = sum(S_i) / len(S_i)
        P_e = 0
        for i in [-1, 1]:
            P_e += math.pow(C_i[i] / (k * N), 2)

        k = (P_a - P_e) / (1 - P_e)
        return k




def get_agreement_table(review_predictions: List[Dict[int, int]]) -> Dict[int, Dict[int,int]]:
    """
    Builds an agreement table from the student predictions.

    @param review_predictions: a list of predictions for each student, the predictions are encoded as dictionaries, with the key being the review id and the value the predicted sentiment
    @return: an agreement table, which for each review contains the number of predictions that predicted each sentiment.
    """
    res = dict()
    for i in range(4):
        res[i]=dict()
        for j in [-1,0,1]:
            res[i][j]=0
    for student in review_predictions:
        for review_id in student:
            res[review_id][student[review_id]]+=1
    return res



def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_nuanced'), include_nuance=True)
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    split_training_data = generate_random_cross_folds(tokenized_data, n=10)

    n = len(split_training_data)
    accuracies = []
    for i in range(n):
        test = split_training_data[i]
        train_unflattened = split_training_data[:i] + split_training_data[i+1:]
        train = [item for sublist in train_unflattened for item in sublist]

        dev_tokens = [x['text'] for x in test]
        dev_sentiments = [x['sentiment'] for x in test]

        class_priors = nuanced_class_log_probabilities(train)
        nuanced_log_probabilities = nuanced_conditional_log_probabilities(train)
        preds_nuanced = []
        for review in dev_tokens:
            pred = predict_nuanced_sentiment_nbc(review, nuanced_log_probabilities, class_priors)
            preds_nuanced.append(pred)
        acc_nuanced = nuanced_accuracy(preds_nuanced, dev_sentiments)
        accuracies.append(acc_nuanced)

    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Your accuracy on the nuanced dataset: {mean_accuracy}\n")

    review_predictions = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions.csv'))

    print('Agreement table for this year.')

    agreement_table = get_agreement_table(review_predictions)
    print_agreement_table(agreement_table)

    fleiss_kappa = calculate_kappa(agreement_table)

    print(f"The cohen kappa score for the review predictions is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [0, 1]})

    print(f"The cohen kappa score for the review predictions of review 1 and 2 is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [2, 3]})

    print(f"The cohen kappa score for the review predictions of review 3 and 4 is {fleiss_kappa}.\n")

    review_predictions_four_years = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions_2019_2023.csv'))
    agreement_table_four_years = get_agreement_table(review_predictions_four_years)

    print('Agreement table for the years 2019 to 2022.')
    print_agreement_table(agreement_table_four_years)

    fleiss_kappa = calculate_kappa(agreement_table_four_years)

    print(f"The cohen kappa score for the review predictions from 2019 to 2022 is {fleiss_kappa}.")



if __name__ == '__main__':
    main()
