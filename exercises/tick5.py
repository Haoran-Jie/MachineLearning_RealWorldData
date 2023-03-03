import math
from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, print_binary_confusion_matrix
from exercises.tick1 import accuracy, predict_sentiment,read_lexicon
from exercises.tick2 import predict_sentiment_nbc, calculate_smoothed_log_probabilities, \
    calculate_class_log_probabilities
from exercises.tick4 import sign_test


def generate_random_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, random.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    size = len(training_data)

    res = [training_data[i*(size//n):(i+1)*(size//n)] for i in range(n-1)]
    res.append(training_data[(n-1)*(size//n):])
    return res


def generate_stratified_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, stratified.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    res = [[] for i in range(n)]
    posptr=0
    negptr=0
    for training_instance in training_data:
        if training_instance['sentiment']==1:
            res[posptr].append(training_instance)
            posptr = (posptr+1)%n
        else:
            res[negptr].append(training_instance)
            negptr = (negptr+1)%n
    return res




def cross_validate_nbc(split_training_data: List[List[Dict[str, Union[List[str], int]]]]) -> List[float]:
    """
    Perform an n-fold cross validation, and return the mean accuracy and variance.

    @param split_training_data: a list of n folds, where each fold is a list of training instances, where each instance
        is a dictionary with two fields: 'text' and 'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or
        -1, for positive and negative sentiments.
    @return: list of accuracy scores for each fold
    """
    accuracy_scores = []
    for i in range(len(split_training_data)):
        curtest = split_training_data[i].copy()
        curtrain = split_training_data.copy()
        del curtrain[i]
        curtrain = [item for sublist in curtrain for item in sublist]
        class_log_probabilities = calculate_class_log_probabilities(curtrain)
        log_probabilities = calculate_smoothed_log_probabilities(curtrain)
        pred = [predict_sentiment_nbc(item['text'],log_probabilities,class_log_probabilities) for item in curtest]
        true = [item['sentiment'] for item in curtest]
        accuracy_scores.append(accuracy(pred,true))
    return  accuracy_scores




def cross_validation_accuracy(accuracies: List[float]) -> float:
    """Calculate the mean accuracy across n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: mean accuracy over the cross folds
    """
    return sum(accuracies)/len(accuracies)


def cross_validation_variance(accuracies: List[float]) -> float:
    """Calculate the variance of n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: variance of the cross fold accuracies
    """
    mean = cross_validation_accuracy(accuracies)
    res = sum([math.pow(item-mean,2) for item in accuracies])/len(accuracies)
    return res


def confusion_matrix(predicted_sentiments: List[int], actual_sentiments: List[int]) -> List[List[int]]:
    """
    Calculate the number of times (1) the prediction was POS and it was POS [correct], (2) the prediction was POS but
    it was NEG [incorrect], (3) the prediction was NEG and it was POS [incorrect], and (4) the prediction was NEG and it
    was NEG [correct]. Store these values in a list of lists, [[(1), (2)], [(3), (4)]], so they form a confusion matrix:
                     actual:
                     pos     neg
    predicted:  pos  [[(1),  (2)],
                neg   [(3),  (4)]]

    @param actual_sentiments: a list of the true (gold standard) sentiments
    @param predicted_sentiments: a list of the sentiments predicted by a system
    @returns: a confusion matrix
    """
    res = [[0,0],[0,0]]
    mapping = {1:0,-1:1}
    for pred,actual in zip(predicted_sentiments,actual_sentiments):
        res[mapping[pred]][mapping[actual]]+=1
    return res


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    # First test cross-fold validation
    folds = generate_random_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Random cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Random cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Random cross validation variance: {variance}\n")

    folds = generate_stratified_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Stratified cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Stratified cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Stratified cross validation variance: {variance}\n")

    # Now evaluate on 2016 and test
    class_priors = calculate_class_log_probabilities(tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(tokenized_data)

    preds_test = []
    preds_test_simple = []
    test_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_test'))
    test_tokens = [read_tokens(x['filename']) for x in test_data]
    test_sentiments = [x['sentiment'] for x in test_data]
    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))
    for review in test_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        pred_simple = predict_sentiment(review,lexicon)
        preds_test.append(pred)
        preds_test_simple.append(pred_simple)

    acc_smoothed = accuracy(preds_test, test_sentiments)
    acc_test_simple = accuracy(preds_test_simple,test_sentiments)
    print(f"Smoothed Naive Bayes accuracy on held-out data: {acc_smoothed}")
    print(f"Simple classifier (from tick 1) accuracy on held-out data: {acc_test_simple}")
    pvalue_test = sign_test(test_sentiments,preds_test,preds_test_simple)
    print(f"the p-value: difference between the naive bayes classifier and the simple classifier on the test data: {pvalue_test}")
    # print("Confusion matrix:")
    # print_binary_confusion_matrix(confusion_matrix(preds_test, test_sentiments))

    preds_recent = []
    preds_recent_simple = []
    recent_review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_2016'))
    recent_tokens = [read_tokens(x['filename']) for x in recent_review_data]
    recent_sentiments = [x['sentiment'] for x in recent_review_data]
    for review in recent_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        pred_simple = predict_sentiment(review,lexicon)
        preds_recent.append(pred)
        preds_recent_simple.append(pred_simple)

    acc_smoothed = accuracy(preds_recent, recent_sentiments)
    acc_recent_simple = accuracy(preds_recent_simple,recent_sentiments)
    print(f"Smoothed Naive Bayes accuracy on 2016 data: {acc_smoothed}")
    print(f"Simple classifier (from tick 1) accuracy on 2016 data: {acc_recent_simple}")
    pvalue_2016 = sign_test(recent_sentiments,preds_recent,preds_recent_simple)
    print(f"the p-value: difference between the naive bayes classifier and the simple classifier on the 2016 data: {pvalue_2016}")
    # print("Confusion matrix:")
    # print_binary_confusion_matrix(confusion_matrix(preds_recent, recent_sentiments))


if __name__ == '__main__':
    main()
