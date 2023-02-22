import math
from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, split_data
from exercises.tick1 import accuracy, predict_sentiment, read_lexicon


def calculate_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to prior log probability
    """
    res = dict()
    trainsize = len(training_data)
    possize =0
    for i in training_data:
        if(i["sentiment"]==1):
            possize+=1
    res[1]= math.log(possize/trainsize)
    res[-1]= math.log(1-possize/trainsize)
    return res




def calculate_unsmoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the unsmoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    res = dict()
    res[1]=dict()
    res[-1]=dict()
    possum = 0
    negsum = 0
    for dic in training_data:
        if dic["sentiment"]==1:
            possum+=len(dic["text"])
            for word in dic["text"]:
                if word in res[1]:
                    res[1][word]+=1
                else:
                    res[1][word]=1
        else:
            negsum+=len(dic["text"])
            for word in dic["text"]:
                if word in res[-1]:
                    res[-1][word]+=1
                else:
                    res[-1][word]=1
    for sent in res:
        for word in res[sent]:
            res[sent][word]=math.log(res[sent][word]/(possum if sent==1 else negsum))

    return res


def calculate_smoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment. Use the smoothing
    technique described in the instructions (Laplace smoothing).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: Dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    res = dict()
    res[1] = dict()
    res[-1] = dict()
    possum = 0
    negsum = 0
    wordset = set()
    for dic in training_data:
        if dic["sentiment"] == 1:
            possum += len(dic["text"])
            for word in dic["text"]:
                wordset.add(word)
                if word in res[1]:
                    res[1][word] += 1
                else:
                    res[1][word] = 1
        else:
            negsum += len(dic["text"])
            for word in dic["text"]:
                wordset.add(word)
                if word in res[-1]:
                    res[-1][word] += 1
                else:
                    res[-1][word] = 1
    for word in wordset:
        if word not in res[1]:
            res[1][word]=0
        if word not in res[-1]:
            res[-1][word]=0
    for sent in res:
        for word in res[sent]:
            res[sent][word] = math.log((res[sent][word]+1) / ((possum if sent == 1 else negsum)+len(wordset)))

    return res


def predict_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                          class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior log probability
    @return: predicted sentiment [-1, 1] for the given review
    """
    posprob = class_log_probabilities[1]
    negprob = class_log_probabilities[-1]
    for word in review:
        if word in log_probabilities[1]:
            posprob+=log_probabilities[1][word]
        if word in log_probabilities[-1]:
            negprob+=log_probabilities[-1][word]

    return 1 if (posprob>negprob) else -1




def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    training_data, validation_data = split_data(review_data, seed=0)
    train_tokenized_data = [{'text': read_tokens(x['filename']), 'sentiment': x['sentiment']} for x in training_data]
    dev_tokenized_data = [read_tokens(x['filename']) for x in validation_data]
    validation_sentiments = [x['sentiment'] for x in validation_data]

    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    preds_simple = []
    for review in dev_tokenized_data:
        pred = predict_sentiment(review, lexicon)
        preds_simple.append(pred)

    acc_simple = accuracy(preds_simple, validation_sentiments)
    print(f"Your accuracy using simple classifier: {acc_simple}")

    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    unsmoothed_log_probabilities = calculate_unsmoothed_log_probabilities(train_tokenized_data)
    preds_unsmoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, unsmoothed_log_probabilities, class_priors)
        preds_unsmoothed.append(pred)

    acc_unsmoothed = accuracy(preds_unsmoothed, validation_sentiments)
    print(f"Your accuracy using unsmoothed probabilities: {acc_unsmoothed}")

    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)
    preds_smoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_smoothed.append(pred)

    acc_smoothed = accuracy(preds_smoothed, validation_sentiments)
    print(f"Your accuracy using smoothed probabilities: {acc_smoothed}")


if __name__ == '__main__':
    main()
