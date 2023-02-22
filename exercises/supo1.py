from utils.sentiment_detection import read_tokens, load_reviews, split_data
from tick1 import predict_sentiment_improved, read_lexicon, predict_sentiment
from tick2 import calculate_class_log_probabilities, calculate_smoothed_log_probabilities, calculate_unsmoothed_log_probabilities, predict_sentiment_nbc
import os



def data_preprocessing():
    tokenized_data = read_tokens(os.path.join("data", "supo1", "review"))
    tokenized_data.sort()
    tokenized_data = list(dict.fromkeys(tokenized_data))
    print(tokenized_data)

def q1tick1():
    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))
    review = ['.', '14', '5', 'D600', 'I', 'Nikon', 'That', 'The', 'Z6ii', 'a', 'action', 'an', 'and', 'are', 'at',
              'camera', 'can', 'continuous', 'familiar', 'features', 'fiddly', 'for', 'fps', 'freeze', 'from', 'had',
              'have', 'hidden', 'invaluable', 'is', 'like', 'lovely', 'many', 'menus', 'new', 'newer', 'nine', 'old',
              'photographers', 'prove', 'say', 'shutter', 'someone', 'speed', 'sports', 'that', 'the', 'there', 'those',
              'to', 'upgraded', 'use', 'versus', 'which', 'who', 'wildlife', 'with', 'within', 'years']
    print(f"tick1 - original:{predict_sentiment(review, lexicon)}")
    print(f"tick1 - improved:{predict_sentiment_improved(review, lexicon)}")



def q1tick2():
    review = ['.', '14', '5', 'D600', 'I', 'Nikon', 'That', 'The', 'Z6ii', 'a', 'action', 'an', 'and', 'are', 'at',
              'camera', 'can', 'continuous', 'familiar', 'features', 'fiddly', 'for', 'fps', 'freeze', 'from', 'had',
              'have', 'hidden', 'invaluable', 'is', 'like', 'lovely', 'many', 'menus', 'new', 'newer', 'nine', 'old',
              'photographers', 'prove', 'say', 'shutter', 'someone', 'speed', 'sports', 'that', 'the', 'there', 'those',
              'to', 'upgraded', 'use', 'versus', 'which', 'who', 'wildlife', 'with', 'within', 'years']

    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    training_data, validation_data = split_data(review_data, seed=0)
    train_tokenized_data = [{'text': read_tokens(x['filename']), 'sentiment': x['sentiment']} for x in training_data]
    dev_tokenized_data = [read_tokens(x['filename']) for x in validation_data]
    validation_sentiments = [x['sentiment'] for x in validation_data]

    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    unsmoothed_log_probabilities = calculate_unsmoothed_log_probabilities(train_tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)

    pred1 = predict_sentiment_nbc(review,unsmoothed_log_probabilities, class_priors)
    pred2 = predict_sentiment_nbc(review,smoothed_log_probabilities, class_priors)
    print(f"unsmoothed : {pred1}")
    print(f"smoothed : {pred2}")
def main():
    # q1tick2()
    q1tick1()
if __name__ == "__main__":
    main()