from utils.sentiment_detection import clean_plot, read_tokens, chart_plot, best_fit
from typing import List, Tuple, Callable
import os
import math


def estimate_zipf(token_frequencies_log: List[Tuple[float, float]], token_frequencies: List[Tuple[int, int]]) \
        -> Callable:
    """
    Use the provided least squares algorithm to estimate a line of best fit in the log-log plot of rank against
    frequency. Weight each word by its frequency to avoid distortion in favour of less common words. Use this to
    create a function which given a rank can output an expected frequency.

    @param token_frequencies_log: list of tuples of log rank and log frequency for each word
    @param token_frequencies: list of tuples of rank to frequency for each word used for weighting
    @return: a function estimating a word's frequency from its rank
    """
    params = best_fit(token_frequencies_log,token_frequencies)
    print("alpha:",-params[0])
    print("k:",math.exp(params[1]))
    def rank_frequency_projection(rank):
        log_rank = math.log(rank)
        log_frequency = params[1]+log_rank*params[0]
        frequency = math.exp(log_frequency)
        return frequency
    return rank_frequency_projection


def count_token_frequencies(dataset_path: str) -> List[Tuple[str, int]]:
    """
    For each of the words in the dataset, calculate its frequency within the dataset.

    @param dataset_path: a path to a folder with a list of  reviews
    @returns: a list of the frequency for each word in the form [(word, frequency), (word, frequency) ...], sorted by
        frequency in descending order
    """
    mapp = dict()
    for i in range(35566):
        if(i%1000==0):
            print(i)
        nowfilepath = os.path.join(dataset_path, str(i))
        tokenized_data = read_tokens(nowfilepath)
        for word in tokenized_data:
            if word in mapp:
                mapp[word] += 1
            else:
                mapp[word] = 1
    return sorted(mapp.items(), key= lambda item: item[1], reverse=True)


def draw_frequency_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the provided chart plotting program to plot the most common 10000 word ranks against their frequencies.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    top_frequencies = frequencies[:10000]
    top_frequencies = [ (i+1,item[1])for i,item in enumerate(top_frequencies)]
    chart_plot(top_frequencies,title="frequency vs. rank for top 10000 words",x_label="rank",y_label="frequency")




def draw_selected_words_ranks(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot your 10 selected words' word frequencies (from Task 1) against their
    ranks. Plot the Task 1 words on the frequency-rank plot as a separate series on the same plot (i.e., tell the
    plotter to draw it as an additional graph on top of the graph from above function).

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    import matplotlib.pyplot as plt
    wordlist = ["great","well","interesting","mistaken","awesome","unique","pain","relax","thought-provoking","fun"]
    top_frequencies = frequencies[:10000]
    original_top_frequencies = top_frequencies.copy()
    top_frequencies = [(i + 1, item[1]) for i, item in enumerate(top_frequencies)]
    plt.plot([x[0] for x in top_frequencies], [x[1] for x in top_frequencies], '-o', markersize=3)
    additional_plot = []
    for i,x in enumerate(original_top_frequencies):
        if x[0] in wordlist:
            additional_plot.append((i+1,x[1]))
    plt.plot([x[0] for x in additional_plot],[x[1] for x in additional_plot],'-o',label = "the chosen words")
    plt.xlabel("rank")
    plt.ylabel("frequency")
    plt.legend()
    directory = 'figures/sentiment_detection/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, 'plot with the chosen words.png'), dpi=300)
    # chart_plot(top_frequencies, title="frequency vs. rank for top 10000 words", x_label="rank", y_label="frequency")


def draw_zipf(frequencies: List[Tuple[str, int]]) -> None:
    """
    Use the chart plotting program to plot the logs of your first 10000 word frequencies against the logs of their
    ranks. Also use your estimation function to plot a line of best fit.

    @param frequencies: A sorted list of tuples with the first element being the word and second its frequency
    """
    import matplotlib.pyplot as plt
    top_frequencies = frequencies[:10000]
    top_frequencies = [(i + 1, item[1]) for i, item in enumerate(top_frequencies)]
    top_frequencies_log = [(math.log(i + 1), math.log(item[1])) for i, item in enumerate(top_frequencies)]
    estimate_zipf(top_frequencies_log,top_frequencies)
    x_label = "log(rank)"
    y_label = "log(frequency)"
    title = "log(requency vs. rank for top 10000 words)"
    plt.plot([x[0] for x in top_frequencies_log], [x[1] for x in top_frequencies_log], '-o', markersize=3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    params = best_fit(top_frequencies_log, top_frequencies)
    x = list(range(11))
    y = [params[1] + i * params[0] for i in x]
    plt.plot(x, y, label="line of best fit")
    plt.legend()
    directory = 'figures/sentiment_detection/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, f'{title}.png'), dpi=300)




def compute_type_count(dataset_path: str) -> List[Tuple[int, int]]:
    """
     Go through the words in the dataset; record the number of unique words against the total number of words for total
     numbers of words that are powers of 2 (starting at 2^0, until the end of the data-set)

     @param dataset_path: a path to a folder with a list of  reviews
     @returns: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    types = set()
    res = []
    tokensize=0
    for i in range(35566):
        if (i % 1000 == 0):
            print(i)
        nowfilepath = os.path.join(dataset_path, str(i))
        tokenized_data = read_tokens(nowfilepath)
        for word in tokenized_data:
            tokensize+=1
            types.add(word)
            if is_powerof2(tokensize):
                res.append((tokensize,len(types)))
    print("total number of token in all texts:",tokensize)
    return res

def is_powerof2(num):
    if num <= 0:
        return False
    return (math.log2(num) % 1) == 0

def draw_heap(type_counts: List[Tuple[int, int]]) -> None:
    """
    Use the provided chart plotting program to plot the logs of the number of unique words against the logs of the
    number of total words.

    @param type_counts: the number of types for every n tokens with n being 2^k in the form [(#tokens, #types), ...]
    """
    import matplotlib.pyplot as plt
    x_label = "log(token size)"
    y_label = "log(types size)"
    title = "Heap's law"
    plt.plot([math.log(x[0]) for x in type_counts], [math.log(x[1]) for x in type_counts], '-o', markersize=3)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    directory = 'figures/sentiment_detection/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, f'{title}.png'), dpi=300)


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    frequencies = count_token_frequencies(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))

    draw_frequency_ranks(frequencies)
    draw_selected_words_ranks(frequencies)

    clean_plot()
    draw_zipf(frequencies)

    clean_plot()
    tokens = compute_type_count(os.path.join('data', 'sentiment_detection', 'reviews_large', 'reviews'))
    draw_heap(tokens)


if __name__ == '__main__':
    main()
