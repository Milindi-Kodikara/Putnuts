# File containing helper functions for sentiment analysis
# @author Milindi Kodikara, RMIT University, 2024

from collections import Counter
import re

import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

import charts
from utils import *


def process(text, tokeniser, stop_words):
    """
        Perform the processing
        @param text: the text (reddit post) to process

        @returns: list of (valid) tokens in text
    """
    # conversion to lowercase
    text = text.lower()
    text = re.sub(u"(\u2018|\u2019|\u2014|\u201C|\u201D)", "", text)

    # tokenizer
    tokens = tokeniser.tokenize(text)

    # strip whitespace
    tokens_stripped = [tok.strip() for tok in tokens]

    # TODO: Remove abbreviations and replace acronyms with natural language words
    # TODO: Remove emojis
    # TODO: Remove user name tags and mentions

    # regex pattern for digits
    # Note: the list comprehension in return statement essentially remove all strings of digits or fractions, e.g., 6.15
    regex_digit = re.compile("^\d+\s|\s\d+\s|\s\d+|\d+$")
    # regex pattern to remove http
    regex_http = re.compile("^http")

    # remove stop words, http, remove digits (with decimals and fractions)
    return [tok for tok in tokens_stripped if
            tok not in stop_words and regex_digit.match(tok) is None and regex_http.match(tok) is None]


def compute_term_freq(token_list, generate_visual, color=oliver):
    term_freq = 50
    term_freq_counter = Counter()

    term_freq_counter.update(token_list)

    print("-----------------\nTerm frequency\n-----------------\n")
    for term, count in term_freq_counter.most_common(term_freq):
        print(term + ': ' + str(count))
    print("----------------------------------\n")

    if generate_visual:
        y = [count for term, count in term_freq_counter.most_common(term_freq)]
        x = [term for term, count in term_freq_counter.most_common(term_freq)]

        charts.generate_bar_chart(x, y, color, "Term frequency distribution", 'Term frequency',
                                  '# of words with term frequency')


def compute_count_sentiment(token_list, positive_words, negative_words):
    positive_word_count = len([tok for tok in token_list if tok in positive_words])
    negative_word_count = len([tok for tok in token_list if tok in negative_words])

    sentiment = positive_word_count - negative_word_count

    return sentiment


def print_sentiment(sentiment):
    print('\n\n------------Count sentiment value------------\n')
    if sentiment > 0:
        print(oliver_rgb + str(sentiment), end='')
    elif sentiment < 0:
        print(mabel_rgb + str(sentiment), end='')
    else:
        print(charles_rgb + str(sentiment), end='')
    print('\n------------------------------------\n\n')


def print_coloured_tokens(method, token_list, sentiment, positive_words=None, negative_words=None):
    if positive_words is None:
        positive_words = []
    if method == 'Count':
        for token in token_list:
            if token in positive_words:
                print(oliver_rgb + token + ', ', end='')
            elif token in negative_words:
                print(mabel_rgb + token + ', ', end='')
            else:
                print(charles_rgb + token + ', ', end='')

        print_sentiment(sentiment)

    if method == 'Vader':
        for cat, score in sentiment.items():
            print(cat)
            print_sentiment(score)


def sentiment_analysis(method, omitb_df, b_print):
    """
    @returns: list of reddit posts' sentiments, in the format of [date, sentiment]
    """
    set_pos_words = []
    set_neg_words = []
    if method == 'Count':
        # load pos, neg word lists
        set_pos_words = read_file('positive-words.txt')
        set_neg_words = read_file('negative-words.txt')

    sentiment_list = []
    vader_sentiment_analyser = SentimentIntensityAnalyzer()

    for row in omitb_df.itertuples(index=False):

        token_list = row.Processed_tokens
        date = row.UTC_Date
        sentiment = 0

        # compute sentiment
        if method == 'Vader':
            sentiment = vader_sentiment_analyser.polarity_scores(" ".join(token_list))
            sentiment_list.append([date, sentiment['compound']])
        elif method == 'Count':
            sentiment = compute_count_sentiment(token_list, set_pos_words, set_neg_words)
            # save the date and sentiment of each reddit post
            sentiment_list.append([date, sentiment])

        if b_print:
            print_coloured_tokens(method, token_list, sentiment, set_pos_words, set_neg_words)

    return sentiment_list
