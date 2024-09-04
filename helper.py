# File containing helper functions for sentiment analysis
# @author Milindi Kodikara, RMIT University, 2024
from collections import Counter
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

import visualiser
from utils import *


def process(text, tokeniser, stemmer, stop_words, print_processing=False):
    """
        Perform the processing
        @param text: the text (reddit post) to process

        @returns: list of (valid) tokens in text
    """
    start = '\n\n------------------------------------\n'
    end = '\n------------------------------------\n\n'
    if print_processing:
        print(charles_rgb + f'{start}Initial text\n{text}\n{end}')

    # conversion to lowercase
    text = text.lower()
    if print_processing:
        print(mabel_rgb + f'{start}Lowercase text{start}{text}\n{end}', end='')

    # remove curly inverted commas
    text = re.sub(u"(\u2018|\u2019|\u2014|\u201C|\u201D)", "", text)

    if print_processing:
        print(mabel_rgb + f'{start}Inverted comma removed text{start}{text}\n{end}', end='')

    # remove emojis
    text = re.sub(regex_emojis, '', text)
    if print_processing:
        print(mabel_rgb + f'{start}Emoji removed text{start}{text}\n{end}', end='')

    # remove username tags, mentions, and links
    text = re.sub(r'(r/|@|https?)\S+|#', '', text)
    if print_processing:
        print(mabel_rgb + f'{start}Tags, mentions and links removed text{start}{text}\n{end}', end='')

    # tokenizer
    tokens = tokeniser.tokenize(text)

    if print_processing:
        print(mabel_rgb + f'{start}Tokenized text{start}{tokens}\n{end}', end='')

    # strip whitespace
    tokens = [tok.strip() for tok in tokens]

    if print_processing:
        print(mabel_rgb + f'{start}Whitespace stripped tokenized text{start}{tokens}\n{end}', end='')

    # remove duplicates, TODO: Check if I should use stemmer
    # tokens = set([stemmer.stem(tok) for tok in tokens])

    # remove digits
    tokens = [tok for tok in tokens if not tok.isdigit()]
    if print_processing:
        print(mabel_rgb + f'{start}Digits removed tokenized text{start}{tokens}\n{end}', end='')

    # remove stop words
    tokens = [tok for tok in tokens if tok not in stop_words]

    if print_processing:
        print(mabel_rgb + f'{start}Stop words removed tokenized text{start}{tokens}\n{end}', end='')

    if print_processing:
        print(oliver_rgb + f'{start}Final tokenized text{start}{tokens}\n{end}', end='')

    return tokens


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

        visualiser.generate_bar_chart(x, y, color, "Term frequency distribution", 'Term frequency',
                                  '# of words with term frequency')


def compute_count_sentiment(token_list, positive_words, negative_words):
    positive_word_count = len([tok for tok in token_list if tok in positive_words])
    negative_word_count = len([tok for tok in token_list if tok in negative_words])

    sentiment = positive_word_count - negative_word_count

    return sentiment


def print_sentiment(sentiment, prefix=''):
    start = '\n\n------------Count sentiment value------------\n'
    end = '\n------------------------------------\n\n'
    if sentiment > 0:
        print(oliver_rgb + start + prefix + str(sentiment) + end, end='')
    elif sentiment < 0:
        print(mabel_rgb + start + prefix + str(sentiment) + end, end='')
    else:
        print(charles_rgb + start + prefix + str(sentiment) + end, end='')


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
            print(*token_list, sep=', ')
            prefix = '{}: '.format(cat)
            print_sentiment(score, prefix)


def sentiment_analysis(method, omitb_df):
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

    for row in omitb_df.itertuples(index=True):
        print_processing = True if row[0] <= 10 else False

        token_list = row.Processed_tokens
        date = row.UTC_Date
        sentiment = 0

        # compute sentiment
        if method == 'Vader':
            sentiment = vader_sentiment_analyser.polarity_scores(" ".join(token_list))
            sentiment_list.append([pd.to_datetime(date, unit='s'), sentiment['compound']])
        elif method == 'Count':
            sentiment = compute_count_sentiment(token_list, set_pos_words, set_neg_words)
            # save the date and sentiment of each reddit post
            sentiment_list.append([pd.to_datetime(date, unit='s'), sentiment])

        if print_processing:
            post = row.Post
            num_comments = row.Num_comments
            date = row.Date

            start = '\n\n------------Analysing sentiment------------\n'
            end = '\n------------------------------------\n\n'
            formatted_post = f'Date: {date}\n\nPost:\n{post}\n\nNum Comments: {num_comments}'

            print(charles_rgb + start + formatted_post + end, end='')

            print_coloured_tokens(method, token_list, sentiment, set_pos_words, set_neg_words)

    return sentiment_list
