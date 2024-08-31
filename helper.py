# File containing helper functions for sentiment analysis
# @author Milindi Kodikara, RMIT University, 2024

from collections import Counter
import re
import charts
import nltk


oliver = '#275c4d'


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


