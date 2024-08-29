# File containing helper functions for sentiment analysis
# @author Milindi Kodikara, RMIT University, 2024

import re


def process(text, tokeniser, stop_words):
    """
        Perform the processing
        @param text: the text (reddit post) to process

        @returns: list of (valid) tokens in text
    """
    # conversion to lowercase
    text = text.lower()

    # tokenizer
    tokens = tokeniser.tokenize(text)

    # strip whitespace
    tokens_stripped = [tok.strip() for tok in tokens]

    # TODO: Remove abbreviations and replace acronyms with natural language words
    # TODO: Remove emojis
    # TODO: Remove user name tags and mentions

    # regex pattern for digits
    # Note: the list comprehension in return statement essentially remove all strings of digits or fractions, e.g., 6.15
    regex_digit = re.compile("^\d+\s|\s\d+\s|\s\d+$")
    # regex pattern to remove http
    regex_http = re.compile("^http")

    # remove stop words, http, remove digits (with decimals and fractions)
    return [tok for tok in tokens_stripped if
            tok not in stop_words and regex_digit.match(tok) == None and regex_http.match(tok) == None]
