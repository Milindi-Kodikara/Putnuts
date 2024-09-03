import re


def get_color_escape(r, g, b, background=False):
    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)


oliver = '#275c4d'

oliver_rgb = get_color_escape(39, 92, 77)
mabel_rgb = get_color_escape(175, 34, 29)
charles_rgb = get_color_escape(197, 145, 3)
RESET = '\033[0m'


def read_file(filename):
    item_list = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            item_list.append(line.strip())

    return set(item_list)

# https://stackoverflow.com/questions/73804264/removing-emojis-and-special-characters-in-python


regex_emojis = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002500-\U00002BEF"  # chinese char
                          u"\U00002702-\U000027B0"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          u"\U0001f926-\U0001f937"
                          u"\U00010000-\U0010ffff"
                          u"\u2640-\u2642"
                          u"\u2600-\u2B55"
                          u"\u200d"
                          u"\u23cf"
                          u"\u23e9"
                          u"\u231a"
                          u"\ufe0f"
                          u"\u3030"
                          "]+", re.UNICODE)
