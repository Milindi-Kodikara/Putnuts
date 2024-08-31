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
