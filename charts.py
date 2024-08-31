import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = (10, 8)  # default plot size
sns.set(style='whitegrid', palette='Dark2')


def generic_chart(title, x_label, y_label):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=90)
    plt.show()


def generate_bar_chart(x, y, color, title, x_label, y_label):
    plt.bar(x, y, color=color)
    generic_chart(title, x_label, y_label)


def generate_scatter_plot(x, y, color, title, x_label, y_label):
    plt.scatter(x, y, color=color)
    generic_chart(title, x_label, y_label)
