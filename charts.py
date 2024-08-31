import matplotlib.pyplot as plt
import pandas as pd
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


# TODO: Fix index issue here
def generate_time_series(item_list):
    series = pd.DataFrame(item_list, columns=['date', 'sentiment'])
    series.set_index('date', inplace=True)
    series[['sentiment']] = series[['sentiment']].apply(pd.to_numeric)
    # TODO: play with this for different resolution, '1H' is by hour, '1M' is by minute etc
    new_series = series.resample('1D').sum()
    # this plots and shows the time series
    new_series.plot()
    plt.show()
