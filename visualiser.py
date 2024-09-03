import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams["figure.figsize"] = (40, 24)  # default plot size
sns.set(style='whitegrid', palette='Dark2')


def generic_chart(title, x_label, y_label):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=90)
    plt.tight_layout()
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


def display_topics(model, feature_names, num_top_words):
    """
    Prints out the most associated words for each feature.

    @param model: lda model
    @param feature_names: list of strings, representing the list of features/words.
    @param num_top_words: number of words to print per topic.
    """

    # print out the topic distributions
    for topic_id, topic_distribution_list in enumerate(model.components_):
        print("Topic %d:" % (topic_id))
        print(" ".join([feature_names[i] for i in topic_distribution_list.argsort()[:-num_top_words - 1:-1]]))


def display_word_cloud(model, feature_names):
    """
    Displays the word cloud of the topic distributions, stored in model.

    @param model: lda model.
    @param feature_names: list of strings, representing the list of features/words.
    """

    # this normalises each row/topic to sum to one
    # use this normalised_components to display your wordclouds
    normalised_components = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]

    topic_num = len(model.components_)
    # number of wordclouds for each row
    plot_col_num = 3
    # number of wordclouds for each column
    plot_row_num = int(math.ceil(topic_num / plot_col_num))

    for topicId, lTopicDist in enumerate(normalised_components):
        l_word_prob = {feature_names[i]: wordProb for i, wordProb in enumerate(lTopicDist)}
        wordcloud = WordCloud(background_color='black')
        wordcloud.fit_words(frequencies=l_word_prob)
        plt.subplot(plot_row_num, plot_col_num, topicId + 1)
        plt.title('Topic %d:' % (topicId + 1))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

    plt.show(block=True)