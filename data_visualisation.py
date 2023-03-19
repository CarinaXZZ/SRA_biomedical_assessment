import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utilities import pivot_table


def df_plot_bar(df, xlabel=None, ylabel=None, output=None, logscale=True):
    text_position = 0.6
    ax = df.plot.bar(figsize=(16, 8), width=0.9)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.tight_layout()
    if logscale:
        ax.set_yscale('log')
    column_names = df.columns.tolist()
    for i, column in enumerate(column_names):
        for idx, value in enumerate(df[column]):
            ax.text(idx - text_position*(10**(-1*i)), value + 0.7, str(value))
    if output:
        plt.savefig(output)
    plt.show()


def plot_cross_table(df, category1, category2, xlabel=None, ylabel=None, output=None, logscale=True):
    cross_table = pivot_table(df, category1, category2)
    df_plot_bar(cross_table, xlabel=xlabel,
                ylabel=ylabel, output=output, logscale=logscale)


def plot_word_cloud(words, output=None):
    text = " ".join(words)
    wordcloud = WordCloud(width=1600, height=800, background_color='white',
                          max_font_size=150).generate(text)
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    if output:
        plt.savefig(output)
    plt.show()

