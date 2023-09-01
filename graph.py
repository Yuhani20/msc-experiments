from rouge import Rouge
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = "data/bartnewnew/"
sns.set(rc={'figure.figsize': (6, 3)})
sns.set_style('whitegrid')

summaries = []
highlights = []

for batch in range(0, 1035):
    with open(path + str(batch) + "summary.txt", "r") as file:
        summaries.append(file.read())
        file.close()

    with open(path + str(batch) + "highlights.txt", "r") as file:
        highlights.append(file.read())
        file.close()

rouge = Rouge()
scores = rouge.get_scores(summaries, highlights)
print(rouge.get_scores(summaries, highlights, avg=True))
df = pd.DataFrame(list(zip(summaries, highlights)), columns=['hypothesis', 'reference'])

sc = pd.json_normalize(scores)

df = pd.concat([df, sc], axis=1)


def plot_hist(df):
    df1 = pd.concat([df['rouge-1.r'], df['rouge-1.p'], df['rouge-1.f']], axis=1)
    df2 = pd.concat([df['rouge-2.r'], df['rouge-2.p'], df['rouge-2.f']], axis=1)
    dfl = pd.concat([df['rouge-l.r'], df['rouge-l.p'], df['rouge-l.f']], axis=1)

    ax1 = df1.plot.hist(bins=100, alpha=0.3)
    ax2 = df2.plot.hist(bins=100, alpha=0.3)
    ax3 = dfl.plot.hist(bins=100, alpha=0.3)

    ax1.axis(xmin=0, xmax=1)
    ax2.axis(xmin=0, xmax=1)
    ax3.axis(xmin=0, xmax=1)

    plt.show()


def plot_corr(df):
    df1 = pd.concat([df['rouge-1.r'], df['rouge-1.p'], df['rouge-1.f']], axis=1)
    df2 = pd.concat([df['rouge-2.r'], df['rouge-2.p'], df['rouge-2.f']], axis=1)
    dfl = pd.concat([df['rouge-l.r'], df['rouge-l.p'], df['rouge-l.f']], axis=1)

    df1.columns = ['Recall', 'Precision', 'F1']
    df1['Type'] = 'Unigram'
    df2.columns = ['Recall', 'Precision', 'F1']
    df2['Type'] = 'Bigram'
    dfl.columns = ['Recall', 'Precision', 'F1']
    dfl['Type'] = 'LCS'

    dft = pd.concat([df1, df2, dfl], axis=0).reset_index(drop=True)

    g = sns.pairplot(dft, hue="Type", palette = 'bright', plot_kws={"alpha": 0.1,"s":3}, grid_kws={"layout_pad":1.5})

    g.axes[0,1].set_xlim(0,1)
    g.axes[0,2].set_xlim(0,1)
    g.axes[1,2].set_xlim(0,1)
    g.axes[1,0].set_xlim(0,1)
    g.axes[2,0].set_xlim(0,1)
    g.axes[2,1].set_xlim(0,1)
    g.axes[0,1].set_ylim(0,1)
    g.axes[0,2].set_ylim(0,1)
    g.axes[1,2].set_ylim(0,1)
    g.axes[1,0].set_ylim(0,1)
    g.axes[2,0].set_ylim(0,1)
    g.axes[2,1].set_ylim(0,1)

    plt.show()
    return


plot_hist(df)
plot_corr(df)
