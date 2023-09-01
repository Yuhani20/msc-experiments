from rouge import Rouge
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = "data/bartnewnew/"
sns.set(rc={'figure.figsize': (6, 3)})
sns.set_style('whitegrid')

originals = []
summaries = []
highlights = []

for batch in range(0, 1035):
    with open(path + str(batch) + "summary.txt", "r") as file:
        summaries.append(file.read())
        file.close()

    with open(path + str(batch) + "highlights.txt", "r") as file:
        highlights.append(file.read())
        file.close()

    with open("data/copy/" + str(batch) + "summary.txt", "r") as file:
        originals.append(file.read())
        file.close()

rouge = Rouge()
scores = rouge.get_scores(summaries, highlights)
print(rouge.get_scores(summaries, highlights, avg=True))
df = pd.DataFrame(list(zip(summaries, highlights, originals)), columns=['hypothesis', 'reference', 'originals'])

sc = pd.json_normalize(scores)

df = pd.concat([df, sc], axis=1)

def len_tab(df):
    df['hyp_len'] = df['hypothesis'].str.len()
    df['doc_len'] = df['originals'].str.len()

    df1 = pd.concat([df['rouge-1.r'], df['rouge-1.p'], df['rouge-1.f'], df['hyp_len'], df['doc_len']], axis=1)
    df2 = pd.concat([df['rouge-2.r'], df['rouge-2.p'], df['rouge-2.f'], df['hyp_len'], df['doc_len']], axis=1)
    dfl = pd.concat([df['rouge-l.r'], df['rouge-l.p'], df['rouge-l.f'], df['hyp_len'], df['doc_len']], axis=1)

    df1.columns = ['Recall', 'Precision', 'F1', 'Hyp Len', 'Doc Len']
    df1['Type'] = 'Unigram'
    df2.columns = ['Recall', 'Precision', 'F1', 'Hyp Len', 'Doc Len']
    df2['Type'] = 'Bigram'
    dfl.columns = ['Recall', 'Precision', 'F1', 'Hyp Len', 'Doc Len']
    dfl['Type'] = 'LCS'

    dft = pd.concat([df1, df2, dfl], axis=0).reset_index(drop=True)

    # g = sns.scatterplot(x=dft['Hyp Len'],y=dft['F1'], alpha= 0.1,s=3,palette = 'bright', hue=dft["Type"])

    # g = sns.kdeplot(x=dft['Hyp Len'],y=dft['F1'],palette = 'bright', hue=dft["Type"])

    # g = sns.lineplot(x='Hyp Len', y='F1', hue="Type", err_style="band", palette = 'bright', estimator="median", ci='sd', data=dft)

    g = sns.lmplot(x='Hyp Len', y='F1', hue="Type", palette = 'bright', scatter_kws={'alpha': 0.1,'s':3}, order=10, data=dft)

    g.set(ylim=(0, 1))
    g.set(xlim=(0, 1000))

    plt.show()
    return


len_tab(df)
print(df)
