from datasets import load_dataset
from rouge import Rouge
from transformers import pipeline
import torch
import os

from transformers.pipelines.pt_utils import KeyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

summarizer = pipeline("summarization")

summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf", truncation=True)

for batch in range(0,3777):
    split = "train[" + str(batch) + ":" + str(batch+1) + "]"
    ds = load_dataset('ccdv/arxiv-summarization', split=split)

    summaries = []
    highlights = []
    results = summarizer(KeyDataset(ds, "article"))

    for id, summary in enumerate(results):
        highlights.append(ds[id]['abstract'])
        summaries.append(summary[0]['summary_text'])

    rouge = Rouge()
    scores = rouge.get_scores(summaries, highlights, avg=True)

    print(summaries)
    print(highlights)
    print(scores)

    with open(str(batch) +"summary.txt", "w+") as file:
        file.writelines(summaries)
        file.close()

    with open(str(batch) + "highlights.txt", "w+") as file:
        file.writelines(highlights)
        file.close()

