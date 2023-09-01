from datasets import load_dataset
from rouge import Rouge
from transformers import pipeline
import torch
import os

from transformers.pipelines.pt_utils import KeyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

for batch in range(0,100000):
    split = "train[" + str(batch) + ":" + str(batch+1) + "]"
    ds = load_dataset('cnn_dailymail', '3.0.0', split=split)

    print(ds['article'])
    print(ds['highlights'])

    with open(str(batch) +"summary.txt", "w+") as file:
        file.writelines(ds['article'])
        file.close()

    with open(str(batch) + "highlights.txt", "w+") as file:
        file.writelines(ds['highlights'])
        file.close()

