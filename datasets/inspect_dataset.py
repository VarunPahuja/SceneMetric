from datasets import load_dataset
from collections import Counter

dataset = load_dataset("szymonrucinski/types-of-film-shots")

# get label names
label_names = dataset["train"].features["label"].names

labels = [sample["label"] for sample in dataset["train"]]
count = Counter(labels)

for k, v in count.items():
    print(label_names[k], v)