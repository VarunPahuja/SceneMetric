from datasets import load_dataset
import os
from PIL import Image

# download dataset
dataset = load_dataset("szymonrucinski/types-of-film-shots")

print(dataset)

# create save folder
os.makedirs("datasets/film_shots", exist_ok=True)

# save a few samples to inspect
for i in range(10):
    sample = dataset["train"][i]
    img = sample["image"]
    label = sample["label"]

    path = f"datasets/film_shots/sample_{i}_{label}.jpg"
    img.save(path)

    print(f"Saved {path}")