from datasets import load_dataset
import os

dataset = load_dataset("szymonrucinski/types-of-film-shots")

# get label names
label_names = dataset["train"].features["label"].names

def map_label(label_id):
    label = label_names[label_id].lower()

    if "close" in label or "detail" in label:
        return "CLOSE"
    elif "medium" in label:
        return "MEDIUM"
    elif "long" in label or "full" in label:
        return "WIDE"
    else:
        return None  # skip ambiguous

# create folders
base_path = "datasets/processed"
for cls in ["CLOSE", "MEDIUM", "WIDE"]:
    os.makedirs(f"{base_path}/{cls}", exist_ok=True)

count = {"CLOSE":0, "MEDIUM":0, "WIDE":0}

for i, sample in enumerate(dataset["train"]):
    mapped = map_label(sample["label"])

    if mapped is None:
        continue

    img = sample["image"]
    path = f"{base_path}/{mapped}/{i}.jpg"
    img.save(path)

    count[mapped] += 1

print("Final distribution:", count)