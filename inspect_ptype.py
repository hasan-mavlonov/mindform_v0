from datasets import load_dataset

ds = load_dataset(
    "jingjietan/pandora-big5"
)

train = ds["train"]

seen = set()

for row in train:
    seen.add(row["ptype"])

print(
    sorted(seen)
)

print(
    "Count:",
    len(seen)
)