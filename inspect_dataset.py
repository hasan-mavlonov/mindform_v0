# inspect_dataset.py

from datasets import load_dataset

ds = load_dataset("jingjietan/pandora-big5")

print(ds)

print(ds["train"][0])