# inspect_labels.py

from datasets import load_dataset

ds = load_dataset(
    "jingjietan/pandora-big5"
)

sample = ds["train"]

for i in range(10):
    print(
        sample[i]["text"]
    )

    print(
        sample[i]["O"],
        sample[i]["C"],
        sample[i]["E"],
        sample[i]["A"],
        sample[i]["N"]
    )

    print("-" * 50)