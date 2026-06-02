from datasets import load_dataset

ds = load_dataset(
    "jingjietan/pandora-big5"
)

train = ds["train"]

for trait in ["O", "C", "E", "A", "N"]:

    unique_values = len(
        set(
            train.select(range(10000))[trait]
        )
    )

    print(
        trait,
        "unique values:",
        unique_values
    )