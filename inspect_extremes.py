# inspect_extremes.py

from datasets import load_dataset

ds = load_dataset(
    "jingjietan/pandora-big5"
)

train = ds["train"]

high_extroverts = []
low_extroverts = []

for sample in train:

    e = sample["E"]

    if e >= 90 and len(high_extroverts) < 30:
        high_extroverts.append(
            (
                e,
                sample["text"]
            )
        )

    if e <= 10 and len(low_extroverts) < 30:
        low_extroverts.append(
            (
                e,
                sample["text"]
            )
        )

    if (
        len(high_extroverts) >= 30
        and
        len(low_extroverts) >= 30
    ):
        break


print("\n" + "=" * 80)
print("HIGH EXTRAVERSION")
print("=" * 80)

for e, text in high_extroverts:

    print(f"\nE={e}")
    print(text)
    print("-" * 60)


print("\n" + "=" * 80)
print("LOW EXTRAVERSION")
print("=" * 80)

for e, text in low_extroverts:

    print(f"\nE={e}")
    print(text)
    print("-" * 60)