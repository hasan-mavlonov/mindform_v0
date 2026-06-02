from datasets import load_dataset

ds = load_dataset(
    "jingjietan/pandora-big5",
    split="train"
)

for ptype in [0, 1, 2, 3, 4]:

    rows = []

    for row in ds:

        if row["ptype"] == ptype:
            rows.append(row)

        if len(rows) >= 100:
            break

    print("\nPTYPE", ptype)

    print(
        "O:",
        sum(r["O"] for r in rows) / len(rows)
    )

    print(
        "C:",
        sum(r["C"] for r in rows) / len(rows)
    )

    print(
        "E:",
        sum(r["E"] for r in rows) / len(rows)
    )

    print(
        "A:",
        sum(r["A"] for r in rows) / len(rows)
    )

    print(
        "N:",
        sum(r["N"] for r in rows) / len(rows)
    )