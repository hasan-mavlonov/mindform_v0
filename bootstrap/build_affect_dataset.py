"""Build a (text -> appraisal) dataset for training the appraisal head.

Real supervision for valence / intensity / agency comes from EmoBank's
Valence-Arousal-Dominance annotations (dominance ~ sense of control/agency). The
remaining appraisal dims are filled with the heuristic extractor as weak labels;
swap in appraisal-annotated event corpora (e.g. the enVENT / ISEAR appraisal
datasets) to supervise outcome / novelty / threat properly.

None of these corpora are longitudinal. Run locally (needs `datasets`):
    python bootstrap/build_affect_dataset.py

NOTE: adjust the dataset id / column names below to match your chosen source.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from appraisal import _heuristic_appraise

OUT = "data/appraisal_dataset.jsonl"


def vad_to_appraisal(text, v, a, d):
    """EmoBank VAD (~1..5, centred at 3) -> appraisal, overriding heuristic dims."""
    appraisal = _heuristic_appraise(text)
    appraisal["valence"] = (v - 3.0) / 2.0       # -> [-1, 1]
    appraisal["intensity"] = (a - 1.0) / 4.0     # arousal -> [0, 1]
    appraisal["agency"] = (d - 3.0) / 2.0        # dominance ~ control -> [-1, 1]
    return appraisal


def main(out=OUT):
    ds = load_dataset("Blablablab/EmoBank", split="train")  # cols: text, V, A, D
    rows = []
    for r in ds:
        appraisal = vad_to_appraisal(
            r["text"], float(r["V"]), float(r["A"]), float(r["D"])
        )
        rows.append({"text": r["text"], "appraisal": appraisal})

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
