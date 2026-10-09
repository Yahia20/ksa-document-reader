"""Generate the synthetic dev and test sets.

    python -m synth.make_dataset            # dev (20 per type) + test (60 per type)

Writes data/synthetic/<split>/<id>.jpg and data/synthetic/<split>/labels.jsonl.
The dev split only uses the development templates. The test split uses different
random seeds and gives one third of each family to the held-out templates.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

from .degrade import degrade
from .documents import DOC_TYPES, TEMPLATES, build
from .fake import Faketory
from .render import Renderer

ROOT = Path(__file__).resolve().parents[1] / "data" / "synthetic"
SPLITS = {"dev": (20, 1000), "test": (60, 2000)}  # docs per type, seed
LEVEL_MIX = (("scan", 0.30), ("photo", 0.45), ("hard", 0.25))


def levels_for(n: int) -> list[str]:
    out: list[str] = []
    for name, share in LEVEL_MIX:
        out += [name] * round(n * share)
    return (out + ["photo"] * n)[:n]


def make_split(split: str, renderer: Renderer) -> None:
    per_type, seed = SPLITS[split]
    out_dir = ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    fk = Faketory(seed)
    r = random.Random(seed)
    clean = out_dir / "_clean.png"
    with open(out_dir / "labels.jsonl", "w", encoding="utf-8") as labels:
        for doc_type in DOC_TYPES:
            dev_t, held_t = TEMPLATES[doc_type]
            levels = levels_for(per_type)
            r.shuffle(levels)
            for i in range(per_type):
                heldout = split == "test" and bool(held_t) and i % 3 == 2
                template = r.choice(held_t if heldout else dev_t)
                doc = build(doc_type, template, fk)
                renderer.render(template, doc.context, clean)
                img = degrade(cv2.imread(str(clean)), levels[i], seed * 7919 + DOC_TYPES.index(doc_type) * 1000 + i)
                doc_id = f"{split}_{doc_type}_{i:03d}"
                cv2.imwrite(str(out_dir / f"{doc_id}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                labels.write(json.dumps({
                    "id": doc_id, "doc_type": doc_type, "template": template, "level": levels[i],
                    "heldout_template": heldout, "truth": doc.truth}, ensure_ascii=False) + "\n")
            print(f"{split}: {doc_type} done", flush=True)
    clean.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), action="append")
    args = ap.parse_args()
    with Renderer() as renderer:
        for split in args.split or list(SPLITS):
            make_split(split, renderer)


if __name__ == "__main__":
    main()
