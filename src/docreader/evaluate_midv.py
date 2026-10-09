"""Score MRZ reading on real phone video frames: the curated MIDV-500 benchmark.

    python scripts/fetch_midv500.py              # ~8 GB download, keeps ~1 GB of frames
    python -m docreader.evaluate_midv          # resumes if interrupted

3,315 frames of 12 passport specimens, filmed on a table, in hand, on a keyboard,
on clutter, and partly out of frame. Ground truth and the commercial baseline come
from Dynamsoft's open benchmark (github.com/yushulx/python-mrz-scanner-sdk):

    Dynamsoft Capture Vision  48.33% exact MRZ   (commercial SDK)
    FastMRZ                    7.30%            (open source, Tesseract)
    PassportEye                0.27%            (open source, Tesseract)

Exact match is the strict, comparable number. What matters in production is the
next one: of the frames the reader calls proven, how many are actually right.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path, PurePosixPath

from . import mrz
from .evaluate import keep_awake

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "midv500"
BASELINES = {"Dynamsoft Capture Vision (commercial)": 0.4833, "FastMRZ (open source)": 0.0730,
             "PassportEye (open source)": 0.0027}
CONDITIONS = {"T": "table", "K": "keyboard", "H": "in hand", "C": "clutter", "P": "partly out of frame"}
FIELDS = ["document_number", "surname", "given_names", "birth_date", "expiry_date", "nationality", "sex"]


def resummarise() -> dict:
    """Recompute metrics.json from predictions.jsonl without reading the frames again."""
    out_dir = ROOT / "results" / "midv500"
    preds = [json.loads(l) for l in open(out_dir / "predictions.jsonl", encoding="utf-8")]
    metrics = summarise(preds, {p["frame"]: p["truth"] for p in preds})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run(limit: int | None, fresh: bool) -> None:
    """Read every labelled frame. Each result is appended to partial.jsonl as soon as it
    exists, so a run that is interrupted (sleep, shutdown) resumes where it stopped."""
    labels = []
    with open(DATA / "labels.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            stem = PurePosixPath(row["imagepath"]).stem
            img = DATA / "images" / stem[:2] / f"{stem}.jpg"
            if img.exists():
                labels.append((stem, str(img), [l for l in row["label"].split("#") if l]))
    if limit:
        labels = labels[::max(1, len(labels) // limit)][:limit]
    truth = {stem: lines for stem, _, lines in labels}
    out_dir = ROOT / "results" / "midv500"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "partial.jsonl"
    if fresh:
        partial.unlink(missing_ok=True)
    done = {}
    if partial.exists():
        for line in open(partial, encoding="utf-8"):
            try:
                rec = json.loads(line)
                done[rec["frame"]] = rec
            except json.JSONDecodeError:
                pass  # a line cut off by the interruption
    # Every third frame first: after the first third the run is already an even sample
    # across documents and filming conditions; the rest then completes the full set.
    order = labels[0::3] + labels[1::3] + labels[2::3]
    todo = [(s, p) for s, p, _ in order if s not in done]
    print(f"{len(done)} frames already read, {len(todo)} to go", flush=True)

    keep_awake()
    from .pipeline import DocumentReader
    reader = DocumentReader()
    with open(partial, "a", encoding="utf-8") as fh:
        for i, (frame, path) in enumerate(todo, 1):
            res = reader.read(path).to_dict()
            rec = {"frame": frame, "mrz_lines": res["mrz_lines"], "mrz_proven": res["mrz_proven"],
                   "doc_type": res["doc_type"], "fields": res["fields"], "seconds": res["seconds"]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done[frame] = rec
            if i % 100 == 0:
                print(f"{len(done)}/{len(labels)}", flush=True)

    preds = [done[s] for s, _, _ in labels]
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps({**p, "truth": truth[p["frame"]]}, ensure_ascii=False) + "\n")
    metrics = summarise(preds, truth)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    partial.unlink()
    print(json.dumps({k: v for k, v in metrics.items() if k != "by_document"}, indent=2))


def _truth_fields(lines: list[str]) -> dict[str, str]:
    res = mrz.parse(lines)
    return {n: res.fields[n].value for n in FIELDS}


def summarise(preds: list[dict], truth: dict[str, list[str]]) -> dict:
    n = len(preds)
    exact = [p for p in preds if p["mrz_lines"] == truth[p["frame"]]]
    proven = [p for p in preds if p["mrz_proven"]]
    proven_right = [p for p in proven if p["mrz_lines"] == truth[p["frame"]]]
    core = ("document_number", "surname", "given_names", "birth_date", "expiry_date")
    proven_data_right = [p for p in proven if all(
        str(p["fields"][n]["value"]) == str(_truth_fields(truth[p["frame"]])[n]) for n in core)]

    ver = ver_ok = 0
    per_field = defaultdict(lambda: [0, 0, 0])  # verified, verified correct, total
    for p in preds:
        want = _truth_fields(truth[p["frame"]])
        for name, value in want.items():
            f = p["fields"].get(name)
            per_field[name][2] += 1
            if f and f["status"] == "verified":
                ok = str(f["value"]) == str(value)
                ver += 1
                ver_ok += ok
                per_field[name][0] += 1
                per_field[name][1] += ok

    def group(key) -> dict:
        g = defaultdict(list)
        for p in preds:
            g[key(p)].append(p)
        return {k: {"frames": len(v),
                    "exact": round(sum(x["mrz_lines"] == truth[x["frame"]] for x in v) / len(v), 4),
                    "proven": round(sum(x["mrz_proven"] for x in v) / len(v), 4)}
                for k, v in sorted(g.items())}

    return {
        "frames": n,
        "exact_mrz_rate": round(len(exact) / n, 4),
        "baselines_exact_mrz_rate": BASELINES,
        "proven_rate": round(len(proven) / n, 4),
        "proven_precision": round(len(proven_right) / len(proven), 4) if proven else None,
        "proven_wrong": len(proven) - len(proven_right),
        # Proven frames whose five core fields are right; the rest of the 88 characters
        # (filler, optional data) may still differ from the label.
        "proven_data_precision": round(len(proven_data_right) / len(proven), 4) if proven else None,
        "proven_data_wrong": len(proven) - len(proven_data_right),
        "verified_fields": ver,
        "verified_field_precision": round(ver_ok / ver, 4) if ver else None,
        "by_field": {k: {"verified_rate": round(v[0] / v[2], 4), "verified_precision":
                         round(v[1] / v[0], 4) if v[0] else None} for k, v in per_field.items()},
        "by_condition": group(lambda p: CONDITIONS[p["frame"][0]]),
        "by_document": group(lambda p: p["frame"][2:4]),
        "seconds_per_frame": round(sum(p["seconds"] for p in preds) / n, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fresh", action="store_true", help="ignore frames read by an interrupted run")
    ap.add_argument("--summarise-only", action="store_true", help="recompute metrics from predictions.jsonl")
    a = ap.parse_args()
    if a.summarise_only:
        resummarise()
    else:
        run(a.limit, a.fresh)


if __name__ == "__main__":
    main()
