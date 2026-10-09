"""Score the reader on a labelled split.

    python -m docreader.evaluate --split dev
    python -m docreader.evaluate --split test --workers 4

For every field in the ground truth it records the status the reader gave and
whether the value was right, then reports:

    verified precision   share of "verified" fields that are correct (the bar: 97%+)
    automation rate      share of all fields that came back "verified"
    read accuracy        how often unproven "read" fields were right anyway
    document type        accuracy over the 10 classes
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .text import norm_ar

ROOT = Path(__file__).resolve().parents[2]
SKIP = {"mrz", "qr_phase", "line_count", "invoice_type"}


def norm(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (int,)):
        return str(v)
    s = str(v).strip()
    try:
        return f"{Decimal(s.replace(',', '')):.2f}" if s.replace(",", "").replace(".", "", 1).isdigit() and "." in s else \
            " ".join(norm_ar(s).upper().split())
    except InvalidOperation:
        return " ".join(norm_ar(s).upper().split())


def lines_key(items) -> list[tuple[str, str, str]]:
    return sorted((str(int(i["qty"])), f"{Decimal(str(i['unit'])):.2f}", f"{Decimal(str(i['net'])):.2f}")
                  for i in items or [])


def score_doc(truth: dict, doc_type: str, pred: dict) -> list[dict]:
    out = []
    fields = pred["fields"]
    for name, want in truth.items():
        if name in SKIP:
            continue
        key = "line_items" if name == "lines" else name
        got = fields.get(key)
        status = got["status"] if got else "missing"
        value = got["value"] if got else None
        if name == "lines":
            ok = value is not None and lines_key(value) == lines_key(want)
        elif name == "name_ar":
            ok = value is not None and norm(value).replace(" ", "") == norm(want).replace(" ", "")
        else:
            ok = value is not None and norm(value) == norm(want)
        out.append({"field": key, "status": status, "correct": bool(ok),
                    "got": value if name != "lines" else len(value or []), "want": want if name != "lines" else len(want)})
    return out


def keep_awake() -> None:
    """Ask Windows not to sleep while frames are being read (no-op elsewhere)."""
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


_reader = None


def _init(threads: int) -> None:
    global _reader
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    from .pipeline import DocumentReader
    _reader = DocumentReader(threads=threads)


def _work(item: tuple[str, str]) -> dict:
    doc_id, path = item
    return {"id": doc_id, **_reader.read(path).to_dict()}


def run(split: str, workers: int, limit: int | None) -> None:
    data_dir = ROOT / "data" / "synthetic" / split
    labels = [json.loads(l) for l in open(data_dir / "labels.jsonl", encoding="utf-8")]
    if limit:
        labels = labels[::max(1, len(labels) // limit)][:limit]
    out_dir = ROOT / "results" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    threads = max(1, (os.cpu_count() or 4) // workers)
    keep_awake()
    jobs = [(l["id"], str(data_dir / f"{l['id']}.jpg")) for l in labels]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(threads,)) as ex:
        preds = {p["id"]: p for p in ex.map(_work, jobs, chunksize=2)}

    rows = []
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for l in labels:
            p = preds[l["id"]]
            fields = score_doc(l["truth"], l["doc_type"], p)
            rec = {"id": l["id"], "doc_type": l["doc_type"], "pred_type": p["doc_type"],
                   "type_basis": p["type_basis"], "level": l["level"], "heldout": l["heldout_template"],
                   "template": l["template"], "seconds": p["seconds"], "fields": fields,
                   "mrz_lines": p["mrz_lines"], "pred_fields": p["fields"]}
            rows.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    metrics = summarise(rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report(metrics)


def _agg(fields: list[dict]) -> dict:
    c = Counter((f["status"], f["correct"]) for f in fields)
    ver = c[("verified", True)] + c[("verified", False)]
    read = c[("read", True)] + c[("read", False)]
    n = len(fields)
    return {
        "fields": n,
        "verified": ver, "verified_correct": c[("verified", True)],
        "verified_precision": round(c[("verified", True)] / ver, 4) if ver else None,
        "automation_rate": round(ver / n, 4) if n else None,
        "read": read, "read_accuracy": round(c[("read", True)] / read, 4) if read else None,
        "review": n - ver - read,
    }


def summarise(rows: list[dict]) -> dict:
    allf = [dict(f, doc_type=r["doc_type"], level=r["level"], heldout=r["heldout"]) for r in rows for f in r["fields"]]
    by = lambda key: {k: _agg([f for f in allf if f[key] == k]) for k in sorted({f[key] for f in allf}, key=str)}
    by_field = defaultdict(list)
    for f in allf:
        by_field[f"{f['doc_type']}.{f['field']}"].append(f)
    types = Counter((r["doc_type"], r["pred_type"]) for r in rows)
    docs_ok = sum(r["doc_type"] == r["pred_type"] for r in rows)
    straight = [r for r in rows if r["fields"] and all(f["status"] == "verified" for f in r["fields"])]
    return {
        "documents": len(rows),
        "doc_type_accuracy": round(docs_ok / len(rows), 4),
        "confusion": [{"true": t, "pred": p, "n": n} for (t, p), n in sorted(types.items())],
        "overall": _agg(allf),
        "by_doc_type": by("doc_type"),
        "by_level": by("level"),
        "by_heldout_template": by("heldout"),
        "by_field": {k: _agg(v) for k, v in sorted(by_field.items())},
        "straight_through_docs": len(straight),
        "straight_through_correct": sum(all(f["correct"] for f in r["fields"]) for r in straight),
        "seconds_per_doc": round(sum(r["seconds"] for r in rows) / len(rows), 2),
    }


def print_report(m: dict) -> None:
    o = m["overall"]
    print(f"documents {m['documents']}  type accuracy {m['doc_type_accuracy']:.1%}  "
          f"{m['seconds_per_doc']} s/doc")
    print(f"fields {o['fields']}  verified {o['verified']} precision {o['verified_precision']}  "
          f"automation {o['automation_rate']}  read acc {o['read_accuracy']}  review {o['review']}")
    for group in ("by_doc_type", "by_level", "by_heldout_template"):
        print(f"-- {group}")
        for k, a in m[group].items():
            print(f"   {str(k):24s} n={a['fields']:4d} ver_prec={a['verified_precision']} auto={a['automation_rate']} "
                  f"read_acc={a['read_accuracy']} review={a['review']}")
    bad = [c for c in m["confusion"] if c["true"] != c["pred"]]
    if bad:
        print("-- misclassified", bad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run(a.split, a.workers, a.limit)


if __name__ == "__main__":
    main()
