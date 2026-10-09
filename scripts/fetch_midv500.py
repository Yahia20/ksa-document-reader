"""Download the MIDV-500 passport clips and keep only the frames in the MRZ benchmark.

MIDV-500 (Arlazarov et al., 2019) is a public set of phone videos of specimen identity
documents. The curated labels (3,315 frames with a complete, visible MRZ) come from
Dynamsoft's open benchmark: github.com/yushulx/python-mrz-scanner-sdk
(examples/official/benchmark-midv500/labels.csv).

Each archive is ~650 MB. It is downloaded, the labelled frames are converted to JPG,
then the archive is deleted, so the disk never holds more than one archive.

    python scripts/fetch_midv500.py
"""
from __future__ import annotations

import csv
import ftplib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "midv500"
FTP_HOST, FTP_DIR = "smartengines.com", "midv-500/dataset"
LABELS_URL = ("https://raw.githubusercontent.com/yushulx/python-mrz-scanner-sdk/main/"
              "examples/official/benchmark-midv500/labels.csv")  # MIT licence
ARCHIVES = {
    "05": "05_aze_passport", "06": "06_bra_passport", "11": "11_cze_passport",
    "16": "16_deu_passport_new", "17": "17_deu_passport_old", "18": "18_dza_passport",
    "25": "25_grc_passport", "27": "27_hrv_passport", "28": "28_hun_passport",
    "32": "32_lva_passport", "34": "34_mda_passport", "41": "41_srb_passport",
}


def wanted_frames() -> dict[str, set[str]]:
    """Map document code -> set of frame stems, e.g. '05' -> {'CA05_01', ...}."""
    if not (ROOT / "labels.csv").exists():
        ROOT.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LABELS_URL, ROOT / "labels.csv")
    by_doc: dict[str, set[str]] = {}
    with open(ROOT / "labels.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            stem = PurePosixPath(row["imagepath"]).stem  # CA05_01
            by_doc.setdefault(stem[2:4], set()).add(stem)
    return by_doc


def download(name: str, dest: Path) -> None:
    """Fetch one archive. The server's control connection often times out during a
    ~7 minute transfer, so completion is judged by file size, not by its reply."""
    ftp = ftplib.FTP(FTP_HOST, timeout=120)
    ftp.login()
    ftp.cwd(FTP_DIR)
    ftp.voidcmd("TYPE I")
    expected = ftp.size(name)
    if dest.exists() and dest.stat().st_size == expected:
        return
    conn = ftp.transfercmd(f"RETR {name}")
    with open(dest, "wb") as fh:
        while chunk := conn.recv(1 << 16):
            fh.write(chunk)
    conn.close()
    try:
        ftp.voidresp()
        ftp.quit()
    except (TimeoutError, ftplib.Error, OSError):
        pass
    if dest.stat().st_size != expected:
        raise IOError(f"{name}: got {dest.stat().st_size} of {expected} bytes")


def extract(zip_path: Path, frames: set[str]) -> int:
    done = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            p = PurePosixPath(info.filename)
            if p.suffix.lower() not in (".tif", ".tiff") or p.stem not in frames:
                continue
            cond = p.stem[:2]  # CA, CS, HA ...
            out = ROOT / "images" / cond / f"{p.stem}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src:
                Image.open(io.BytesIO(src.read())).convert("RGB").save(out, quality=92)
            done += 1
    return done


def main() -> None:
    by_doc = wanted_frames()
    tmp = ROOT / "_download"
    tmp.mkdir(parents=True, exist_ok=True)
    for code, folder in ARCHIVES.items():
        frames = by_doc.get(code, set())
        have = {p.stem for p in (ROOT / "images").glob(f"*/*{code}_*.jpg")}
        if frames <= have:
            print(f"{folder}: already have {len(frames)} frames", flush=True)
            continue
        zip_path = tmp / f"{folder}.zip"
        print(f"{folder}: downloading", flush=True)
        for attempt in range(1, 4):
            try:
                download(f"{folder}.zip", zip_path)
                break
            except (OSError, ftplib.Error) as exc:
                print(f"{folder}: attempt {attempt} failed: {exc}", flush=True)
        else:
            raise SystemExit(f"{folder}: giving up after 3 attempts")
        n = extract(zip_path, frames)
        zip_path.unlink()
        print(f"{folder}: kept {n}/{len(frames)} frames", flush=True)
    tmp.rmdir()


if __name__ == "__main__":
    sys.exit(main())
