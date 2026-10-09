"""OCR with RapidOCR (ONNX builds of PaddleOCR).

    detection          PP-OCRv6 small    finds text lines in any script, once per page
    Latin recognition  PP-OCRv6 small    reads every line: numbers, codes, English
    Arabic recognition PP-OCRv5 Arabic   reads lines the Latin model is unsure of,
                                         and every MRZ-looking line

PP-OCRv6 (June 2026) is the stronger recogniser but does not cover Arabic yet. Each
line therefore keeps two readings: numbers and codes come from the Latin one, Arabic
words from the Arabic one, and on the MRZ the check digits pick between them.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy as np


ARABIC = re.compile(r"[؀-ۿ]")


@dataclass
class TextLine:
    latin: str
    latin_score: float
    box: np.ndarray  # 4 x 2, clockwise from top-left
    arabic: str = ""
    arabic_score: float = 0.0

    @property
    def text(self) -> str:
        """The more confident reading."""
        if self.arabic and (self.arabic_score > self.latin_score or not self.latin.strip()):
            return self.arabic
        return self.latin

    @property
    def readings(self) -> list[str]:
        return [t for t in (self.latin, self.arabic) if t.strip()]

    @property
    def cx(self) -> float:
        return float(self.box[:, 0].mean())

    @property
    def cy(self) -> float:
        return float(self.box[:, 1].mean())

    @property
    def height(self) -> float:
        return float(np.linalg.norm(self.box[3] - self.box[0]))


@dataclass
class Row:
    lines: list[TextLine]

    @property
    def text(self) -> str:
        return " ".join(l.text for l in self.lines)

    @property
    def latin(self) -> str:
        return " ".join(l.latin for l in self.lines)

    @property
    def arabic(self) -> str:
        return " ".join(l.arabic for l in self.lines if l.arabic)

    @property
    def arabic_or_latin(self) -> str:
        return " ".join(l.arabic or l.latin for l in self.lines)


class OCR:
    def __init__(self, threads: int = -1):
        self.threads = threads

    def _params(self, params: dict) -> dict:
        return {"Global.log_level": "warning", "Global.use_cls": False,
                "EngineConfig.onnxruntime.intra_op_num_threads": self.threads, **params}
    @cached_property
    def _page(self):
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
        return RapidOCR(params=self._params({
            "Det.ocr_version": OCRVersion.PPOCRV6, "Det.model_type": ModelType.SMALL, "Det.lang_type": "en",
            "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.MOBILE,
            "Rec.lang_type": LangRec.ARABIC,
        }))

    @cached_property
    def _latin(self):
        from rapidocr import ModelType, OCRVersion, RapidOCR
        return RapidOCR(params=self._params({
            "Det.ocr_version": OCRVersion.PPOCRV6, "Det.model_type": ModelType.SMALL, "Det.lang_type": "en",
            "Rec.ocr_version": OCRVersion.PPOCRV6, "Rec.model_type": ModelType.SMALL, "Rec.lang_type": "en",
        }))

    def detect(self, img: np.ndarray) -> np.ndarray:
        det = self._latin(img, use_rec=False)
        return np.zeros((0, 4, 2), np.float32) if det.boxes is None else det.boxes.astype(np.float32)

    def _recognize(self, engine, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        from rapidocr.ch_ppocr_rec.typings import TextRecInput
        if not crops:
            return []
        res = engine.text_rec(TextRecInput(img=crops))
        return list(zip(res.txts, (float(x) for x in res.scores)))

    def read(self, img: np.ndarray, arabic: str = "auto") -> list[TextLine]:
        """Detect lines once, read them all with the Latin model, and read with the
        Arabic model too where needed ("auto"), on every line ("all") or never ("none")."""
        boxes = self.detect(img)
        crops = [crop_line(img, b, pad=0.08) for b in boxes]
        lines = [TextLine(t, s, b) for (t, s), b in zip(self._recognize(self._latin, crops), boxes)]
        if arabic == "none":
            return lines
        idx = [i for i, l in enumerate(lines)
               if arabic == "all" or l.latin_score < 0.93 or _mrz_like(l.latin)]
        for i, (t, s) in zip(idx, self._recognize(self._page, [crops[i] for i in idx])):
            lines[i].arabic, lines[i].arabic_score = t, s
        return lines


def _mrz_like(text: str) -> bool:
    t = text.replace(" ", "")
    return len(t) >= 24 and (t.count("<") >= 2 or sum(c.isdigit() for c in t) >= 12)


def crop_line(img: np.ndarray, box: np.ndarray, pad: float = 0.12) -> np.ndarray:
    """Cut a text line out of the page and straighten it."""
    tl, tr, br, bl = box.astype(np.float32)
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    # Grow the quadrilateral a little so the first and last characters are kept.
    c = box.mean(axis=0)
    grown = c + (box - c) * np.float32([1 + pad * h / max(w, 1), 1 + 2 * pad])
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(grown.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (int(w), int(h)), borderMode=cv2.BORDER_REPLICATE)


def page_angle(lines: list[TextLine]) -> float:
    """Dominant text angle in degrees, weighted by line length."""
    num = den = 0.0
    for l in lines:
        d = l.box[1] - l.box[0]
        length = float(np.hypot(*d))
        if length < 40:
            continue
        a = math.atan2(float(d[1]), float(d[0]))
        num += a * length
        den += length
    return math.degrees(num / den) if den else 0.0


def group_rows(lines: list[TextLine]) -> list[Row]:
    """Group text lines into visual rows, after undoing the page rotation, and
    order each row left to right."""
    if not lines:
        return []
    a = -math.radians(page_angle(lines))
    ca, sa = math.cos(a), math.sin(a)
    items = []
    for l in lines:
        x, y = l.cx, l.cy
        items.append((x * sa + y * ca, x * ca - y * sa, l))  # (rotated y, rotated x, line)
    items.sort(key=lambda t: t[0])
    rows: list[list[tuple]] = []
    for it in items:
        if rows:
            last = rows[-1]
            ref_y = sum(t[0] for t in last) / len(last)
            ref_h = sum(t[2].height for t in last) / len(last)
            if abs(it[0] - ref_y) < 0.5 * max(ref_h, it[2].height):
                last.append(it)
                continue
        rows.append([it])
    return [Row([t[2] for t in sorted(r, key=lambda t: t[1])]) for r in rows]
