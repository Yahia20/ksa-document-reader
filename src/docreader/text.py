"""Text helpers: Arabic normalisation, amounts, dates, identifiers."""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

_DIACRITICS = re.compile(r"[ً-ْٰـ]")  # harakat, dagger alef, tatweel
_EASTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def norm_ar(s: str) -> str:
    """Fold the Arabic spelling variants OCR and typists mix up."""
    s = _DIACRITICS.sub("", s)
    s = re.sub("[إأآٱ]", "ا", s)
    return s.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")


def squash(s: str) -> str:
    """Lower-case, Arabic-normalised, with spaces and punctuation removed."""
    return re.sub(r"[\s\W_]+", "", norm_ar(s).lower())


def western_digits(s: str) -> str:
    return s.translate(_EASTERN)


AMOUNT = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+)[.٫](\d{2})(?!\d)")
INTEGER = re.compile(r"(?<![\d.,/:-])(\d{1,3})(?![\d.,/:])")
VAT_NUMBER = re.compile(r"(?<!\d)3\d{13}3(?!\d)")
INVOICE_NO = re.compile(r"(?<![A-Z0-9])([A-Z]{1,4}[-/]?(?:\d{4}[-/]?\d{3,6}|\d{6}))(?![0-9])")
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def amounts(s: str) -> list[tuple[int, Decimal]]:
    """Money values with their position in the string."""
    s = western_digits(s)
    return [(m.start(), Decimal(m.group(1).replace(",", "") + "." + m.group(2))) for m in AMOUNT.finditer(s)]


def integers(s: str) -> list[tuple[int, int]]:
    s = western_digits(s)
    return [(m.start(), int(m.group(1))) for m in INTEGER.finditer(s)]


def find_dates(s: str) -> list[date]:
    s = western_digits(s)
    out: list[date] = []
    pats = [
        (r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", (1, 2, 3)),
        (r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", (3, 2, 1)),
    ]
    for pat, (yi, mi, di) in pats:
        for m in re.finditer(pat, s):
            try:
                out.append((m.start(), date(int(m.group(yi)), int(m.group(mi)), int(m.group(di)))))
            except ValueError:
                pass
    for m in re.finditer(r"(?<!\d)(\d{1,2})\s?([A-Za-z]{3})[a-z]*\.?\s?(\d{4})", s):
        mon = MONTHS.get(m.group(2).lower())
        if mon:
            try:
                out.append((m.start(), date(int(m.group(3)), mon, int(m.group(1)))))
            except ValueError:
                pass
    return [d for _, d in sorted(out, key=lambda t: t[0])]
