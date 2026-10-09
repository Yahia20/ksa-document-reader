"""Read one document image: classify it, extract its fields, and prove what can be proven.

Every field comes back with a status:

    verified  proven independently: an MRZ check digit, an error-corrected barcode,
              the invoice arithmetic, or two separate reads that agree
    read      read once and plausible, but nothing independent confirms it
    review    a check failed, two readings disagree, or the field was not found

Only "verified" fields are meant to flow into another system without a person
looking at them.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

from . import bcbp, mrz, zatca
from .geometry import rectify
from .ocr import OCR, Row, group_rows
from .text import (INVOICE_NO, VAT_NUMBER, amounts, find_dates, integers, norm_ar, squash)

DOC_TYPES = ["passport", "id_card_td1", "id_card_td2", "visa", "tax_invoice",
             "simplified_tax_invoice", "credit_note", "debit_note", "boarding_pass", "other"]
INVOICE_TYPES = DOC_TYPES[4:8]

# Titles, checked in this order (a credit note also says "invoice" somewhere).
INVOICE_TITLES = [(t, [squash(k) for k in keys]) for t, keys in [
    ("credit_note", ["إشعار دائن", "credit note"]),
    ("debit_note", ["إشعار مدين", "debit note"]),
    ("simplified_tax_invoice", ["فاتورة ضريبية مبسطة", "simplified tax invoice"]),
    ("tax_invoice", ["فاتورة ضريبية", "tax invoice"]),
]]
ID_TITLES = [(t, [squash(k) for k in keys]) for t, keys in [
    ("boarding_pass", ["boarding pass", "بطاقة صعود"]),
    ("visa", ["visa", "تأشيرة"]),
    ("passport", ["passport", "جواز سفر"]),
    ("id_card_td1", ["identity card", "بطاقة الهوية"]),
    ("id_card_td2", ["travel document", "وثيقة سفر"]),
]]
REFERENCE_WORDS = [squash(w) for w in ["original", "reference", "ref", "الأصلية", "المرجع"]]
# Words printed as labels on ID pages; whatever Arabic is left is likely the name.
ID_LABEL_WORDS = set(norm_ar(w) for w in (
    "يوتوبيا جواز سفر السفر النوع الرمز رقم الجواز اللقب الاسم الكامل الأسماء الجنسية تاريخ الميلاد "
    "الجنس محل الإصدار الانتهاء بطاقة الهوية وثيقة تأشيرة ملاحظات جهة وزارة الداخلية بالعربية").split())


@dataclass
class Field:
    value: object
    status: str  # verified | read | review
    source: str  # mrz | qr | barcode | ocr | arithmetic | ...
    note: str = ""


@dataclass
class Result:
    doc_type: str
    type_basis: str
    fields: dict[str, Field] = field(default_factory=dict)
    seconds: float = 0.0
    mrz_lines: list[str] = field(default_factory=list)
    mrz_proven: bool = False  # every check passes and no MRZ field is left for review

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fields"] = {k: {**asdict(v), "value": _jsonable(v.value)} for k, v in self.fields.items()}
        return d

    @property
    def needs_review(self) -> list[str]:
        return [k for k, f in self.fields.items() if f.status != "verified"]


def _jsonable(v):
    if isinstance(v, Decimal):
        return f"{v:.2f}"
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


# ---------------------------------------------------------------------------
# Barcodes
# ---------------------------------------------------------------------------
def read_barcodes(img: np.ndarray) -> list[zxingcpp.Barcode]:
    """Try the page as is, then a sharpened, enlarged grey copy for small or soft codes."""
    found = zxingcpp.read_barcodes(img)
    if found:
        return found
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(grey, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    sharp = cv2.addWeighted(big, 1.8, cv2.GaussianBlur(big, (0, 0), 2.0), -0.8, 0)
    return zxingcpp.read_barcodes(sharp)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------
class DocumentReader:
    def __init__(self, threads: int = -1):
        self.ocr = OCR(threads)

    def read(self, image: str | Path | np.ndarray, arabic: str = "auto") -> Result:
        t0 = time.perf_counter()
        img = cv2.imread(str(image)) if not isinstance(image, np.ndarray) else image
        if img is None:
            raise ValueError(f"cannot read image: {image}")

        # Barcodes: the photo as taken, then the flattened page if nothing decoded.
        flat, flattened = rectify(img)
        codes = read_barcodes(img) or (read_barcodes(flat) if flattened else [])
        qr = next((q for q in (_try(zatca.parse_qr, c.text) for c in codes) if q), None)
        bp = next((b for b in (_try(bcbp.parse, c.text) for c in codes) if b), None)

        rows, parsed, page = self._ocr_pass(img, arabic)
        # Second look on the flattened page when the MRZ is not fully proven, or when
        # nothing at all was recognised.
        weak_mrz = parsed is not None and not parsed.all_checks_pass
        nothing = parsed is None and qr is None and bp is None and not _has_title(page)
        if flattened and (weak_mrz or nothing):
            rows2, parsed2, page2 = self._ocr_pass(flat, arabic)
            if _mrz_rank(parsed2) > _mrz_rank(parsed) or (nothing and _has_title(page2)):
                rows, parsed, page = rows2, parsed2, page2

        doc_type, basis = classify(parsed, bp, qr, page)
        if doc_type in ("passport", "id_card_td1", "id_card_td2", "visa"):
            result = Result(doc_type, basis, _id_fields(parsed, rows))
        elif doc_type in INVOICE_TYPES:
            result = Result(doc_type, basis, _invoice_fields(doc_type, qr, rows))
        elif doc_type == "boarding_pass":
            result = Result(doc_type, basis, _boarding_fields(bp, rows))
        else:
            result = Result("other", basis)
        if parsed is not None:
            result.mrz_lines = parsed.lines
            # Usable as is: every MRZ check passes, the five core fields are verified and
            # nothing else waits for review.
            core = ("document_number", "surname", "given_names", "birth_date", "expiry_date")
            result.mrz_proven = (parsed.all_checks_pass and doc_type in MRZ_TYPES.values()
                                 and all(n in result.fields and result.fields[n].status == "verified" for n in core)
                                 and all(f.status != "review" for f in result.fields.values()))
        result.seconds = round(time.perf_counter() - t0, 2)
        return result


    def _ocr_pass(self, img: np.ndarray, arabic: str):
        rows = group_rows(self.ocr.read(img, arabic=arabic))
        parsed = mrz.find_mrz([[r.latin, r.arabic_or_latin] for r in rows])
        page = squash(" ".join(r.latin + " " + r.arabic for r in rows))
        return rows, parsed, page


def _mrz_rank(parsed: mrz.MRZResult | None) -> tuple:
    if parsed is None:
        return (-1,)
    return (parsed.composite_ok is True, sum(f.status == "verified" for f in parsed.fields.values()))


def _has_title(page: str) -> bool:
    return any(k in page for _, keys in INVOICE_TITLES + ID_TITLES for k in keys)


def _try(fn, arg):
    try:
        return fn(arg)
    except (ValueError, TypeError, UnicodeDecodeError):
        return None


MRZ_TYPES = {"TD3": "passport", "TD1": "id_card_td1", "TD2": "id_card_td2", "MRV-A": "visa", "MRV-B": "visa"}


def classify(parsed, bp, qr, page: str) -> tuple[str, str]:
    """Standards first (MRZ, boarding-pass barcode, ZATCA QR), printed titles second.
    An MRZ that does not check out as a whole only counts when nothing else fits."""
    if parsed is not None and parsed.whole_ok:
        return MRZ_TYPES[parsed.layout], f"MRZ ({parsed.layout})"
    if bp is not None:
        return "boarding_pass", "IATA boarding-pass barcode"
    for doc_type, keys in INVOICE_TITLES:
        if any(k in page for k in keys):
            return doc_type, "printed title" + (" + ZATCA QR" if qr else "")
    if qr is not None:
        # Only simplified (B2C) invoices carry ZATCA's stamp in tag 9.
        return ("simplified_tax_invoice" if 9 in qr.raw_tags else "tax_invoice"), "ZATCA QR only"
    for doc_type, keys in ID_TITLES:
        if any(k in page for k in keys):
            if parsed is not None and MRZ_TYPES[parsed.layout] == doc_type:
                return doc_type, f"printed title + partial MRZ ({parsed.layout})"
            return doc_type, "printed title only"
    if parsed is not None:
        return MRZ_TYPES[parsed.layout], f"partial MRZ ({parsed.layout})"
    return "other", "no supported standard or title found"


# ---------------------------------------------------------------------------
# Identity documents
# ---------------------------------------------------------------------------
def _id_fields(parsed: mrz.MRZResult | None, rows: list[Row]) -> dict[str, Field]:
    names = ["document_number", "surname", "given_names", "nationality", "birth_date", "sex",
             "expiry_date", "issuing_state"]
    if parsed is None:
        return {n: Field(None, "review", "mrz", "MRZ not found") for n in names}
    mrz_set = set(parsed.lines)
    viz_rows = [r for r in rows if mrz.clean_line(r.latin) not in mrz_set and not mrz.looks_like_mrz(r.latin)]
    viz_tokens = set()
    for r in viz_rows:
        viz_tokens.update(t for t in "".join(c if c.isalnum() else " " for c in r.latin.upper()).split())

    out: dict[str, Field] = {}
    for n in names:
        f = parsed.fields.get(n)
        if f is None:
            continue
        value, status, note = f.value, f.status, f.reason
        if n in ("surname", "given_names"):
            # The printed name and the MRZ name are two separate reads of the same text.
            for cand in [f.value, *f.alternatives]:
                if cand and all(len(tok) >= 2 and tok in viz_tokens for tok in cand.split()):
                    value, status, note = cand, "verified", "MRZ name matches the printed name"
                    break
            else:
                status = "review" if f.status == "review" else "read"
                note = "printed name not matched" if f.status != "review" else f.reason
        elif n in ("nationality", "issuing_state") and f.status == "read" and len(f.value) == 3 \
                and f.value in viz_tokens and parsed.whole_ok:
            status, note = "verified", "MRZ code matches the printed code"
        elif n == "document_number" and f.reason != "check digit failed":
            # The check digit alone is not proof for a document number: it is blind to
            # G/6, S/8, L/1-type swaps, and on real photos two misreads (BD -> 80) can
            # cancel out. The composite digit adds nothing here (same weight offset).
            # So the number must also be printed on the page: exactly one candidate that
            # passes the check digit and matches the printed text.
            printed = [c for c in dict.fromkeys([f.value, *f.alternatives])
                       if c and c in viz_tokens and mrz.passes_field_check(parsed, n, c)]
            if len(printed) == 1 and parsed.whole_ok:
                value, status, note = printed[0], "verified", "check digit + matches the printed number"
            elif f.status == "verified":
                status, note = "read", "check digit passed, but the printed number was not found to confirm it"
        out[n] = Field(value, status, "mrz", note)

    name_ar = _arabic_name(viz_rows)
    if name_ar:
        out["name_ar"] = Field(name_ar, "read", "ocr", "Arabic name from the printed page")
    return out


def _arabic_name(rows: list[Row]) -> str:
    best = ""
    for r in rows:
        for l in r.lines:
            txt = norm_ar(l.arabic).strip()
            words = [w for w in txt.split() if w]
            if not 2 <= len(words) <= 5 or any(w in ID_LABEL_WORDS for w in words):
                continue
            if not all(any("ء" <= c <= "ي" for c in w) for w in words):
                continue
            if len(txt) > len(best):
                best = l.arabic.strip()
    return best


# ---------------------------------------------------------------------------
# Invoices
# ---------------------------------------------------------------------------
def _invoice_fields(doc_type: str, qr: zatca.ZatcaQR | None, rows: list[Row]) -> dict[str, Field]:
    out: dict[str, Field] = {}
    latin_rows = [r.latin for r in rows]
    page_text = "\n".join(latin_rows)

    # -- identifiers -------------------------------------------------------
    numbers, originals = _invoice_numbers(rows)
    out["invoice_number"] = Field(numbers[0], "read", "ocr") if numbers else \
        Field(None, "review", "ocr", "invoice number not found")
    if doc_type in ("credit_note", "debit_note"):
        out["original_invoice"] = Field(originals[0], "read", "ocr") if originals else \
            Field(None, "review", "ocr", "reference to the original invoice not found")

    vats = list(dict.fromkeys(VAT_NUMBER.findall(page_text.replace(" ", ""))))
    dates = find_dates(page_text)

    # -- seller, date, totals ----------------------------------------------
    triple = _totals_triple(rows)
    items = _line_items(rows)
    if qr is not None:
        out["seller_name"] = Field(qr.seller_name, "verified", "qr", "from the ZATCA QR code")
        vat_ok = zatca.valid_vat_number(qr.seller_vat)
        out["seller_vat"] = Field(qr.seller_vat, "verified" if vat_ok else "review", "qr",
                                  "from the ZATCA QR code" if vat_ok else "QR VAT number breaks ZATCA format")
        qr_date = qr.timestamp[:10]
        printed = {d.isoformat() for d in dates}
        if printed and qr_date not in printed:
            out["issue_date"] = Field(qr_date, "review", "qr", "printed date differs from the QR")
        else:
            out["issue_date"] = Field(qr_date, "verified", "qr", "from the ZATCA QR code")
        total, vat = qr.total, qr.vat_total
        sub = total - vat
        # Printed totals that add up on their own but disagree with the QR: possible tampering.
        mismatch = triple is not None and triple[2] != total and _printed_outside_items(triple[2], rows)
        status = "review" if mismatch else "verified"
        note = "printed totals differ from the QR" if mismatch else "from the ZATCA QR code"
        out["total"] = Field(total, status, "qr", note)
        out["vat_total"] = Field(vat, status, "qr", note)
        out["subtotal"] = Field(sub, status, "qr", "total minus VAT from the QR" if not mismatch else note)
        if not zatca.vat_matches(sub, vat, Decimal("0.05")):
            for k in ("total", "vat_total", "subtotal"):
                out[k] = Field(out[k].value, "review", "qr", "VAT is not 15% of the subtotal")
    else:
        if vats:
            ok = zatca.valid_vat_number(vats[0])
            out["seller_vat"] = Field(vats[0], "read" if ok else "review", "ocr", "QR not readable")
        else:
            out["seller_vat"] = Field(None, "review", "ocr", "not found")
        seller = _arabic_seller(rows)
        out["seller_name"] = Field(seller, "read" if seller else "review", "ocr", "QR not readable")
        out["issue_date"] = Field(dates[0].isoformat(), "read", "ocr", "QR not readable") if dates else \
            Field(None, "review", "ocr", "not found")
        if triple is not None:
            sub, vat, total = triple
            lines_sum = sum((n for _, _, n in items), Decimal("0"))
            # The triple must be the invoice totals, not one line's net/VAT/total: the
            # total has to be printed outside the line-item rows, and the lines found
            # must add up to the subtotal.
            outside = _printed_outside_items(total, rows)
            proven = outside and (not items or lines_sum == sub)
            st = "verified" if proven else "read"
            note = ("subtotal + VAT = total, VAT = 15%, lines add up" if proven
                    else "totals add up, but the line items do not confirm them")
            out["subtotal"] = Field(sub, st, "arithmetic", note)
            out["vat_total"] = Field(vat, st, "arithmetic", note)
            out["total"] = Field(total, st, "arithmetic", note)
        else:
            for k in ("subtotal", "vat_total", "total"):
                out[k] = Field(None, "review", "ocr", "QR not readable and totals do not add up")

    if doc_type != "simplified_tax_invoice":
        buyer = [v for v in vats if v != out["seller_vat"].value]
        out["buyer_vat"] = Field(buyer[0], "read", "ocr") if buyer else \
            Field(None, "review", "ocr", "buyer VAT number not found")

    # -- line items ----------------------------------------------------------
    lines_sum = sum((n for _, _, n in items), Decimal("0"))
    sub_field = out.get("subtotal")
    value = [{"qty": q, "unit": u, "net": n} for q, u, n in items]
    if items and sub_field is not None and sub_field.status == "verified" and lines_sum == sub_field.value:
        out["line_items"] = Field(value, "verified", "arithmetic", "qty x price = net on every line; lines sum to the subtotal")
    elif items:
        out["line_items"] = Field(value, "review", "ocr", f"lines sum to {lines_sum:.2f}, subtotal differs or unproven")
    else:
        out["line_items"] = Field([], "review", "ocr", "no line items found")
    return out


def _invoice_numbers(rows: list[Row]) -> tuple[list[str], list[str]]:
    numbers, originals = [], []
    prev_ref = False
    for r in rows:
        both = squash(r.latin + " " + r.arabic)
        is_ref = any(w in both for w in REFERENCE_WORDS)
        for m in INVOICE_NO.finditer(r.latin.replace(" ", "")):
            val = m.group(1)
            if val.isdigit():
                continue
            (originals if (is_ref or prev_ref) else numbers).append(val)
        # A label alone on its row ("Reference invoice") applies to the next row.
        prev_ref = is_ref and not INVOICE_NO.search(r.latin.replace(" ", ""))
    return list(dict.fromkeys(numbers)), list(dict.fromkeys(originals))


def _all_amounts(rows: list[Row]) -> list[Decimal]:
    return [a for r in rows for _, a in amounts(r.latin)]


TOTAL_WORDS = [squash(w) for w in ["total", "amount due", "الإجمالي", "المستحق", "المجموع"]]


def _printed_outside_items(value: Decimal, rows: list[Row]) -> bool:
    """The value is printed on a totals row: one with a total-type label that is not
    itself a line item."""
    for r in rows:
        if value in [a for _, a in amounts(r.latin)] and _row_item(r) is None:
            words = squash(r.latin + " " + r.arabic)
            if any(w in words for w in TOTAL_WORDS):
                return True
    return False


def _totals_triple(rows: list[Row]) -> tuple[Decimal, Decimal, Decimal] | None:
    """Find subtotal, VAT and total by arithmetic, not by labels: a + b = c with
    b = 15% of a. The largest such triple is the invoice total."""
    vals = sorted(set(_all_amounts(rows)))
    best = None
    pool = set(vals)
    for a in vals:
        for b in vals:
            if b >= a or (a + b) not in pool:
                continue
            if abs((a * Decimal("0.15")).quantize(Decimal("0.01")) - b) <= Decimal("0.05"):
                if best is None or a + b > best[2]:
                    best = (a, b, a + b)
    return best


def _row_item(r: Row) -> tuple[int, Decimal, Decimal] | None:
    """qty, unit price and net amount on one row, with qty x price = net."""
    amts = amounts(r.latin)
    for pq, q in integers(r.latin):
        if not 1 <= q <= 999:
            continue
        for i, (pu, u) in enumerate(amts):
            if pu <= pq:
                continue
            for _, n in amts[i + 1:]:
                if (u * q).quantize(Decimal("0.01")) == n:
                    return q, u, n
    return None


def _line_items(rows: list[Row]) -> list[tuple[int, Decimal, Decimal]]:
    return [it for it in (_row_item(r) for r in rows) if it]


def _arabic_seller(rows: list[Row]) -> str:
    for r in rows[:6]:
        for l in r.lines:
            t = l.arabic.strip()
            if t and (t.startswith("شركة") or t.startswith("مؤسسة")):
                return t
    return ""


# ---------------------------------------------------------------------------
# Boarding passes
# ---------------------------------------------------------------------------
BOARDING_FIELDS = ["passenger_name", "pnr", "from_airport", "to_airport", "carrier", "flight_number",
                   "julian_date", "seat", "sequence", "cabin"]


def _boarding_fields(bp: bcbp.BoardingPass | None, rows: list[Row]) -> dict[str, Field]:
    if bp is not None:
        return {n: Field(getattr(bp, n), "verified", "barcode", "IATA BCBP barcode (error-corrected)")
                for n in BOARDING_FIELDS}
    # Barcode unreadable: fall back to the printed text, unproven.
    found = _boarding_from_print(rows)
    return {n: Field(found[n], "read", "ocr", "barcode not readable; printed text")
            if found.get(n) is not None else Field(None, "review", "ocr", "barcode not readable")
            for n in BOARDING_FIELDS}


def _value_near(rows: list[Row], label: str, pattern: str) -> str | None:
    """Value printed after a label on the same line or row, or just below it."""
    lab = squash(label)
    rx = re.compile(pattern)
    for ri, r in enumerate(rows):
        for li, l in enumerate(r.lines):
            sq = l.latin.upper()
            if lab not in squash(sq):
                continue
            tail = sq[sq.upper().find(label.split()[-1].upper()) + len(label.split()[-1]):] \
                if label.split()[-1].upper() in sq else ""
            m = rx.search(tail)
            if m:
                return m.group(1)
            for other in r.lines[li + 1:li + 2]:
                m = rx.fullmatch(other.latin.upper().strip())
                if m:
                    return m.group(1)
            # Next rows: the line horizontally closest to the label.
            for below in rows[ri + 1:ri + 3]:
                near = min(below.lines, key=lambda x: abs(x.box[0][0] - l.box[0][0]))
                if abs(near.box[0][0] - l.box[0][0]) < 2.5 * l.height:
                    m = rx.fullmatch(near.latin.upper().strip())
                    if m:
                        return m.group(1)
    return None


def _boarding_from_print(rows: list[Row]) -> dict:
    out: dict = {}
    text = "\n".join(r.latin.upper() for r in rows)
    m = re.search(r"\b([A-Z]{2,})/([A-Z]{2,})", text)
    if m:
        out["passenger_name"] = f"{m.group(1)}/{m.group(2)}"[:20]
    for r in rows:
        m = re.fullmatch(r"\s*([A-Z]{3})\s*\W{0,4}\s*([A-Z]{3})\s*", r.latin.upper())
        if m:
            out["from_airport"], out["to_airport"] = m.group(1), m.group(2)
            break
    pnr = _value_near(rows, "Booking ref", r"\b([A-Z0-9]{6})\b") or \
        (re.search(r"\bREF\s*([A-Z0-9]{6})\b", text) or [None, None])[1]
    out["pnr"] = pnr
    flight = _value_near(rows, "Flight", r"\b([A-Z0-9]{2}\s?\d{1,4})\b")
    if flight:
        f = flight.replace(" ", "")
        out["carrier"], out["flight_number"] = f[:2], f[2:].lstrip("0")
    out["seat"] = _value_near(rows, "Seat", r"\b(\d{1,2}[A-K])\b")
    seq = _value_near(rows, "Seq", r"\b(\d{1,4})\b")
    out["sequence"] = seq.lstrip("0") if seq else None
    out["cabin"] = _value_near(rows, "Class", r"\b([A-Z])\b")
    day = _value_near(rows, "Date", r"\b(\d{2}[A-Z]{3})\b")
    mon = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    if day and day[2:] in mon:
        # Day of year, as in the barcode, counted in a non-leap year.
        try:
            out["julian_date"] = date(2026, mon.index(day[2:]) + 1, int(day[:2])).timetuple().tm_yday
        except ValueError:
            pass
    return out
