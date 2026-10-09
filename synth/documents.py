"""Build one synthetic document: template + context to render, and the ground truth."""
from __future__ import annotations

import base64
import random
import re
from dataclasses import dataclass
from datetime import date, timedelta

import zxingcpp

from docreader import bcbp, zatca
from docreader.mrz import check_digit

from .fake import AIRLINES, Faketory, Person, money

DOC_TYPES = ["passport", "id_card_td1", "id_card_td2", "visa", "tax_invoice",
             "simplified_tax_invoice", "credit_note", "debit_note", "boarding_pass", "other"]

# Templates per type. The second list holds designs kept out of development and used
# only in the test split, to measure how the reader copes with a layout it never saw.
TEMPLATES = {
    "passport": (["passport_a.html", "passport_b.html"], ["passport_c.html"]),
    "id_card_td1": (["id_td1.html"], []),
    "id_card_td2": (["id_td2.html"], []),
    "visa": (["visa.html"], []),
    "tax_invoice": (["invoice_classic.html", "invoice_modern.html"], ["invoice_arabic.html"]),
    "simplified_tax_invoice": (["receipt.html", "invoice_classic.html"], ["invoice_arabic.html"]),
    "credit_note": (["invoice_classic.html", "invoice_modern.html", "receipt.html"], ["invoice_arabic.html"]),
    "debit_note": (["invoice_classic.html", "invoice_modern.html"], ["invoice_arabic.html"]),
    "boarding_pass": (["boarding_a.html"], ["boarding_b.html"]),
    "other": (["other_letter.html", "other_statement.html"], []),
}
INVOICE_KIND = {"tax_invoice": "standard", "simplified_tax_invoice": "simplified",
                "credit_note": "credit", "debit_note": "debit"}
TITLES = {
    "standard": ("فاتورة ضريبية", "Tax Invoice"),
    "simplified": ("فاتورة ضريبية مبسطة", "Simplified Tax Invoice"),
    "credit": ("إشعار دائن", "Credit Note"),
    "debit": ("إشعار مدين", "Debit Note"),
}
NAT_NAMES = {"SAU": "SAUDI", "EGY": "EGYPTIAN", "JOR": "JORDANIAN", "YEM": "YEMENI", "SDN": "SUDANESE",
             "ARE": "EMIRATI", "KWT": "KUWAITI", "BHR": "BAHRAINI", "OMN": "OMANI", "IND": "INDIAN",
             "PAK": "PAKISTANI", "PHL": "FILIPINO", "GBR": "BRITISH", "USA": "AMERICAN",
             "DEU": "GERMAN", "FRA": "FRENCH", "TUR": "TURKISH", "IDN": "INDONESIAN"}
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
ACCENTS = ["#0b6e4f", "#1d4e89", "#7b2d26", "#5b3e96", "#00695c", "#ad5d00", "#37474f"]
# MRZ font size in px per template so the OCR-B pitch matches ICAO's 2.54 mm at 10 px/mm.
# OCR-B advance is 0.723 em, so 35 px gives a 25.3 px pitch.
MRZ_SIZE = {"passport_a.html": 35, "passport_b.html": 35, "passport_c.html": 34,
            "id_td1.html": 35, "id_td2.html": 35, "visa.html": 35}


@dataclass
class SynthDoc:
    doc_type: str
    template: str
    context: dict
    truth: dict


# ---------------------------------------------------------------------------
# MRZ construction
# ---------------------------------------------------------------------------
def _f(s: str, n: int) -> str:
    return (s.replace(" ", "<") + "<" * n)[:n]


def _names(p: Person, width: int) -> str:
    return _f(p.surname.replace(" ", "<") + "<<" + p.given_names.replace(" ", "<"), width)


def _yymmdd(d: date) -> str:
    return d.strftime("%y%m%d")


def mrz_td3(code: str, p: Person, number: str, expiry: date) -> list[str]:
    l1 = _f(code, 2) + "UTO" + _names(p, 39)
    doc = _f(number, 9)
    dob, exp = _yymmdd(p.birth_date), _yymmdd(expiry)
    personal = "<" * 14
    body = doc + check_digit(doc) + p.nationality + dob + check_digit(dob) + p.sex + exp + check_digit(exp)
    body += personal + "<"
    comp = body[0:10] + body[13:20] + body[21:43]
    return [l1, body + check_digit(comp)]


def mrz_td2(code: str, p: Person, number: str, expiry: date) -> list[str]:
    l1 = _f(code, 2) + "UTO" + _names(p, 31)
    doc = _f(number, 9)
    dob, exp = _yymmdd(p.birth_date), _yymmdd(expiry)
    body = doc + check_digit(doc) + p.nationality + dob + check_digit(dob) + p.sex + exp + check_digit(exp)
    body += "<" * 7
    comp = body[0:10] + body[13:20] + body[21:35]
    return [l1, body + check_digit(comp)]


def mrz_td1(code: str, p: Person, number: str, expiry: date) -> list[str]:
    doc = _f(number, 9)
    l1 = _f(code, 2) + "UTO" + doc + check_digit(doc) + "<" * 15
    dob, exp = _yymmdd(p.birth_date), _yymmdd(expiry)
    l2 = dob + check_digit(dob) + p.sex + exp + check_digit(exp) + p.nationality + "<" * 11
    comp = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
    return [l1, l2 + check_digit(comp), _names(p, 30)]


def mrz_mrva(p: Person, number: str, expiry: date) -> list[str]:
    l1 = "V<" + "UTO" + _names(p, 39)
    doc = _f(number, 9)
    dob, exp = _yymmdd(p.birth_date), _yymmdd(expiry)
    l2 = doc + check_digit(doc) + p.nationality + dob + check_digit(dob) + p.sex + exp + check_digit(exp)
    return [l1, l2 + "<" * 16]


def fit_person(p: Person, width: int) -> Person:
    """Keep the full name inside the MRZ so the truth is never a truncated name."""
    while len(p.surname) + 2 + len(p.given_names) > width and " " in p.given_names:
        p.given_names = p.given_names.rsplit(" ", 1)[0]
        p.given_ar = p.given_ar.rsplit(" ", 1)[0] if p.given_ar else p.given_ar
    if len(p.surname) + 2 + len(p.given_names) > width:
        p.given_names = p.given_names[: max(1, width - len(p.surname) - 2)]
    return p


def print_date(d: date, r: random.Random) -> str:
    style = r.randint(0, 3)
    if style == 0:
        return f"{d.day:02d} {MONTHS[d.month - 1]} {d.year}"
    if style == 1:
        return d.strftime("%d.%m.%Y")
    if style == 2:
        return d.strftime("%d/%m/%Y")
    return d.isoformat()


def _svg(text: str, fmt) -> str:
    """Barcode as inline SVG that scales with CSS (zxing omits the viewBox)."""
    svg = zxingcpp.create_barcode(text, fmt).to_svg(add_quiet_zones=True)
    svg = svg[svg.find("<svg"):]
    m = re.match(r'<svg width="(\d+)" height="(\d+)"', svg)
    w, h = m.group(1), m.group(2)
    return svg.replace(m.group(0), f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
                                   f'shape-rendering="crispEdges"', 1)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build(doc_type: str, template: str, fk: Faketory) -> SynthDoc:
    r = fk.r
    if doc_type in ("passport", "id_card_td1", "id_card_td2", "visa"):
        return _id_doc(doc_type, template, fk)
    if doc_type in INVOICE_KIND:
        return _invoice(doc_type, template, fk)
    if doc_type == "boarding_pass":
        return _boarding(template, fk)
    return _other(template, fk, r)


def _id_doc(doc_type: str, template: str, fk: Faketory) -> SynthDoc:
    r = fk.r
    width = {"passport": 39, "id_card_td1": 30, "id_card_td2": 31, "visa": 39}[doc_type]
    p = fit_person(fk.person(), width)
    number = fk.doc_number()
    years = {"passport": 10, "id_card_td1": 5, "id_card_td2": 5, "visa": 1}[doc_type]
    issued, expiry = fk.issue_expiry(years)
    if doc_type == "visa":
        expiry = issued + timedelta(days=r.choice([30, 90, 180, 365]))
    code = {"passport": "P", "id_card_td1": r.choice(["I", "ID"]), "id_card_td2": r.choice(["I", "AC"]),
            "visa": "V"}[doc_type]
    lines = {"passport": mrz_td3, "id_card_td1": mrz_td1, "id_card_td2": mrz_td2}.get(doc_type)
    mrz = lines(code, p, number, expiry) if lines else mrz_mrva(p, number, expiry)
    name_ar = f"{p.given_ar} {p.surname_ar}".strip() if p.surname_ar else ""
    ctx = {
        "mrz": "\n".join(mrz), "mrz_size": MRZ_SIZE[template], "doc_number": number, "code": code,
        "surname": p.surname, "given_names": p.given_names, "name_ar": name_ar, "sex": p.sex,
        "nationality": p.nationality, "nationality_name": NAT_NAMES.get(p.nationality, p.nationality),
        "birth_print": print_date(p.birth_date, r), "issue_print": print_date(issued, r),
        "expiry_print": print_date(expiry, r), "place_of_birth": p.place_of_birth,
        "c1": r.choice(["#e3f2fd", "#e8f5e9", "#fff8e1", "#fce4ec"]),
        "c2": r.choice(["#b3e5fc", "#c8e6c9", "#ffe0b2", "#d1c4e9"]),
        "visa_type": r.choice(["TOURIST", "BUSINESS", "VISIT", "TRANSIT"]),
        "entries": r.choice(["SINGLE", "MULTIPLE"]), "stay": r.choice([30, 60, 90]),
        "remarks": r.choice(["NONE", "NOT VALID FOR EMPLOYMENT", "HAJJ AND UMRAH NOT PERMITTED"]),
    }
    truth = {
        "document_number": number, "surname": p.surname, "given_names": p.given_names,
        "nationality": p.nationality, "birth_date": p.birth_date.isoformat(), "sex": p.sex,
        "expiry_date": expiry.isoformat(), "issuing_state": "UTO", "mrz": mrz,
    }
    if name_ar and doc_type != "visa":  # the visa design prints no Arabic name
        truth["name_ar"] = name_ar
    return SynthDoc(doc_type, template, ctx, truth)


def _qr_payload(inv, kind: str, r: random.Random) -> tuple[str, int]:
    ts = inv.issued.strftime("%Y-%m-%dT%H:%M:%S") + r.choice(["Z", "", "Z"])
    tags = {1: inv.seller.name_ar, 2: inv.seller.vat, 3: ts, 4: f"{inv.total:.2f}", 5: f"{inv.vat_total:.2f}"}
    phase = 2 if r.random() < 0.6 else 1
    if phase == 2:  # random stand-ins for the hash, signature, key and stamp
        tags[6] = base64.b64encode(r.randbytes(32)).decode()
        tags[7] = base64.b64encode(r.randbytes(71)).decode()
        tags[8] = r.randbytes(88)
        if kind == "simplified":
            tags[9] = r.randbytes(72)
    return zatca.encode_tlv(tags), phase


def _invoice(doc_type: str, template: str, fk: Faketory) -> SynthDoc:
    r = fk.r
    kind = INVOICE_KIND[doc_type]
    inv = fk.invoice(kind)
    if template == "receipt.html":  # a till receipt carries no buyer block
        inv.buyer = None
    payload, phase = _qr_payload(inv, kind, r)
    title_ar, title_en = TITLES[kind]
    if kind == "credit" and r.random() < 0.3:
        title_ar = "إشعار دائن للفاتورة الضريبية"
    thousands = r.random() < 0.6
    fmt = (lambda x: f"{x:,.2f}") if thousands else (lambda x: f"{x:.2f}")
    date_style = r.randint(0, 3)
    d = inv.issued
    date_print = [d.strftime("%Y-%m-%d"), d.strftime("%d/%m/%Y"), d.strftime("%d-%m-%Y %H:%M"),
                  f"{d.day:02d} {MONTHS[d.month - 1].title()} {d.year}"][date_style]
    narrow = template == "receipt.html"
    ctx = {
        "inv": inv, "fmt": fmt, "title_ar": title_ar, "title_en": title_en, "date_print": date_print,
        "qr_svg": _svg(payload, zxingcpp.BarcodeFormat.QRCode), "qr_px": r.randint(190, 260) if not narrow else 300,
        "accent": r.choice(ACCENTS), "cur": r.choice(["SAR", "ر.س", "SAR"]),
    }
    truth = {
        "invoice_type": doc_type, "invoice_number": inv.number, "issue_date": d.date().isoformat(),
        "seller_name": inv.seller.name_ar, "seller_vat": inv.seller.vat,
        "subtotal": f"{inv.subtotal:.2f}", "vat_total": f"{inv.vat_total:.2f}", "total": f"{inv.total:.2f}",
        "line_count": len(inv.lines),
        "lines": [{"qty": l.qty, "unit": f"{l.unit:.2f}", "net": f"{l.net:.2f}"} for l in inv.lines],
        "qr_phase": phase,
    }
    if inv.buyer:
        truth["buyer_vat"] = inv.buyer.vat
    if inv.original_number:
        truth["original_invoice"] = inv.original_number
    return SynthDoc(doc_type, template, ctx, truth)


def _boarding(template: str, fk: Faketory) -> SynthDoc:
    r = fk.r
    p = fk.person()
    f = fk.flight()
    given = p.given_names.split(" ")[0]
    pax20 = f"{p.surname}/{given}"[:20]
    seat_num, seat_letter = f["seat"][:-1], f["seat"][-1]
    bp = bcbp.BoardingPass(
        passenger_name=pax20, pnr=f["pnr"], from_airport=f["from"], to_airport=f["to"],
        carrier=f["carrier"], flight_number=f"{int(f['flight']):04d} ", julian_date=f["julian"],
        cabin=f["cabin"], seat=f"{int(seat_num):03d}{seat_letter}", sequence=f"{int(f['sequence']):04d} ")
    text = bcbp.build(bp)
    fmt = zxingcpp.BarcodeFormat.PDF417 if template == "boarding_a.html" else \
        r.choice([zxingcpp.BarcodeFormat.Aztec, zxingcpp.BarcodeFormat.QRCode])
    day = date(2026, 1, 1) + timedelta(days=f["julian"] - 1)
    ctx = {"f": f, "pax": f"{p.surname}/{p.given_names} {'MR' if p.sex == 'M' else 'MS'}",
           "date_print": f"{day.day:02d}{MONTHS[day.month - 1]}", "bc_svg": _svg(text, fmt),
           "bc_w": 520 if template == "boarding_a.html" else 380, "accent": r.choice(ACCENTS)}
    truth = {"passenger_name": pax20, "pnr": f["pnr"], "from_airport": f["from"], "to_airport": f["to"],
             "carrier": f["carrier"], "flight_number": str(int(f["flight"])), "julian_date": f["julian"],
             "seat": f"{int(seat_num)}{seat_letter}", "sequence": str(int(f["sequence"])), "cabin": f["cabin"]}
    return SynthDoc("boarding_pass", template, ctx, truth)


def _other(template: str, fk: Faketory, r: random.Random) -> SynthDoc:
    c = fk.company("hotels")
    p = fk.person()
    if template == "other_letter.html":
        ctx = {"hotel_en": c.name_en, "hotel_ar": c.name_ar, "guest": f"{p.given_names.title()} {p.surname.title()}",
               "conf": f"HB{r.randint(100000, 999999)}", "checkin": print_date(date(2026, 1, 1) + timedelta(days=r.randint(0, 300)), r),
               "nights": r.randint(1, 9), "room": r.choice(["Deluxe King", "Twin Room", "Junior Suite"]),
               "accent": r.choice(ACCENTS)}
    else:
        rows = []
        for _ in range(r.randint(4, 10)):
            amt = money(r.uniform(50, 9000))
            dr = r.random() < 0.6
            rows.append((print_date(date(2026, 1, 1) + timedelta(days=r.randint(0, 300)), r),
                         r.choice(["Card payment", "Transfer", "Fee", "Refund", "Cash deposit"]),
                         f"{amt:,.2f}" if dr else "", "" if dr else f"{amt:,.2f}"))
        ctx = {"company_en": c.name_en, "company_ar": c.name_ar,
               "customer": f"{p.given_names.title()} {p.surname.title()}", "period": "2026",
               "rows": rows, "qr_svg": _svg(f"https://example.com/account/{r.randint(10**8, 10**9)}",
                                            zxingcpp.BarcodeFormat.QRCode)}
    return SynthDoc("other", template, ctx, {})
