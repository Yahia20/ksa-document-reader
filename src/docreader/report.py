"""Build the results page (site/index.html) from the evaluation output.

    python -m docreader.report

Reads results/test/metrics.json, results/test/predictions.jsonl and, when present,
results/midv500/metrics.json. Copies a few example images into site/examples/.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import cv2
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"

TYPE_NAMES = {
    "passport": ("Passport", "جواز سفر"),
    "id_card_td1": ("ID card", "بطاقة هوية"),
    "id_card_td2": ("Travel document (TD2)", "وثيقة سفر (TD2)"),
    "visa": ("Visa", "تأشيرة"),
    "tax_invoice": ("Tax invoice", "فاتورة ضريبية"),
    "simplified_tax_invoice": ("Simplified tax invoice", "فاتورة ضريبية مبسطة"),
    "credit_note": ("Credit note", "إشعار دائن"),
    "debit_note": ("Debit note", "إشعار مدين"),
    "boarding_pass": ("Boarding pass", "بطاقة صعود الطائرة"),
    "other": ("Other document", "مستند آخر"),
}
FIELD_NAMES = {
    "document_number": ("Document number", "رقم الوثيقة"), "surname": ("Surname", "اللقب"),
    "given_names": ("Given names", "الاسم"), "nationality": ("Nationality", "الجنسية"),
    "birth_date": ("Date of birth", "تاريخ الميلاد"), "sex": ("Sex", "الجنس"),
    "expiry_date": ("Expiry date", "تاريخ الانتهاء"), "issuing_state": ("Issuing state", "جهة الإصدار"),
    "name_ar": ("Name in Arabic", "الاسم بالعربية"), "invoice_number": ("Invoice number", "رقم الفاتورة"),
    "issue_date": ("Issue date", "تاريخ الإصدار"), "seller_name": ("Seller", "البائع"),
    "seller_vat": ("Seller VAT no.", "الرقم الضريبي للبائع"), "buyer_vat": ("Buyer VAT no.", "الرقم الضريبي للمشتري"),
    "subtotal": ("Subtotal", "الإجمالي قبل الضريبة"), "vat_total": ("VAT 15%", "ضريبة القيمة المضافة"),
    "total": ("Total", "الإجمالي"), "line_items": ("Line items", "بنود الفاتورة"),
    "original_invoice": ("Original invoice", "الفاتورة الأصلية"), "passenger_name": ("Passenger", "المسافر"),
    "pnr": ("Booking reference", "رقم الحجز"), "from_airport": ("From", "من"), "to_airport": ("To", "إلى"),
    "carrier": ("Airline code", "رمز شركة الطيران"), "flight_number": ("Flight", "رقم الرحلة"),
    "julian_date": ("Day of year", "يوم الرحلة في السنة"), "seat": ("Seat", "المقعد"),
    "sequence": ("Check-in no.", "رقم تسجيل الوصول"), "cabin": ("Cabin", "الدرجة"),
}
LEVEL_NAMES = {"scan": ("Scanned", "ممسوح ضوئيًا"), "photo": ("Phone photo", "صورة بالجوال"),
               "hard": ("Difficult photo", "صورة صعبة")}
SOURCE_NAMES = {"mrz": ("MRZ", "المنطقة المقروءة آليًا"), "qr": ("ZATCA QR", "رمز QR الضريبي"),
                "barcode": ("Barcode", "الباركود"), "ocr": ("Printed text", "النص المطبوع"),
                "arithmetic": ("Arithmetic", "الحساب")}
# Examples for the page: (doc_type, preferred image quality, prefer a held-out design).
EXAMPLES = [("passport", "photo", False), ("tax_invoice", "photo", True),
            ("simplified_tax_invoice", "scan", False), ("id_card_td1", "photo", False),
            ("boarding_pass", "photo", False), ("credit_note", "hard", False)]


def _pick_examples(preds: list[dict], data_dir: Path) -> list[dict]:
    out = []
    for doc_type, level, held in EXAMPLES:
        cands = [p for p in preds if p["doc_type"] == doc_type and p["pred_type"] == doc_type
                 and p["level"] == level and (p["heldout"] or not held)]
        if held:
            cands = [p for p in cands if p["heldout"]] or cands
        # Show an honest one: correct where verified, and for the hard photo, one
        # that sends something to review.
        def good(p):
            ok = all(f["correct"] for f in p["fields"] if f["status"] == "verified")
            some_review = any(f["status"] != "verified" for f in p["fields"])
            return ok and (some_review if level == "hard" else True)
        cands = [p for p in cands if good(p)] or cands
        if not cands:
            continue
        p = max(cands, key=lambda p: sum(f["status"] == "verified" for f in p["fields"]))
        img = cv2.imread(str(data_dir / f"{p['id']}.jpg"))
        scale = 900 / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        name = f"examples/{p['id']}.jpg"
        (SITE / "examples").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(SITE / name), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        correct = {f["field"]: f["correct"] for f in p["fields"]}
        fields = []
        for k, f in p["pred_fields"].items():
            v = f["value"]
            if isinstance(v, list):
                v = f"{len(v)} lines" if v else "—"
            fields.append({"name": FIELD_NAMES.get(k, (k, k)), "value": "—" if v is None else v,
                           "status": f["status"], "source": SOURCE_NAMES.get(f["source"], (f["source"],) * 2),
                           "correct": correct.get(k)})
        out.append({"id": p["id"], "image": name, "type": TYPE_NAMES[doc_type],
                    "level": LEVEL_NAMES[p["level"]], "heldout": p["heldout"], "fields": fields,
                    "seconds": p["seconds"]})
    return out


def _pct(x: float | None, d: int = 1) -> str:
    if x is None:
        return "—"
    return "100%" if x == 1 else f"{100 * x:.{d}f}%"


def build() -> Path:
    m = json.loads((ROOT / "results" / "test" / "metrics.json").read_text(encoding="utf-8"))
    preds = [json.loads(l) for l in open(ROOT / "results" / "test" / "predictions.jsonl", encoding="utf-8")]
    midv_path = ROOT / "results" / "midv500" / "metrics.json"
    midv = json.loads(midv_path.read_text(encoding="utf-8")) if midv_path.exists() else None
    examples = _pick_examples(preds, ROOT / "data" / "synthetic" / "test")

    env = Environment(loader=FileSystemLoader(Path(__file__).parent), autoescape=True)
    html = env.get_template("report_template.html").render(
        m=m, midv=midv, examples=examples, types=TYPE_NAMES, levels=LEVEL_NAMES,
        built=date.today().isoformat(), pct=_pct)
    SITE.mkdir(exist_ok=True)
    out = SITE / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
