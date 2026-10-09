"""ZATCA (Saudi e-invoicing, "Fatoora") QR code: TLV encode/decode and tax checks.

Every e-invoice in Saudi Arabia carries a QR code holding Base64 of TLV records
(1-byte tag, 1-byte length, UTF-8 value):

    1 seller name   2 VAT number   3 timestamp   4 total incl. VAT   5 VAT total
    6 invoice hash  7 ECDSA signature  8 public key  9 ZATCA stamp (phase 2 only)

The QR is error-corrected, so its values are exact. Matching them against the
printed text proves what OCR read.
"""
from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

VAT_RATE = Decimal("0.15")
TAG_NAMES = {1: "seller_name", 2: "seller_vat", 3: "timestamp", 4: "total", 5: "vat_total",
             6: "invoice_hash", 7: "signature", 8: "public_key", 9: "zatca_stamp"}
TEXT_TAGS = {1, 2, 3, 4, 5, 6, 7}


@dataclass
class ZatcaQR:
    seller_name: str
    seller_vat: str
    timestamp: str
    total: Decimal
    vat_total: Decimal
    phase: int
    raw_tags: dict[int, bytes]


def encode_tlv(fields: dict[int, str | bytes]) -> str:
    out = bytearray()
    for tag in sorted(fields):
        value = fields[tag]
        data = value.encode("utf-8") if isinstance(value, str) else value
        if len(data) > 255:
            raise ValueError(f"tag {tag} value is longer than 255 bytes")
        out += bytes([tag, len(data)]) + data
    return base64.b64encode(bytes(out)).decode("ascii")


def decode_tlv(payload: str | bytes) -> dict[int, bytes]:
    """Decode Base64 TLV. Raises ValueError when the payload is not ZATCA TLV."""
    if isinstance(payload, bytes):
        payload = payload.decode("ascii", errors="ignore")
    try:
        raw = base64.b64decode(payload.strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("not Base64") from exc
    tags: dict[int, bytes] = {}
    i = 0
    while i < len(raw):
        if i + 2 > len(raw):
            raise ValueError("truncated TLV header")
        tag, length = raw[i], raw[i + 1]
        if tag not in TAG_NAMES or i + 2 + length > len(raw):
            raise ValueError("unexpected TLV tag or length")
        tags[tag] = raw[i + 2:i + 2 + length]
        i += 2 + length
    if not {1, 2, 3, 4, 5} <= tags.keys():
        raise ValueError("missing mandatory ZATCA tags 1-5")
    return tags


def parse_qr(payload: str | bytes) -> ZatcaQR:
    tags = decode_tlv(payload)
    text = {t: v.decode("utf-8") for t, v in tags.items() if t in TEXT_TAGS}
    try:
        total, vat = Decimal(text[4]), Decimal(text[5])
    except InvalidOperation as exc:
        raise ValueError("amounts in tags 4/5 are not numbers") from exc
    return ZatcaQR(
        seller_name=text[1], seller_vat=text[2], timestamp=text[3], total=total,
        vat_total=vat, phase=2 if 6 in tags else 1, raw_tags=tags,
    )


def valid_vat_number(vat: str) -> bool:
    """ZATCA rule BR-KSA-39/40: 15 digits, first and last digit 3."""
    return bool(re.fullmatch(r"3\d{13}3", vat))


def vat_matches(subtotal: Decimal, vat: Decimal, tolerance: Decimal = Decimal("0.01")) -> bool:
    return abs((subtotal * VAT_RATE).quantize(Decimal("0.01")) - vat) <= tolerance
