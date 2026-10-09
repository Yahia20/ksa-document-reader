"""IATA Bar Coded Boarding Pass (BCBP, Resolution 792): build and parse the mandatory items.

The first 60 characters of a boarding-pass barcode are fixed-width:

    M 1 SURNAME/GIVEN........ E PNR.... FRM TO_ CAR FLT__ JJJ C SEAT SEQ__ S XX
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

LAYOUT = (  # name, width
    ("format_code", 1), ("legs", 1), ("passenger_name", 20), ("eticket", 1), ("pnr", 7),
    ("from_airport", 3), ("to_airport", 3), ("carrier", 3), ("flight_number", 5),
    ("julian_date", 3), ("cabin", 1), ("seat", 4), ("sequence", 5), ("status", 1),
    ("var_size", 2),
)
MANDATORY_LEN = sum(w for _, w in LAYOUT)  # 60


@dataclass
class BoardingPass:
    passenger_name: str
    pnr: str
    from_airport: str
    to_airport: str
    carrier: str
    flight_number: str
    julian_date: int
    cabin: str
    seat: str
    sequence: str

    def flight_date(self, year: int) -> date:
        return date(year, 1, 1) + timedelta(days=self.julian_date - 1)


def build(bp: BoardingPass) -> str:
    vals = {
        "format_code": "M", "legs": "1", "passenger_name": bp.passenger_name, "eticket": "E",
        "pnr": bp.pnr, "from_airport": bp.from_airport, "to_airport": bp.to_airport,
        "carrier": bp.carrier, "flight_number": bp.flight_number,
        "julian_date": f"{bp.julian_date:03d}", "cabin": bp.cabin, "seat": bp.seat,
        "sequence": bp.sequence, "status": "1", "var_size": "00",
    }
    return "".join(vals[n][:w].ljust(w) for n, w in LAYOUT)


def parse(text: str) -> BoardingPass:
    if len(text) < MANDATORY_LEN or text[0] != "M" or not text[1].isdigit():
        raise ValueError("not an IATA BCBP string")
    vals, i = {}, 0
    for name, width in LAYOUT:
        vals[name] = text[i:i + width]
        i += width
    if not vals["julian_date"].isdigit() or not 1 <= int(vals["julian_date"]) <= 366:
        raise ValueError("bad julian date")
    return BoardingPass(
        passenger_name=vals["passenger_name"].strip(), pnr=vals["pnr"].strip(),
        from_airport=vals["from_airport"].strip(), to_airport=vals["to_airport"].strip(),
        carrier=vals["carrier"].strip(), flight_number=vals["flight_number"].strip().lstrip("0"),
        julian_date=int(vals["julian_date"]), cabin=vals["cabin"], seat=vals["seat"].strip().lstrip("0"),
        sequence=vals["sequence"].strip().lstrip("0"),
    )
