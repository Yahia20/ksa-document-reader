"""Machine Readable Zone (MRZ) parsing and verification, ICAO Doc 9303.

The MRZ is the two or three lines of OCR-B text at the bottom of a passport, ID card or
visa. Most of its fields carry a check digit, so a misread is caught by arithmetic
instead of by guessing. This module turns noisy OCR lines into a parsed MRZ and says,
field by field, whether the value was proven by a check digit.

Formats handled:
    TD3   passport            2 x 44
    TD2   ID / travel doc     2 x 36
    TD1   ID card             3 x 30
    MRV-A visa (full page)    2 x 44, first letter V
    MRV-B visa (small)        2 x 36, first letter V
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from itertools import product

from .countries import ICAO_CODES

WEIGHTS = (7, 3, 1)

# OCR confusions between letters and digits. A field that must be numeric is
# repaired with TO_DIGIT, a field that must be alphabetic with TO_ALPHA.
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1", "Z": "2",
            "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"}
TO_ALPHA = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A", "7": "T"}
# Letter/digit pairs whose values differ by a multiple of 10 (G=16 vs 6, S=28 vs 8,
# L=21 vs 1 ...). Swapping one for the other leaves every 7-3-1 check digit unchanged,
# so in an alphanumeric field the check digit cannot tell them apart.
BLIND_TWINS = {"A": "0", "K": "0", "U": "0", "B": "1", "L": "1", "C": "2", "M": "2",
               "D": "3", "N": "3", "E": "4", "F": "5", "P": "5", "Z": "5", "G": "6", "Q": "6",
               "H": "7", "R": "7", "I": "8", "S": "8", "J": "9", "T": "9"}
DIGIT_TWINS: dict[str, list[str]] = {}
for _letter, _digit in BLIND_TWINS.items():
    DIGIT_TWINS.setdefault(_digit, []).append(_letter)


def blind_spots(value: str) -> list[int]:
    """Positions where a letter sits next to a digit (or a digit next to a letter) and
    has a twin the check digit cannot tell apart. A misread there goes unnoticed."""
    out = []
    for i, c in enumerate(value):
        nb = [value[j] for j in (i - 1, i + 1) if 0 <= j < len(value) and value[j] != "<"]
        if c in BLIND_TWINS and any(n.isdigit() for n in nb):
            out.append(i)
        elif c.isdigit() and any(n.isalpha() for n in nb):
            out.append(i)
    return out


def blind_alternatives(value: str) -> list[str]:
    """Other values with the same check digit, one blind-spot swap away."""
    alts = []
    for i in blind_spots(value):
        c = value[i]
        for twin in ([BLIND_TWINS[c]] if c.isalpha() else DIGIT_TWINS.get(c, [])):
            alts.append(value[:i] + twin + value[i + 1:])
    return alts


# Swaps tried on alphanumeric document numbers when the check digit fails.
AMBIGUOUS = {"0": "O", "O": "0", "1": "I", "I": "1", "5": "S", "S": "5", "8": "B", "B": "8",
             "2": "Z", "Z": "2", "6": "G", "G": "6", "Q": "0", "D": "0"}


def char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    if c == "<":
        return 0
    if "A" <= c <= "Z":
        return ord(c) - 55
    raise ValueError(f"not an MRZ character: {c!r}")


def check_digit(s: str) -> str:
    return str(sum(char_value(c) * WEIGHTS[i % 3] for i, c in enumerate(s)) % 10)


def is_valid_check(data: str, digit: str) -> bool:
    """A check digit may be '<' when the whole field is empty filler."""
    if digit == "<":
        return set(data) <= {"<"}
    return digit.isdigit() and check_digit(data) == digit


# ---------------------------------------------------------------------------
# Layouts. Each field: (name, line, start, end, kind, check_digit_position or None)
# kind: "a" alphabetic, "n" numeric, "an" alphanumeric, "name", "sex"
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    name: str
    lines: int
    width: int
    fields: tuple
    composite: tuple | None  # (line, position, [(line, start, end), ...])


TD3 = Layout("TD3", 2, 44, (
    ("document_code", 0, 0, 2, "a", None),
    ("issuing_state", 0, 2, 5, "a", None),
    ("names", 0, 5, 44, "name", None),
    ("document_number", 1, 0, 9, "an", (1, 9)),
    ("nationality", 1, 10, 13, "a", None),
    ("birth_date", 1, 13, 19, "n", (1, 19)),
    ("sex", 1, 20, 21, "sex", None),
    ("expiry_date", 1, 21, 27, "n", (1, 27)),
    ("personal_number", 1, 28, 42, "an", (1, 42)),
), (1, 43, [(1, 0, 10), (1, 13, 20), (1, 21, 43)]))

TD2 = Layout("TD2", 2, 36, (
    ("document_code", 0, 0, 2, "a", None),
    ("issuing_state", 0, 2, 5, "a", None),
    ("names", 0, 5, 36, "name", None),
    ("document_number", 1, 0, 9, "an", (1, 9)),
    ("nationality", 1, 10, 13, "a", None),
    ("birth_date", 1, 13, 19, "n", (1, 19)),
    ("sex", 1, 20, 21, "sex", None),
    ("expiry_date", 1, 21, 27, "n", (1, 27)),
    ("optional_data", 1, 28, 35, "an", None),
), (1, 35, [(1, 0, 10), (1, 13, 20), (1, 21, 35)]))

TD1 = Layout("TD1", 3, 30, (
    ("document_code", 0, 0, 2, "a", None),
    ("issuing_state", 0, 2, 5, "a", None),
    ("document_number", 0, 5, 14, "an", (0, 14)),
    ("optional_data", 0, 15, 30, "an", None),
    ("birth_date", 1, 0, 6, "n", (1, 6)),
    ("sex", 1, 7, 8, "sex", None),
    ("expiry_date", 1, 8, 14, "n", (1, 14)),
    ("nationality", 1, 15, 18, "a", None),
    ("optional_data_2", 1, 18, 29, "an", None),
    ("names", 2, 0, 30, "name", None),
), (1, 29, [(0, 5, 30), (1, 0, 7), (1, 8, 15), (1, 18, 29)]))

MRVA = Layout("MRV-A", 2, 44, (
    ("document_code", 0, 0, 2, "a", None),
    ("issuing_state", 0, 2, 5, "a", None),
    ("names", 0, 5, 44, "name", None),
    ("document_number", 1, 0, 9, "an", (1, 9)),
    ("nationality", 1, 10, 13, "a", None),
    ("birth_date", 1, 13, 19, "n", (1, 19)),
    ("sex", 1, 20, 21, "sex", None),
    ("expiry_date", 1, 21, 27, "n", (1, 27)),
    ("optional_data", 1, 28, 44, "an", None),
), None)

MRVB = Layout("MRV-B", 2, 36, (
    ("document_code", 0, 0, 2, "a", None),
    ("issuing_state", 0, 2, 5, "a", None),
    ("names", 0, 5, 36, "name", None),
    ("document_number", 1, 0, 9, "an", (1, 9)),
    ("nationality", 1, 10, 13, "a", None),
    ("birth_date", 1, 13, 19, "n", (1, 19)),
    ("sex", 1, 20, 21, "sex", None),
    ("expiry_date", 1, 21, 27, "n", (1, 27)),
    ("optional_data", 1, 28, 36, "an", None),
), None)


LAYOUTS = {l.name: l for l in (TD3, TD2, TD1, MRVA, MRVB)}


def passes_field_check(res: "MRZResult", name: str, value: str) -> bool:
    """Would this value pass the check digit printed for the field?"""
    _, _, a, b, _, chk = next(f for f in LAYOUTS[res.layout].fields if f[0] == name)
    padded = (value.replace(" ", "<") + "<" * (b - a))[:b - a]
    return chk is not None and is_valid_check(padded, res.lines[chk[0]][chk[1]])


def layout_for(lines: list[str]) -> Layout | None:
    shape = (len(lines), len(lines[0]) if lines else 0)
    visa = bool(lines) and lines[0][:1] == "V"
    return {
        (2, 44): MRVA if visa else TD3,
        (2, 36): MRVB if visa else TD2,
        (3, 30): TD1,
    }.get(shape)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class MRZField:
    value: str
    status: str  # "verified" | "read" | "review"
    reason: str = ""
    alternatives: list[str] = field(default_factory=list)  # other readings that also passed


@dataclass
class MRZResult:
    layout: str
    lines: list[str]
    fields: dict[str, MRZField] = field(default_factory=dict)
    composite_ok: bool | None = None
    corrections: int = 0
    whole_ok: bool = False

    @property
    def plausible(self) -> bool:
        """Is this an MRZ at all? A real document code and issuing state, a valid sex
        value, and at least one of the two dates shaped like a date."""
        f = self.fields
        codes = all(f.get(n) is not None and f[n].status == "read" for n in ("document_code", "issuing_state"))
        sex = f.get("sex") is not None and f["sex"].status == "read"
        dates = any(re.fullmatch(r"\d{4}-(\d{2}|\?\?)-(\d{2}|\?\?)", str(f[n].value))
                    for n in ("birth_date", "expiry_date") if n in f)
        return codes and sex and dates

    @property
    def all_checks_pass(self) -> bool:
        return self.whole_ok and not any(f.reason == "check digit failed" for f in self.fields.values())


# ---------------------------------------------------------------------------
# Cleaning OCR text into candidate MRZ lines
# ---------------------------------------------------------------------------
MRZ_CHARS = re.compile(r"[^A-Z0-9<]")


def clean_line(text: str) -> str:
    """Normalise one OCR line to the MRZ alphabet."""
    t = text.upper().replace(" ", "")
    t = t.replace("«", "<<").replace("‹", "<").replace("〈", "<").replace("≤", "<")
    t = re.sub(r"[\(\[\{]", "<", t)
    t = MRZ_CHARS.sub("", t)
    # Runs of '<' are often read as K or C inside long filler stretches.
    t = re.sub(r"(?<=<)[KC](?=<)", "<", t)
    t = re.sub(r"<[KC]{1,2}<", lambda m: "<" * len(m.group()), t)
    return t


# Birth date + check, sex, expiry date + check: the backbone of every data line.
DATES_SEX = re.compile(r"[0-9OQDILZSBGT]{7}[MFX<HNKEP][0-9OQDILZSBGT]{7}")


def looks_like_mrz(text: str) -> bool:
    t = clean_line(text)
    return len(t) >= 26 and (t.count("<") >= 2 or bool(re.match(r"^[PIACV][A-Z<][A-Z<]{3}", t))
                             or bool(DATES_SEX.search(t)))


def fit_width(line: str, width: int, pad_right: bool = True) -> str:
    """Pad or trim filler so the line has the expected width."""
    if len(line) == width:
        return line
    if len(line) < width:
        return line + "<" * (width - len(line)) if pad_right else line
    excess = len(line) - width
    stripped = line.rstrip("<")
    if len(line) - len(stripped) >= excess:
        return line[:width]
    return line[:width]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _fix_kind(s: str, kind: str) -> str:
    if kind == "n":
        return "".join(TO_DIGIT.get(c, c) for c in s)
    if kind == "a":
        return "".join(TO_ALPHA.get(c, c) for c in s)
    if kind == "name":
        return "".join(TO_ALPHA.get(c, c) for c in s)
    if kind == "sex":
        return {"H": "M", "N": "M", "E": "F", "P": "F", "K": "<"}.get(s, s)
    return s


def _parse_date(yymmdd: str, future: bool) -> str | None:
    # ICAO allows unknown month/day as filler, e.g. "83<<<<".
    if re.fullmatch(r"\d{2}(\d{2}|<<)(\d{2}|<<)", yymmdd) and "<" in yymmdd:
        yy = int(yymmdd[:2])
        century = 1900 if (not future and yy > date.today().year % 100) else 2000
        mm = yymmdd[2:4] if yymmdd[2:4] != "<<" else "??"
        dd = yymmdd[4:] if yymmdd[4:] != "<<" else "??"
        return f"{century + yy}-{mm}-{dd}"
    if not yymmdd.isdigit():
        return None
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:])
    this_year = date.today().year % 100
    if future:  # expiry: up to ~20 years ahead
        century = 2000 if yy <= this_year + 20 else 1900
    else:  # birth: never in the future
        century = 1900 if yy > this_year else 2000
    try:
        return date(century + yy, mm, dd).isoformat()
    except ValueError:
        return None


def _split_names(raw: str) -> tuple[str, str]:
    """SURNAME<<GIVEN<NAMES<<<<... The given names end at the next double filler;
    anything after it is a misread filler character (often an E or K), not a name."""
    raw = raw.rstrip("<")
    surname, _, given = raw.partition("<<")
    given = given.split("<<")[0]
    return surname.replace("<", " ").strip(), given.replace("<", " ").strip()


def _apply_kinds(lines: list[str], layout: Layout) -> list[str]:
    chars = [list(l) for l in lines]
    for _, li, a, b, kind, chk in layout.fields:
        seg = "".join(chars[li][a:b])
        fixed = _fix_kind(seg, kind)
        chars[li][a:b] = list(fixed)
        if chk:
            cl, cp = chk
            chars[cl][cp] = TO_DIGIT.get(chars[cl][cp], chars[cl][cp])
    if layout.composite:
        cl, cp, _ = layout.composite
        chars[cl][cp] = TO_DIGIT.get(chars[cl][cp], chars[cl][cp])
    return ["".join(c) for c in chars]


def _composite_ok(lines: list[str], layout: Layout) -> bool | None:
    if not layout.composite:
        return None
    cl, cp, parts = layout.composite
    data = "".join(lines[l][a:b] for l, a, b in parts)
    return is_valid_check(data, lines[cl][cp])


def _repair_document_number(lines: list[str], layout: Layout) -> tuple[list[str], int]:
    """Try letter/digit swaps (O/0, I/1, S/5 ...) on the document number when its
    check digit fails, and keep a swap only if exactly one variant passes.

    The composite digit is no second guard here: the document number sits at the
    same 7-3-1 weight offset in both sums, so any swap that fixes the field digit
    also fixes the composite. When the misread was not a letter/digit swap, a wrong
    variant still passes about one time in ten, so a repaired number is only ever
    offered as a suggestion for review, never reported as verified.
    """
    spec = next(f for f in layout.fields if f[0] == "document_number")
    _, li, a, b, _, (cl, cp) = spec
    seg, digit = lines[li][a:b], lines[cl][cp]
    if is_valid_check(seg, digit):
        return lines, 0
    positions = [i for i, c in enumerate(seg) if c in AMBIGUOUS]
    if not positions or len(positions) > 4:
        return lines, 0
    passing = []
    for mask in product([False, True], repeat=len(positions)):
        if not any(mask):
            continue
        cand = list(seg)
        for i, flip in zip(positions, mask):
            if flip:
                cand[i] = AMBIGUOUS[cand[i]]
        cand = "".join(cand)
        if not is_valid_check(cand, digit):
            continue
        trial = list(lines)
        trial[li] = trial[li][:a] + cand + trial[li][b:]
        if layout.composite is None or _composite_ok(trial, layout):
            passing.append((trial, sum(mask)))
    if len(passing) == 1:
        return passing[0]
    return lines, 0


def parse(lines: list[str]) -> MRZResult | None:
    """Parse already-cleaned MRZ lines of equal, standard width."""
    layout = layout_for(lines)
    if layout is None:
        return None
    lines = _apply_kinds(lines, layout)
    lines, corrections = _repair_document_number(lines, layout)
    result = MRZResult(layout.name, lines, corrections=corrections)
    result.composite_ok = _composite_ok(lines, layout)
    # One field check digit is only a 1-in-10 guard, too weak on its own for a garbled
    # line. Fields are proven only when the MRZ holds together as a whole: the composite
    # digit passes (TD1/TD2/TD3) or, on visas, which have none, every field digit passes;
    # and the document code and issuing state are real values.
    field_checks = [is_valid_check(lines[li][a:b], lines[chk[0]][chk[1]])
                    for _, li, a, b, _, chk in layout.fields if chk]
    whole_ok = bool(result.composite_ok) if layout.composite else all(field_checks)
    state = lines[0][2:5]
    structure_ok = lines[0][0] in "PIACV" and (state in ICAO_CODES or state.replace("<", "") in ICAO_CODES)
    result.whole_ok = whole_ok and structure_ok

    for name, li, a, b, kind, chk in layout.fields:
        raw = lines[li][a:b]
        if name == "names":
            surname, given = _split_names(raw)
            ok = bool(surname) and re.fullmatch(r"[A-Z ]+", surname + " " + given) is not None
            status = "read" if ok else "review"
            result.fields["surname"] = MRZField(surname, status, "no check digit for names")
            result.fields["given_names"] = MRZField(given, status, "no check digit for names")
            continue

        value = raw.rstrip("<").replace("<", " ").strip() if kind in ("an", "name") else raw
        if chk:
            cl, cp = chk
            ok = is_valid_check(raw, lines[cl][cp])
            if name.endswith("_date"):
                parsed = _parse_date(raw, future=(name == "expiry_date"))
                ok = ok and parsed is not None
                value = parsed or raw
            alternatives = []
            if ok and name == "document_number" and corrections:
                status, reason = "review", "suggested O/0-type fix, please confirm"
            elif ok and kind == "an" and blind_spots(value.replace(" ", "")):
                status, reason = "review", "check digit cannot rule out letter/digit look-alikes"
                alternatives = blind_alternatives(value.replace(" ", ""))
            elif ok and not result.whole_ok:
                status, reason = "review", "field digit passed but the MRZ as a whole does not check out"
            elif ok:
                status, reason = "verified", "check digit"
            else:
                status, reason = "review", "check digit failed"
            result.fields[name] = MRZField(value, status, reason, alternatives)
        elif name in ("issuing_state", "nationality"):
            code = raw.replace("<", "")
            ok = raw in ICAO_CODES or code in ICAO_CODES
            result.fields[name] = MRZField(code, "read" if ok else "review",
                                           "ICAO country code" if ok else "unknown country code")
        elif name == "sex":
            ok = raw in ("M", "F", "<", "X")
            result.fields[name] = MRZField({"<": "X"}.get(raw, raw), "read" if ok else "review",
                                           "allowed value" if ok else "unexpected value")
        elif name == "document_code":
            ok = raw[0] in "PIACV"
            result.fields[name] = MRZField(raw.replace("<", ""), "read" if ok else "review",
                                           "allowed value" if ok else "unexpected value")
        else:
            result.fields[name] = MRZField(value, "read", "free field")
    return result


# ---------------------------------------------------------------------------
# From raw OCR lines to the best MRZ reading
# ---------------------------------------------------------------------------
def _score(res: MRZResult) -> tuple:
    verified = sum(f.status == "verified" for f in res.fields.values())
    return (res.whole_ok, res.composite_ok is True, verified, -res.corrections)


def find_mrz(rows: list[str | list[str]]) -> MRZResult | None:
    """Pick the MRZ out of all text rows on a page and parse it.

    Each row may carry several readings (for example from two OCR models); every
    combination is tried and the check digits decide which reading wins. OCR may
    also split one MRZ line into two pieces or miscount filler, so pieces are joined
    and each plausible layout is tried.
    """
    alts = [sorted({clean_line(t) for t in ([r] if isinstance(r, str) else r)}) for r in rows]
    # Join pieces of a line that OCR split in two (adjacent, each too short alone).
    cand: list[list[str]] = []
    i = 0
    while i < len(alts):
        here = [c for c in alts[i] if c]
        nxt = [c for c in alts[i + 1] if c] if i + 1 < len(alts) else []
        joined = [a + b for a in here for b in nxt
                  if len(a) < 30 and len(b) < 30 and 28 <= len(a) + len(b) <= 46 and looks_like_mrz(a + b)]
        if joined:
            cand.append(joined)
            i += 2
            continue
        good = [c for c in here if looks_like_mrz(c)]
        if good:
            cand.append(good)
        i += 1
    if not cand:
        return None

    results: list[MRZResult] = []
    seen: set[tuple[str, ...]] = set()
    for layout in (TD3, MRVA, TD2, MRVB, TD1):
        n, w = layout.lines, layout.width
        for i in range(len(cand) - n + 1):
            for window in product(*cand[i:i + n]):
                if any(abs(len(l) - w) > 4 for l in window):
                    continue
                for variant in _width_variants(list(window), w):
                    key = tuple(variant)
                    if key in seen or layout_for(variant) is not layout:
                        continue
                    seen.add(key)
                    res = parse(variant)
                    if res:
                        results.append(res)
    results = [r for r in results if r.plausible]
    if not results:
        return None
    top = max(_score(r) for r in results)
    best_group = [r for r in results if _score(r) == top]
    best = best_group[0]
    # Two different readings that pass the same checks: the fields they disagree
    # on cannot be trusted, whichever one is picked.
    for other in best_group[1:]:
        for name, f in best.fields.items():
            o = other.fields.get(name)
            if o is not None and o.value != f.value:
                if o.value not in f.alternatives:
                    f.alternatives.append(o.value)
                if f.status != "review":
                    f.status, f.reason = "review", "two readings pass the checks"
    return best


def _width_variants(window: list[str], width: int) -> list[list[str]]:
    """Fit each line to the width. When a line is one character off, also try
    adding or removing one '<' inside an existing filler run, which is how OCR
    usually miscounts a long stretch of '<<<<'."""
    base = [fit_width(l, width) for l in window]
    variants = [base]
    for idx, line in enumerate(window):
        if len(line) == width - 1:
            seen = set()
            for p in range(1, width):
                if line[p - 1] != "<" and (p >= len(line) or line[p] != "<"):
                    continue
                s = line[:p] + "<" + line[p:]
                if s not in seen:
                    seen.add(s)
                    v = list(base)
                    v[idx] = s
                    variants.append(v)
        elif len(line) == width + 1:
            seen = set()
            for p in range(len(line)):
                if line[p] != "<":
                    continue
                s = line[:p] + line[p + 1:]
                if s not in seen:
                    seen.add(s)
                    v = list(base)
                    v[idx] = s
                    variants.append(v)
    return variants


def to_text(res: MRZResult) -> str:
    return "\n".join(res.lines)
