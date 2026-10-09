"""Fictional people, companies, items and flights for the synthetic test documents.

Nothing here belongs to a real person or business. Identity documents are issued by
"UTO" (Utopia), the specimen state ICAO uses in Doc 9303, and every document carries
a SPECIMEN mark.
"""
from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

from faker import Faker

# (Arabic, Latin transliteration)
ARAB_MALE = [("محمد", "MOHAMMED"), ("عبدالله", "ABDULLAH"), ("فهد", "FAHAD"), ("خالد", "KHALID"),
             ("سلطان", "SULTAN"), ("عبدالعزيز", "ABDULAZIZ"), ("فيصل", "FAISAL"), ("سعود", "SAUD"),
             ("تركي", "TURKI"), ("نايف", "NAYEF"), ("ماجد", "MAJED"), ("ناصر", "NASSER"),
             ("يوسف", "YOUSEF"), ("عمر", "OMAR"), ("أحمد", "AHMED"), ("بندر", "BANDAR"),
             ("ريان", "RAYAN"), ("سلمان", "SALMAN"), ("مشعل", "MISHAL"), ("عبدالرحمن", "ABDULRAHMAN"),
             ("حسن", "HASSAN"), ("إبراهيم", "IBRAHIM"), ("مصطفى", "MOSTAFA"), ("طارق", "TAREK")]
ARAB_FEMALE = [("نورة", "NOURA"), ("سارة", "SARAH"), ("ريم", "REEM"), ("لمى", "LAMA"), ("هيا", "HAYA"),
               ("منيرة", "MUNIRA"), ("العنود", "ALANOUD"), ("جواهر", "JAWAHER"), ("دانة", "DANA"),
               ("شهد", "SHAHAD"), ("رهف", "RAHAF"), ("لطيفة", "LATIFA"), ("مها", "MAHA"),
               ("فاطمة", "FATIMAH"), ("عائشة", "AISHA"), ("مريم", "MARIAM"), ("ياسمين", "YASMIN")]
ARAB_FAMILY = [("العتيبي", "ALOTAIBI"), ("القحطاني", "ALQAHTANI"), ("الشمري", "ALSHAMMARI"),
               ("الدوسري", "ALDOSARI"), ("الغامدي", "ALGHAMDI"), ("الزهراني", "ALZAHRANI"),
               ("الحربي", "ALHARBI"), ("المطيري", "ALMUTAIRI"), ("السبيعي", "ALSUBAIE"),
               ("العنزي", "ALANAZI"), ("الشهري", "ALSHEHRI"), ("السلمي", "ALSULAMI"),
               ("الرشيد", "ALRASHEED"), ("الحمدان", "ALHAMDAN"), ("البلوي", "ALBALAWI"),
               ("الجهني", "ALJUHANI"), ("عبدالفتاح", "ABDELFATTAH"), ("المصري", "ELMASRY"),
               ("حسانين", "HASSANEIN"), ("النجار", "ELNAGGAR"), ("الخطيب", "ALKHATIB")]
ARAB_NATIONALITIES = ["SAU", "SAU", "SAU", "EGY", "JOR", "YEM", "SDN", "ARE", "KWT", "BHR", "OMN"]
OTHER_NATIONALITIES = {"IND": "en_IN", "PAK": "en_PK", "PHL": "fil_PH", "GBR": "en_GB",
                       "USA": "en_US", "DEU": "de_DE", "FRA": "fr_FR", "TUR": "tr_TR", "IDN": "id_ID"}

# Seller names: (prefix_ar, prefix_en) + core + sector
CORES = [("الواحة", "Al Waha"), ("النخبة", "Al Nukhba"), ("الريادة", "Al Riyada"), ("الأفق", "Al Ofoq"),
         ("سدير", "Sudair"), ("المرجان", "Al Marjan"), ("الياسمين", "Al Yasmin"), ("الصفوة", "Al Safwa"),
         ("البيان", "Al Bayan"), ("الرواد", "Al Rowad"), ("الندى", "Al Nada"), ("الفجر", "Al Fajr"),
         ("سنا", "Sana"), ("مدار", "Madar"), ("ركايز", "Rakayez"), ("نجد", "Najd"), ("تهامة", "Tihama"),
         ("الحجاز", "Al Hijaz"), ("رمال", "Rimal"), ("أصيل", "Aseel")]
SECTORS = {
    "travel": ("للسفر والسياحة", "Travel & Tourism"),
    "trading": ("للتجارة", "Trading"),
    "logistics": ("للخدمات اللوجستية", "Logistics Services"),
    "food": ("للأغذية", "Foods"),
    "tech": ("لتقنية المعلومات", "Information Technology"),
    "hotels": ("للفنادق", "Hotels"),
    "medical": ("للمستلزمات الطبية", "Medical Supplies"),
}
ITEMS = {
    "travel": [("تذكرة طيران الرياض - القاهرة", "Air ticket RUH-CAI", 900, 2400),
               ("تذكرة طيران جدة - دبي", "Air ticket JED-DXB", 600, 1800),
               ("حجز فندق ٣ ليالٍ", "Hotel booking, 3 nights", 750, 3200),
               ("رسوم تأشيرة سياحية", "Tourist visa fee", 300, 535),
               ("تأمين سفر", "Travel insurance", 90, 400),
               ("نقل من المطار", "Airport transfer", 120, 450),
               ("باقة عمرة", "Umrah package", 2500, 7500),
               ("رسوم خدمة", "Service fee", 50, 250)],
    "trading": [("ورق طباعة A4 (كرتون)", "A4 copy paper (carton)", 95, 160),
                ("حبر طابعة أسود", "Black printer toner", 180, 420),
                ("كرسي مكتب", "Office chair", 350, 1200), ("مكتب خشبي", "Wooden desk", 800, 2600),
                ("خزانة ملفات", "Filing cabinet", 450, 1300)],
    "logistics": [("شحن بري الرياض - جدة", "Road freight RUH-JED", 1200, 4800),
                  ("تخزين - شهر", "Warehousing, 1 month", 900, 3500),
                  ("تغليف وتعبئة", "Packing service", 150, 900), ("تخليص جمركي", "Customs clearance", 600, 2200)],
    "food": [("أرز بسمتي ١٠ كجم", "Basmati rice 10 kg", 70, 120), ("زيت دوار الشمس", "Sunflower oil", 28, 60),
             ("تمر سكري ٣ كجم", "Sukkari dates 3 kg", 60, 150), ("قهوة عربية ٥٠٠ جم", "Arabic coffee 500 g", 35, 95),
             ("حليب طويل الأجل", "UHT milk (carton)", 45, 80)],
    "tech": [("رخصة برنامج سنوية", "Annual software licence", 1200, 9000),
             ("لابتوب أعمال", "Business laptop", 3200, 7800), ("شاشة ٢٧ بوصة", "27-inch monitor", 800, 1900),
             ("دعم فني - ساعة", "Technical support, 1 hour", 150, 400), ("استضافة سحابية - شهر", "Cloud hosting, 1 month", 300, 2500)],
    "hotels": [("غرفة ديلوكس - ليلة", "Deluxe room, 1 night", 450, 1500), ("إفطار", "Breakfast", 60, 140),
               ("قاعة اجتماعات", "Meeting room", 900, 3000), ("خدمة غسيل", "Laundry service", 40, 180)],
    "medical": [("قفازات طبية (علبة)", "Medical gloves (box)", 25, 70), ("كمامات (علبة)", "Face masks (box)", 20, 60),
                ("جهاز قياس ضغط", "Blood pressure monitor", 180, 520), ("مطهر ٥ لتر", "Disinfectant 5 L", 45, 130)],
}
CITIES = [("الرياض", "Riyadh"), ("جدة", "Jeddah"), ("الدمام", "Dammam"), ("مكة المكرمة", "Makkah"),
          ("المدينة المنورة", "Madinah"), ("الخبر", "Khobar"), ("أبها", "Abha"), ("تبوك", "Tabuk")]
STREETS = [("طريق الملك فهد", "King Fahd Rd"), ("شارع التحلية", "Tahlia St"), ("طريق الأمير سلطان", "Prince Sultan Rd"),
           ("شارع العليا", "Olaya St"), ("طريق المدينة", "Madinah Rd"), ("شارع الأمير محمد بن عبدالعزيز", "Prince Mohammed bin Abdulaziz St")]
DISTRICTS = [("حي العليا", "Al Olaya"), ("حي الملقا", "Al Malqa"), ("حي الروضة", "Al Rawdah"),
             ("حي الشاطئ", "Al Shati"), ("حي النزهة", "Al Nuzha"), ("حي الفيصلية", "Al Faisaliyah")]
NOTE_REASONS = [("إرجاع بضاعة", "Goods returned"), ("خصم لاحق", "Post-sale discount"),
                ("تعديل سعر", "Price correction"), ("إلغاء حجز", "Booking cancelled"),
                ("رسوم إضافية", "Additional charges")]
AIRPORTS = ["RUH", "JED", "DMM", "MED", "AHB", "TUU", "CAI", "DXB", "AMM", "IST", "LHR", "CDG",
            "KUL", "BOM", "KHI", "MNL", "CGK", "DOH", "BAH", "MCT", "KWI"]
AIRLINES = [("Rimal Airways", "طيران رمال", "RQ"), ("Najd Air", "طيران نجد", "NQ"),
            ("Oasis Wings", "أجنحة الواحة", "OQ"), ("Falcon Link", "فالكون لينك", "FQ")]

MONEY = Decimal("0.01")


def money(x) -> Decimal:
    return Decimal(str(x)).quantize(MONEY, rounding=ROUND_HALF_UP)


def mrz_name(s: str) -> str:
    """Transliterate a Latin name the way ICAO 9303 does: no accents, A-Z only."""
    s = s.upper().replace("Ä", "AE").replace("Ö", "OE").replace("Ü", "UE").replace("ß", "SS")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.replace("'", "").replace("-", " ").replace(".", "")
    return " ".join("".join(c for c in w if "A" <= c <= "Z") for w in s.split()).strip()


@dataclass
class Person:
    surname: str
    given_names: str
    surname_ar: str = ""
    given_ar: str = ""
    sex: str = "M"
    nationality: str = "UTO"
    birth_date: date = date(1990, 1, 1)
    place_of_birth: str = ""


@dataclass
class Company:
    name_ar: str
    name_en: str
    vat: str
    cr: str
    city: tuple
    street: tuple
    district: tuple
    building: str
    postal: str
    sector: str


@dataclass
class Line:
    desc_ar: str
    desc_en: str
    qty: int
    unit: Decimal

    @property
    def net(self) -> Decimal:
        return money(self.unit * self.qty)

    @property
    def vat(self) -> Decimal:
        return money(self.net * Decimal("0.15"))


@dataclass
class Invoice:
    kind: str  # standard | simplified | credit | debit
    number: str
    issued: datetime
    seller: Company
    buyer: Company | None
    lines: list[Line] = field(default_factory=list)
    original_number: str = ""
    reason: tuple = ("", "")

    @property
    def subtotal(self) -> Decimal:
        return money(sum(l.net for l in self.lines))

    @property
    def vat_total(self) -> Decimal:
        return money(sum(l.vat for l in self.lines))

    @property
    def total(self) -> Decimal:
        return self.subtotal + self.vat_total


class Faketory:
    def __init__(self, seed: int):
        self.r = random.Random(seed)
        self.fakers = {loc: Faker(loc) for loc in set(OTHER_NATIONALITIES.values())}
        for i, f in enumerate(self.fakers.values()):
            f.seed_instance(seed * 31 + i)

    # -- people -------------------------------------------------------------
    def person(self) -> Person:
        r = self.r
        sex = r.choice("MF")
        birth = date(1950, 1, 1) + timedelta(days=r.randint(0, 365 * 55))
        if r.random() < 0.6:
            given = r.choice(ARAB_MALE if sex == "M" else ARAB_FEMALE)
            father = r.choice(ARAB_MALE)
            fam = r.choice(ARAB_FAMILY)
            two = r.random() < 0.5
            return Person(
                surname=fam[1], given_names=given[1] + (" " + father[1] if two else ""),
                surname_ar=fam[0], given_ar=given[0] + (" " + father[0] if two else ""),
                sex=sex, nationality=r.choice(ARAB_NATIONALITIES), birth_date=birth,
                place_of_birth=r.choice(CITIES)[1].upper(),
            )
        nat, loc = r.choice(list(OTHER_NATIONALITIES.items()))
        f = self.fakers[loc]
        first = f.first_name_male() if sex == "M" else f.first_name_female()
        if r.random() < 0.25:
            first += " " + (f.first_name_male() if sex == "M" else f.first_name_female())
        return Person(surname=mrz_name(f.last_name()), given_names=mrz_name(first), sex=sex,
                      nationality=nat, birth_date=birth, place_of_birth=mrz_name(f.city())[:20])

    def doc_number(self) -> str:
        r = self.r
        style = r.randint(0, 2)
        if style == 0:
            return r.choice("ABCDEFGHJKLMNPRSTUVWXYZ") + "".join(r.choice("0123456789") for _ in range(8))
        if style == 1:
            return "".join(r.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(2)) + \
                "".join(r.choice("0123456789") for _ in range(7))
        return "".join(r.choice("0123456789") for _ in range(9))

    def issue_expiry(self, years: int = 10) -> tuple[date, date]:
        issued = date(2017, 1, 1) + timedelta(days=self.r.randint(0, 365 * 9))
        anniversary = date(issued.year + years, issued.month, min(issued.day, 28))
        return issued, anniversary - timedelta(days=1)

    # -- companies ----------------------------------------------------------
    def vat_number(self) -> str:
        return "3" + "".join(self.r.choice("0123456789") for _ in range(13)) + "3"

    def company(self, sector: str | None = None) -> Company:
        r = self.r
        sector = sector or r.choice(list(SECTORS))
        core = r.choice(CORES)
        est = r.random() < 0.35
        s_ar, s_en = SECTORS[sector]
        name_ar = f"{'مؤسسة' if est else 'شركة'} {core[0]} {s_ar}"
        name_en = f"{core[1]} {s_en} {'Est.' if est else 'Co.'}"
        return Company(name_ar=name_ar, name_en=name_en, vat=self.vat_number(),
                       cr="10" + "".join(r.choice("0123456789") for _ in range(8)),
                       city=r.choice(CITIES), street=r.choice(STREETS), district=r.choice(DISTRICTS),
                       building=str(r.randint(1000, 9999)), postal=str(r.randint(11000, 34999)),
                       sector=sector)

    def invoice(self, kind: str) -> Invoice:
        r = self.r
        seller = self.company()
        buyer = None if kind == "simplified" else self.company()
        issued = datetime(2025, 1, 1) + timedelta(minutes=r.randint(0, 60 * 24 * 600))
        prefix = r.choice(["INV", "SI", "TX", "F", "BL"])
        if kind == "credit":
            prefix = r.choice(["CN", "CRN", "CR"])
        elif kind == "debit":
            prefix = r.choice(["DN", "DBN", "DR"])
        sep = r.choice(["-", "/", ""])
        number = f"{prefix}{sep}{issued.year}{sep}{r.randint(1, 99999):05d}" if r.random() < 0.6 \
            else f"{prefix}{sep}{r.randint(100000, 999999)}"
        n_lines = r.randint(1, 3) if kind in ("credit", "debit") else r.randint(1, 6)
        catalog = ITEMS[seller.sector]
        picks = r.sample(catalog, min(n_lines, len(catalog)))
        lines = [Line(a, e, r.randint(1, 4) if hi > 1000 else r.randint(1, 12),
                      money(r.uniform(lo, hi) // 5 * 5 + r.choice([0, 0, 0.5, 0.75, 0.99])))
                 for a, e, lo, hi in picks]
        inv = Invoice(kind, number, issued, seller, buyer, lines)
        if kind in ("credit", "debit"):
            inv.original_number = f"INV-{issued.year}-{r.randint(1, 99999):05d}"
            inv.reason = r.choice(NOTE_REASONS)
        return inv

    # -- flights ------------------------------------------------------------
    def flight(self) -> dict:
        r = self.r
        a, b = r.sample(AIRPORTS, 2)
        airline = r.choice(AIRLINES)
        return {
            "airline_en": airline[0], "airline_ar": airline[1], "carrier": airline[2],
            "from": a, "to": b, "flight": str(r.randint(10, 2999)),
            "julian": r.randint(1, 365), "cabin": r.choice("YYYYMBJ"),
            "seat": f"{r.randint(1, 45)}{r.choice('ABCDEF')}",
            "sequence": str(r.randint(1, 350)),
            "pnr": "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6)),
            "boarding": f"{r.randint(0, 23):02d}:{r.choice(['00', '15', '30', '45', '05', '50'])}",
            "gate": f"{r.choice('ABCDE')}{r.randint(1, 40)}",
        }
