# Saudi Document Reader

**[Open the results page](https://yahia20.github.io/ksa-document-reader/)** (English and Arabic) ·
[اقرأ بالعربية](README.ar.md)

Reads passports, ID cards, visas, ZATCA tax invoices and boarding passes from a scan or a phone
photo. Every field it returns carries a status:

| Status | Meaning |
|---|---|
| **verified** | proven independently: an MRZ check digit, an error-corrected barcode, the invoice arithmetic, or two separate reads that agree |
| **read** | read once and plausible, but nothing confirms it |
| **review** | a check failed, two readings disagree, or the field was not found |

Only verified fields are meant to go into another system without a person looking at them.
It is built on open-source models.

## The short answer

1. **3,076 of 3,076 verified fields were correct (100%)** on 600 test documents that were never
   used during development, including a third on page designs the reader had never seen.
2. **64% of all fields came back verified.** The other 36% were flagged with a reason. Worse
   photos lower that share; they do not raise the error rate.
3. **On real phone video of passports (MIDV-500), 99.86% of the fields it verified were correct.**
   It read the whole MRZ exactly in 40.3% of frames, behind the commercial Dynamsoft SDK (48.3%) and
   far ahead of the open-source tools (7.3% and 0.3%).

## Results on the test set

600 documents, 60 of each of 10 types. 30% are scans, 45% are phone photos and 25% are difficult
photos (steep angle, blur, shadow, glare, low resolution). Dev and test sets use different random
seeds, and three designs (a passport, an Arabic-only invoice, a mobile boarding pass) appear only
in the test set.

| | Fields | Verified | Verified and correct | Read: accuracy | Review |
|---|---:|---:|---:|---:|---:|
| **All** | 4,834 | 3,076 (64%) | **100%** | 90.8% | 536 |
| Scans | 1,450 | 72% | 100% | 94.3% | 76 |
| Phone photos | 2,174 | 70% | 100% | 93.3% | 146 |
| Difficult photos | 1,210 | 42% | 100% | 84.3% | 314 |
| Designs never seen | 1,115 | 61% | 100% | 85.8% | 174 |

| Document type | Share of fields verified |
|---|---:|
| Boarding pass | 90% |
| Visa | 69% |
| Travel document (TD2) | 69% |
| Passport | 68% |
| Simplified tax invoice | 64% |
| Tax invoice | 61% |
| ID card (TD1) | 53% |
| Debit note | 51% |
| Credit note | 49% |

Document type was right for 598 of 600 (99.7%): two simplified invoices were labelled as standard
tax invoices. 4.3 seconds per document on average.

Some fields can never be proven from the image: an invoice number, a buyer's VAT number, a name in
Arabic, sex on an ID. They stay **read**. The weakest field is the invoice line items, verified on
28–42% of invoices; they are verified only when every line is found and the lines add up to the
subtotal. Arabic names read by OCR are right 59–81% of the time, which is why they are never
verified.

## Real photos (MIDV-500)

The synthetic test was designed to be hard, but it is still synthetic. MRZ reading was therefore also scored on the
curated MIDV-500 benchmark from Dynamsoft: 3,315 frames from phone videos of 12 real specimen
passports, filmed on a table, in hand, on a keyboard, in clutter and partly out of frame.

| | Frames with the whole MRZ read exactly right |
|---|---:|
| Dynamsoft Capture Vision (commercial SDK) | 48.3% |
| **This reader** (open source) | **40.3%** |
| FastMRZ (open source) | 7.3% |
| PassportEye (open source) | 0.3% |

| | |
|---|---|
| Passport fields marked verified | 9,066, of which **99.86% correct** (13 wrong: 12 names, 1 number) |
| Frames fully proven (five key fields verified, nothing left for review) | 822 of 3,315 (25%), key fields right in **817 (99.4%)** |
| By filming condition, exact MRZ | table 74.7% · in hand 37.4% · clutter 35.4% · keyboard 26.0% · partly out of frame 19.8% |
| Speed | 4.6 s per 1080x1920 frame |

The reader trails the commercial SDK on raw reading, and proves about one frame in four; the rest
go to a person. Most of the 13 wrong verified fields are names where the MRZ read dropped a second
given name that the printed page shows (`LIENE` for `LIENE MARA`).

## How it works

1. **Flatten the photo.** Canny edges, the largest four-cornered outline, a perspective warp (OpenCV).
2. **Decode barcodes** with zxing-cpp: the ZATCA QR (TLV, phases 1 and 2) and the IATA boarding-pass
   barcode (PDF417, Aztec or QR). Both carry error correction, so their values are exact.
3. **Read the text twice.** RapidOCR (ONNX Runtime): PP-OCRv6 small detects lines and reads Latin text
   and digits; PP-OCRv5 Arabic reads Arabic. PP-OCRv6 does not cover Arabic yet, so each line keeps
   both readings.
4. **Check against the standard.**
   - **ICAO 9303 MRZ** (TD1, TD2, TD3, MRV-A/B): field check digits, the composite digit, valid
     country codes, letter/digit repair only where the field type forces it (a date cannot contain O).
     Readings from both OCR models are combined and the check digits choose.
   - **ZATCA**: QR values against the printed page, VAT = 15% of the subtotal, the VAT-number rule
     (15 digits, first and last 3). A printed total that adds up but disagrees with the QR is
     flagged as possible tampering.
   - **No QR?** Totals are found by arithmetic, not by labels: subtotal + VAT = total with VAT at
     15%, printed on a totals row, with the line items adding up to the subtotal.
5. **Give every field a status**, with the reason.

### Where check digits are blind

The ICAO check digit weights characters 7-3-1 and keeps the last digit of the sum. A letter is
worth 10 to 35, so letters 10 apart from a digit look identical to it: G (16) and 6, S (28) and 8,
L (21) and 1. A document number misread as `XK7347G47` instead of `XK7347647` passes its check
digit. The composite digit does not help either, because the document number sits at the same
weight offset in both sums. On real photos two misreads can also cancel out: `BD0002028` read as
`800002028` shifts the weighted sum by −21 and −39, together −60. So a document number is verified
only when the same number is also found in the printed text. Tests pin both cases down
(`test_check_digit_is_blind_to_ten_apart_twins`, `test_two_misreads_can_cancel_in_the_check_digit`).

## Run it

```bash
pip install -r requirements.txt
python -m playwright install chromium     # only to regenerate the synthetic documents

python -m docreader passport.jpg                     # fields and their status
python -m docreader passport.jpg --json              # full result
python -m docreader invoices/ --excel results.xlsx   # a folder into one spreadsheet, colour-coded

python run_pipeline.py                     # data -> tests -> dev and test scores -> results page
python scripts/fetch_midv500.py            # optional: ~8 GB download, keeps ~1 GB of frames
python -m docreader.evaluate_midv
```

On Windows, set `PYTHONPATH=src;.` (or `pip install -e .`) before running the modules directly.

## Project layout

```
src/docreader/   mrz.py  zatca.py  bcbp.py      the three standards
                 ocr.py  geometry.py           OCR and perspective correction
                 pipeline.py                   classification, extraction, statuses
                 evaluate.py  evaluate_midv.py scoring
                 report.py                     builds site/index.html
synth/           fake.py  documents.py         fictional people, companies, MRZ, QR, BCBP
                 templates/  render.py         14 HTML designs rendered with headless Chrome
                 degrade.py                    scan / photo / difficult-photo effects
results/         metrics and every prediction, test and MIDV-500
tests/           unit tests for the standards
```

## Good to know

- **All test documents are fictional.** Identity documents are issued by "Utopia" (UTO), the
  specimen state ICAO uses, and carry a SPECIMEN mark. Companies, VAT numbers and airlines are
  invented. No real customer document was used.
- **The automation rate is for these designs.** A business would confirm it on its own documents.
  The verified-field precision rests on checks the standards define, not on the layout.
- **A valid ZATCA QR proves what the issuer encoded**, not that the invoice is genuine. Checking the
  phase-2 cryptographic stamp needs ZATCA's certificates and is out of scope.
- MIDV-500: Arlazarov et al., *Computer Optics* 2019. Curated labels and baseline scores: Dynamsoft's
  [MIDV-500 MRZ benchmark](https://github.com/yushulx/python-mrz-scanner-sdk/tree/main/examples/official/benchmark-midv500) (MIT).
- Fonts and their licences: [synth/fonts/LICENSES.md](synth/fonts/LICENSES.md).
