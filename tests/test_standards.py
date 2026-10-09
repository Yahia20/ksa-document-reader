"""Unit tests for the three standards the reader relies on: ICAO MRZ, ZATCA QR, IATA BCBP."""
from decimal import Decimal

import pytest

from docreader import bcbp, mrz, zatca

# ICAO Doc 9303 Part 4 specimen (Utopia passport, Anna Maria Eriksson).
ICAO_TD3 = ["P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10"]
# ICAO Doc 9303 Part 5 specimen TD1.
ICAO_TD1 = ["I<UTOD231458907<<<<<<<<<<<<<<<",
            "7408122F1204159UTO<<<<<<<<<<<6",
            "ERIKSSON<<ANNA<MARIA<<<<<<<<<<"]


def test_check_digit_examples():
    assert mrz.check_digit("L898902C3") == "6"
    assert mrz.check_digit("740812") == "2"
    assert mrz.check_digit("120415") == "9"


@pytest.mark.parametrize("lines,layout", [(ICAO_TD3, "TD3"), (ICAO_TD1, "TD1")])
def test_icao_specimens_parse_and_verify(lines, layout):
    res = mrz.find_mrz(lines)
    assert res.layout == layout
    assert res.composite_ok is True
    assert res.fields["surname"].value == "ERIKSSON"
    assert res.fields["given_names"].value == "ANNA MARIA"
    assert res.fields["birth_date"].value == "1974-08-12"
    for name in ("birth_date", "expiry_date"):
        assert res.fields[name].status == "verified"
    # Letters next to digits: the check digit passes but cannot rule out a G/6-type
    # swap, so the number waits for the printed copy to confirm it.
    assert res.fields["document_number"].status == "review"


def test_check_digit_is_blind_to_ten_apart_twins():
    # G = 16 and 6 = 6 differ by 10, so every 7-3-1 weighted sum keeps its last digit.
    assert mrz.check_digit("XK7347G47") == mrz.check_digit("XK7347647")
    assert "XK7347647" in mrz.blind_alternatives("XK7347G47")
    # A digit-for-digit misread is always caught.
    assert mrz.check_digit("123456789") != mrz.check_digit("123456780")


def test_letter_digit_confusion_in_dates_is_repaired():
    noisy = [ICAO_TD3[0], ICAO_TD3[1].replace("7408122", "74O8l22")]
    res = mrz.find_mrz(noisy)
    assert res.lines == ICAO_TD3
    assert res.fields["birth_date"].status == "verified"


def test_wrong_digit_is_caught_not_verified():
    bad = [ICAO_TD3[0], ICAO_TD3[1].replace("7408122", "7408132")]
    res = mrz.find_mrz(bad)
    assert res.fields["birth_date"].status == "review"


def test_document_number_swap_is_never_auto_verified():
    # 'O' read for '0' inside an alphanumeric number. Here several swaps (8/B, 2/Z, O/0)
    # pass the check digit, so no single fix can be trusted.
    noisy = [ICAO_TD3[0], ICAO_TD3[1].replace("L898902C3", "L8989O2C3")]
    res = mrz.find_mrz(noisy)
    assert res.fields["document_number"].status == "review"


def test_blind_twin_misread_goes_to_review_with_the_fix_offered():
    line2 = "XK73476479UTO7408122F1204159<<<<<<<<<<<<<<<2"
    res = mrz.find_mrz([ICAO_TD3[0], line2.replace("XK7347647", "XK7347G47")])
    f = res.fields["document_number"]
    assert f.status == "review"
    assert "XK7347647" in f.alternatives


def test_filler_miscounted_by_ocr():
    shorter = [ICAO_TD3[0].replace("<<<<<<", "<<<<<", 1), ICAO_TD3[1]]
    res = mrz.find_mrz(shorter)
    assert res.lines == ICAO_TD3


def test_mrz_split_across_two_ocr_lines():
    res = mrz.find_mrz([ICAO_TD3[0][:20], ICAO_TD3[0][20:], ICAO_TD3[1]])
    assert res.lines == ICAO_TD3


def test_partial_birth_date_is_allowed():
    line2 = "1221035125DZA83<<<<5M1705058X162<<<<<<<<<<48"  # MIDV-500 Algerian specimen
    data = line2[13:19]
    assert mrz.is_valid_check(data, line2[19])
    assert mrz._parse_date(data, future=False) == "1983-??-??"


# ZATCA example QR from the e-invoicing guidance (seller "Salla", tags 1-5).
ZATCA_SAMPLE = "AQVTYWxsYQIPMzEwMTIyMzkzNTAwMDAzAxQyMDIyLTA0LTI1VDE1OjMwOjAwWgQHMTAwMC4wMAUGMTUwLjAw"


def test_zatca_round_trip():
    payload = zatca.encode_tlv({1: "شركة الواحة للسفر", 2: "310122393500003", 3: "2026-03-14T10:22:31Z",
                                4: "1150.00", 5: "150.00"})
    qr = zatca.parse_qr(payload)
    assert qr.seller_name == "شركة الواحة للسفر"
    assert qr.total == Decimal("1150.00") and qr.vat_total == Decimal("150.00")
    assert qr.phase == 1


def test_zatca_public_sample():
    qr = zatca.parse_qr(ZATCA_SAMPLE)
    assert qr.seller_name == "Salla"
    assert qr.seller_vat == "310122393500003"
    assert zatca.valid_vat_number(qr.seller_vat)


def test_zatca_rejects_ordinary_qr():
    with pytest.raises(ValueError):
        zatca.parse_qr("https://example.com/account/123")


def test_vat_number_rule():
    assert zatca.valid_vat_number("300000000000003")
    assert not zatca.valid_vat_number("300000000000004")
    assert not zatca.valid_vat_number("30000000000003")


def test_bcbp_round_trip():
    bp = bcbp.BoardingPass("ALQAHTANI/FAHAD", "ABC123", "RUH", "JED", "RQ", "0123 ", 294, "Y", "012A", "0045 ")
    text = bcbp.build(bp)
    assert len(text) == bcbp.MANDATORY_LEN
    back = bcbp.parse(text)
    assert (back.passenger_name, back.flight_number, back.seat, back.sequence) == ("ALQAHTANI/FAHAD", "123", "12A", "45")
    assert back.flight_date(2026).isoformat() == "2026-10-21"


def test_misread_filler_after_given_names_is_dropped():
    line1 = "PCAZEHUSEYNLI<<ORKHAN<<<<E<<<<<<<<<<<<<<<<<<"  # MIDV-500 frame CA05_18, '<' read as E
    line2 = "X110003442AZE7503153M230801030LJV5Z<<<<<<<86"
    res = mrz.find_mrz([line1, line2])
    assert res.fields["given_names"].value == "ORKHAN"


def test_two_misreads_can_cancel_in_the_check_digit():
    # MIDV-500 frame CA28_13: "BD0002028" read as "800002028". B->8 and D->0 shift the
    # weighted sum by -21 and -39, together -60, so the check digit still passes. This is
    # why the reader also wants the printed number before it calls a document number verified.
    assert mrz.check_digit("BD0002028") == mrz.check_digit("800002028")
