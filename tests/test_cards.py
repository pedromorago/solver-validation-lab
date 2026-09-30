import pytest

from svlab import card_str, parse, parse_many


def test_round_trip_for_every_card():
    for card in range(52):
        assert parse(card_str(card)) == card


@pytest.mark.parametrize("bad", ["", "A", "1h", "Ax", "Ahh", "10h"])
def test_rejects_malformed_cards(bad):
    with pytest.raises(ValueError):
        parse(bad)


def test_rejects_duplicates():
    with pytest.raises(ValueError):
        parse_many("AhAh")
