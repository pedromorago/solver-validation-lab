"""Cards as small integers: rank * 4 + suit.

Ranks run from 0 (deuce) to 12 (ace); suits are c, d, h, s in that order.
"""

RANKS = "23456789TJQKA"
SUITS = "cdhs"

FULL_DECK: tuple[int, ...] = tuple(range(52))


def rank(card: int) -> int:
    return card >> 2


def suit(card: int) -> int:
    return card & 3


def parse(text: str) -> int:
    """Parses one card such as "Ah" or "td"."""
    if len(text) != 2 or text[0].upper() not in RANKS or text[1].lower() not in SUITS:
        raise ValueError(f"Not a card: {text!r}")
    return RANKS.index(text[0].upper()) * 4 + SUITS.index(text[1].lower())


def parse_many(text: str) -> list[int]:
    """Parses "AhKh" or "Ah Kh" into a list of cards, rejecting duplicates."""
    compact = text.replace(" ", "")
    if len(compact) % 2:
        raise ValueError(f"Odd number of characters in {text!r}")
    cards = [parse(compact[i : i + 2]) for i in range(0, len(compact), 2)]
    if len(set(cards)) != len(cards):
        raise ValueError(f"Duplicate card in {text!r}")
    return cards


def card_str(card: int) -> str:
    return RANKS[rank(card)] + SUITS[suit(card)]
