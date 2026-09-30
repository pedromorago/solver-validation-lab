"""Fixed situations the checks run on. Chosen to cover draws, made hands,
flushes, the wheel, dominated hands, three-way pots and splits."""

from svlab import parse_many as P

# (name, hands, board)
FLOP_SPOTS = [
    ("flush draw vs overpair", [P("AhKh"), P("QsQd")], P("Jh Th 2c")),
    ("open-ender vs top pair", [P("9c8c"), P("AsJd")], P("Jh Td 7s")),
    ("set vs overpair", [P("7d7c"), P("AhAc")], P("7s Kd 2h")),
    ("dominated kicker", [P("AsKd"), P("AcQh")], P("Ah 8c 3d")),
    ("wheel draw", [P("As2s"), P("KdKh")], P("3c 4d 9h")),
    ("same ranks, one backdoor flush", [P("2c3d"), P("2h3s")], P("Ah Kh Qh")),
    ("three-way", [P("AhKh"), P("QsQd"), P("9c8c")], P("Jh Td 2c")),
]

TURN_SPOTS = [
    ("flush draw vs overpair, turn", [P("AhKh"), P("QsQd")], P("Jh Th 2c 5d")),
    ("gutshot vs two pair, turn", [P("Qc9c"), P("JsTc")], P("Jh Td 3s 2h")),
]

RIVER_SPOTS = [
    ("river win", [P("AhKh"), P("QsQd")], P("Jh Th 2c 5d 3h")),
    ("river split", [P("AhKd"), P("AcKs")], P("Qh Jd Ts 4c 2s")),
]
