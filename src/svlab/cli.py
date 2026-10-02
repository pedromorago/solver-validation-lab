"""Command line: `svlab validate`, `svlab iso`, `svlab diff`, `svlab rules`, `svlab pushfold`, `svlab pushfold3`.

Exit codes: 0 when everything passes, 1 when a rule or threshold fails,
2 when a file can't be read or parsed (or the arguments are wrong).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Sequence

from .strategy import RULES, ParseError, Thresholds, Tolerances, compare_isomorphic, diff, load, validate
from .strategy.combos import perm_text

OK, FAILED, UNREADABLE = 0, 1, 2


def _tolerances(args: argparse.Namespace) -> Tolerances:
    return Tolerances(freq=args.freq_tol, ev=args.ev_tol, weight=args.weight_tol)


def _add_tolerances(p: argparse.ArgumentParser) -> None:
    d = Tolerances()
    p.add_argument("--freq-tol", type=float, default=d.freq, help=f"frequency tolerance (default {d.freq})")
    p.add_argument("--ev-tol", type=float, default=d.ev, help=f"EV tolerance, in the file's units (default {d.ev})")
    p.add_argument("--weight-tol", type=float, default=d.weight, help=f"range weight tolerance (default {d.weight})")


def _write_json(path: str | None, data: object) -> None:
    if path:
        Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _load(path: str):
    try:
        return load(path)
    except ParseError as e:
        print(f"{path}: {e}", file=sys.stderr)
    except (OSError, UnicodeDecodeError) as e:
        print(f"{path}: cannot read: {e}", file=sys.stderr)
    return None


def cmd_validate(args: argparse.Namespace) -> int:
    tol = _tolerances(args)
    status = OK
    report = []
    for path in args.files:
        strategy = _load(path)
        if strategy is None:
            report.append({"file": path, "parsed": False})
            status = UNREADABLE
            continue
        findings = validate(strategy, tol)
        combos = sum(len(n.combos) for n in strategy.nodes)
        if findings:
            status = max(status, FAILED)
            print(f"FAIL {path}: {len(findings)} finding(s)")
            for f in findings[: args.limit]:
                print(f"  {f}")
            if len(findings) > args.limit:
                print(f"  ... and {len(findings) - args.limit} more")
        else:
            print(f"ok   {path}: {len(strategy.nodes)} node(s), {combos} combo(s), {len(RULES)} rules")
        report.append({"file": path, "parsed": True, "findings": [asdict(f) for f in findings]})
    _write_json(args.json, {"tolerances": asdict(tol), "files": report})
    return status


def cmd_iso(args: argparse.Namespace) -> int:
    a, b = _load(args.first), _load(args.second)
    if a is None or b is None:
        return UNREADABLE
    result = compare_isomorphic(a, b, _tolerances(args))
    if result.ok and result.perm is not None:
        print(f"ok   {args.second} is {args.first} with suits relabelled ({perm_text(result.perm)})")
    else:
        mapping = "no relabelling" if result.perm is None else f"best relabelling {perm_text(result.perm)}"
        print(f"FAIL {len(result.findings)} finding(s), {mapping}")
        for f in result.findings[: args.limit]:
            print(f"  {f}")
        if len(result.findings) > args.limit:
            print(f"  ... and {len(result.findings) - args.limit} more")
    _write_json(args.json, {"ok": result.ok, "perm": result.perm, "findings": [asdict(f) for f in result.findings]})
    return OK if result.ok else FAILED


def cmd_diff(args: argparse.Namespace) -> int:
    try:
        thresholds = Thresholds.load(args.config) if args.config else Thresholds()
    except (OSError, ValueError) as e:
        print(f"config: {e}", file=sys.stderr)
        return UNREADABLE
    overrides = {f.name: getattr(args, f.name) for f in fields(Thresholds) if getattr(args, f.name, None) is not None}
    thresholds = Thresholds(**{**asdict(thresholds), **overrides})
    base, cand = _load(args.baseline), _load(args.candidate)
    if base is None or cand is None:
        return UNREADABLE
    report = diff(base, cand, thresholds, names=(args.baseline, args.candidate))
    markdown = report.to_markdown()
    if args.markdown:
        Path(args.markdown).write_text(markdown, encoding="utf-8")
    if not args.quiet:
        print(markdown, end="")
    _write_json(args.json, report.to_json())
    print(f"{'PASS' if report.ok else 'FAIL'}: {len(report.violations)} threshold violation(s)", file=sys.stderr)
    return OK if report.ok else FAILED


def cmd_rules(args: argparse.Namespace) -> int:
    for r in RULES:
        print(f"{r.id:26} {r.summary}")
    return OK


def _grid(values, fmt) -> list[str]:
    """A 13x13 hand grid: pairs on the diagonal, suited above it, offsuit below."""
    from .cards import RANKS
    from .preflop import INDEX

    order = RANKS[::-1]
    rows = ["     " + " ".join(f"{r:>4}" for r in order)]
    for a, hi in enumerate(order):
        cells = []
        for b, lo in enumerate(order):
            name = hi + lo if a == b else (hi + lo + "s" if a < b else lo + hi + "o")
            cells.append(fmt(values[INDEX[name]]))
        rows.append(f"{hi:>4} " + " ".join(cells))
    return rows


def cmd_pushfold(args: argparse.Namespace) -> int:
    from .pushfold import Game, solve

    if args.stack <= 0:
        print("--stack must be positive", file=sys.stderr)
        return UNREADABLE
    game = Game.from_table(args.stack)
    sol = solve(game, tolerance=args.tolerance)

    def pct(x: float) -> str:
        return "   ." if x < 0.005 else " 100" if x > 0.995 else f"{100 * x:4.0f}"

    print(f"Heads-up push/fold, {args.stack:g} bb, blinds 0.5/1, no ante")
    print(f"SB shoves {100 * sol.shove_share(game):.1f}% of hands; BB calls with {100 * sol.call_share(game):.1f}%")
    print(f"SB EV {sol.sb_value:+.4f} bb per hand; Nash gap {sol.nash_gap:.2e} bb after {sol.iterations} iterations")
    for title, values in (("SB shove %", sol.shove), ("BB call %", sol.call)):
        print(f"\n{title}")
        print("\n".join(_grid(values, pct)))
    _write_json(args.json, sol.to_json())
    return OK if sol.nash_gap <= args.tolerance else FAILED


def cmd_pushfold3(args: argparse.Namespace) -> int:
    from .pushfold3 import DECISIONS, Game3, solve

    if args.stack <= 0:
        print("--stack must be positive", file=sys.stderr)
        return UNREADABLE
    try:
        game = Game3.from_table(args.stack)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return UNREADABLE
    sol = solve(game, tolerance=args.tolerance, max_iterations=args.max_iterations)

    def pct(x: float) -> str:
        return "   ." if x < 0.005 else " 100" if x > 0.995 else f"{100 * x:4.0f}"

    titles = {
        "btn": "BTN shove %",
        "sb_vs_shove": "SB call % against the BTN's shove",
        "sb_open": "SB shove % after the BTN folds",
        "bb_vs_btn": "BB call % against the BTN's shove (SB folded)",
        "bb_vs_both": "BB call % against the BTN's shove and the SB's call",
        "bb_vs_sb": "BB call % against the SB's shove (BTN folded)",
    }
    print(f"Three-handed push/fold, {args.stack:g} bb each, blinds 0.5/1, no ante")
    print("EV per hand: " + ", ".join(f"{p} {v:+.4f}" for p, v in sol.values.items()))
    print(f"NashConv {sol.nash_conv:.2e} bb after {sol.iterations} iterations")
    for d in DECISIONS:
        print(f"\n{titles[d]}")
        print("\n".join(_grid(sol.strategy[d], pct)))
    _write_json(args.json, sol.to_json(game))
    return OK if sol.nash_conv <= args.tolerance else FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="svlab", description="Validate and compare exported solver strategies.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check strategy files against every rule")
    p.add_argument("files", nargs="+")
    _add_tolerances(p)
    p.add_argument("--json", help="also write the findings to this file")
    p.add_argument("--limit", type=int, default=20, help="findings to print per file (default 20)")
    p.set_defaults(run=cmd_validate)

    p = sub.add_parser("iso", help="check that two files are the same strategy with suits relabelled")
    p.add_argument("first")
    p.add_argument("second")
    _add_tolerances(p)
    p.add_argument("--json")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(run=cmd_iso)

    p = sub.add_parser("diff", help="regression diff from a baseline to a candidate")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--config", help="JSON file of thresholds; flags override it")
    d = Thresholds()
    for f in fields(Thresholds):
        flag = "--" + f.name.replace("_", "-")
        if f.name == "allow_tree_changes":
            p.add_argument(flag, action="store_true", default=None, help="don't fail on added/removed nodes or actions")
        else:
            kind = int if isinstance(getattr(d, f.name), int) else float
            p.add_argument(flag, type=kind, default=None, help=f"default {getattr(d, f.name)}")
    p.add_argument("--markdown", help="write the Markdown report to this file")
    p.add_argument("--json", help="write the JSON report to this file")
    p.add_argument("--quiet", action="store_true", help="don't print the report")
    p.set_defaults(run=cmd_diff)

    p = sub.add_parser("pushfold", help="solve heads-up push/fold at one stack depth")
    p.add_argument("--stack", type=float, required=True, help="effective stack in big blinds")
    p.add_argument("--tolerance", type=float, default=1e-6, help="Nash gap to reach, in bb per hand (default 1e-6)")
    p.add_argument("--json", help="also write the solution to this file")
    p.set_defaults(run=cmd_pushfold)

    p = sub.add_parser("pushfold3", help="solve three-handed push/fold (BTN, SB, BB) at one stack depth")
    p.add_argument("--stack", type=float, required=True, help="every player's stack in big blinds")
    p.add_argument("--tolerance", type=float, default=1e-5, help="NashConv to reach, in bb per hand (default 1e-5)")
    p.add_argument("--max-iterations", type=int, default=5000)
    p.add_argument("--json", help="also write the solution to this file")
    p.set_defaults(run=cmd_pushfold3)

    p = sub.add_parser("rules", help="list the validation rules")
    p.set_defaults(run=cmd_rules)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
