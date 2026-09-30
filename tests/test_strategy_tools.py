"""Regression diff, isomorphism across files, and the command line."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from svlab.cli import main
from svlab.strategy import Thresholds, compare_isomorphic, diff, dumps, load, relabel

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures" / "strategies"
BASE = FIX / "srp_btn_bb_Ks7s2d.json"
TWIN = FIX / "srp_btn_bb_Kh7h2c.json"
MINOR = FIX / "candidate_minor_Ks7s2d.json"
REGRESSED = FIX / "candidate_regressed_Ks7s2d.json"
CONFIG = ROOT / "fixtures" / "diff_thresholds.json"


# ------------------------------------------------------------------- diff


def test_diff_of_the_regressed_candidate():
    report = diff(load(BASE), load(REGRESSED), Thresholds.load(CONFIG))
    assert not report.ok
    assert {v.rule for v in report.violations} == {
        "combos-removed",
        "max-freq-shift",
        "max-ev-shift",
        "max-aggregate-shift",
    }
    btn = next(n for n in report.nodes if n.path == ["check"])
    assert sorted(btn.removed) == ["Ac5c", "Ad5d", "Ah5h", "As5s"]
    before, after = btn.aggregate["bet75"]
    assert after < before - 0.05
    # The nodes that weren't touched didn't move.
    assert all(s.freq_shift == 0 for n in report.nodes if n.path != ["check"] for s in n.shifts)


def test_diff_of_the_minor_candidate_passes_but_not_with_tight_thresholds():
    assert diff(load(BASE), load(MINOR), Thresholds.load(CONFIG)).ok
    tight = diff(load(BASE), load(MINOR), Thresholds(max_freq_shift=0.001))
    assert [v.rule for v in tight.violations] == ["max-freq-shift"] * 3


def test_diff_matches_combos_regardless_of_card_order():
    base = load(BASE)
    node = base.nodes[1]
    flipped = tuple(replace(c, combo=c.combo[2:] + c.combo[:2]) for c in node.combos)
    candidate = replace(base, nodes=(base.nodes[0], replace(node, combos=flipped), base.nodes[2]))
    assert diff(base, candidate).ok


def test_diff_reports_tree_changes():
    base = load(BASE)
    fewer = replace(base, nodes=base.nodes[:2])
    assert [v.rule for v in diff(base, fewer).violations] == ["tree-changed"]
    assert diff(base, fewer, Thresholds(allow_tree_changes=True)).ok


def test_thresholds_config_is_strict():
    with pytest.raises(ValueError, match="unknown threshold 'max_shift'"):
        Thresholds.from_json({"max_shift": 0.1})
    with pytest.raises(ValueError, match="expected a number >= 0"):
        Thresholds.from_json({"max_freq_shift": -1})


# -------------------------------------------------------------------- iso


def test_twin_fixture_is_isomorphic():
    result = compare_isomorphic(load(BASE), load(TWIN))
    assert result.ok and result.perm is not None


def test_iso_needs_isomorphic_boards():
    result = compare_isomorphic(load(BASE), load(FIX / "srp_btn_bb_9h6h3h.json"))
    assert result.perm is None and [f.rule for f in result.findings] == ["iso-board"]


def test_iso_catches_a_combo_that_was_not_relabelled():
    base = load(BASE)
    twin = relabel(base, (1, 0, 3, 2))
    node = twin.nodes[0]
    wrong = (replace(node.combos[10], combo=base.nodes[0].combos[10].combo),) + node.combos[:10] + node.combos[11:]
    twin = replace(twin, nodes=(replace(node, combos=wrong),) + twin.nodes[1:])
    assert "iso-combo" in {f.rule for f in compare_isomorphic(base, twin).findings}


# -------------------------------------------------------------------- CLI


def test_cli_validate(capsys, tmp_path):
    out = tmp_path / "findings.json"
    assert main(["validate", str(BASE), str(FIX / "minimal_Ks7s2d.json"), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["files"][0]["findings"] == []
    broken = tmp_path / "broken.json"
    broken.write_text(BASE.read_text().replace('"bet33", "type": "bet", "size": 1.8', '"bet33", "type": "bet", "size": 180', 1))
    assert main(["validate", str(broken)]) == 1
    assert "[action-legal] nodes[0].actions[1]: size 180 is above the effective stack 97.5" in capsys.readouterr().out


def test_cli_validate_unreadable(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "svlab-strategy/1"}')
    assert main(["validate", str(bad)]) == 2
    assert "(root): missing key 'spot'" in capsys.readouterr().err


def test_cli_iso():
    assert main(["iso", str(BASE), str(TWIN)]) == 0
    assert main(["iso", str(BASE), str(FIX / "srp_btn_bb_9h6h3h.json")]) == 1


def test_cli_diff(tmp_path):
    md, js = tmp_path / "report.md", tmp_path / "report.json"
    assert main(["diff", str(BASE), str(MINOR), "--config", str(CONFIG), "--quiet"]) == 0
    code = main(["diff", str(BASE), str(REGRESSED), "--config", str(CONFIG), "--markdown", str(md), "--json", str(js), "--quiet"])
    assert code == 1
    assert "**Result: FAIL**" in md.read_text()
    assert json.loads(js.read_text())["ok"] is False
    # A flag overrides the config: with generous thresholds the regression passes.
    loose = ["--max-freq-shift", "1", "--max-ev-shift", "10", "--max-aggregate-shift", "1", "--max-combos-removed", "10"]
    assert main(["diff", str(BASE), str(REGRESSED), "--config", str(CONFIG), "--quiet", *loose]) == 0


def test_python_dash_m_entry_point():
    import subprocess
    import sys

    env_path = f"{ROOT / 'src'}"
    result = subprocess.run(
        [sys.executable, "-m", "svlab", "rules"], capture_output=True, text=True, env={"PYTHONPATH": env_path}
    )
    assert result.returncode == 0 and "zero-freq-best-response" in result.stdout


def test_serialised_fixtures_are_byte_stable():
    """The committed files are exactly what the serialiser writes, so a
    regenerated fixture shows up as a clean diff in version control."""
    for path in FIX.glob("*.json"):
        if path.name.startswith("minimal"):
            continue  # hand-written
        assert dumps(load(path)) == path.read_text(), path.name


def test_aggregate_frequencies_can_hide_opposite_per_combo_shifts():
    """Found by Hypothesis: two combos that swap strategies leave every
    range-weighted frequency where it was. Only the per-combo threshold
    sees it, which is why the diff has both."""
    base = load(FIX / "minimal_Ks7s2d.json")
    node = base.nodes[0]
    by_name = {c.combo: c for c in node.combos}
    swapped = tuple(
        replace(c, freqs=by_name["8s6s"].freqs) if c.combo == "Jd9d" else
        replace(c, weight=0.5, freqs=by_name["Jd9d"].freqs) if c.combo == "8s6s" else c
        for c in node.combos
    )
    reweighted = replace(node, combos=tuple(replace(c, weight=0.5) if c.combo == "8s6s" else c for c in node.combos))
    base = replace(base, nodes=(reweighted,))
    candidate = replace(base, nodes=(replace(reweighted, combos=swapped),))
    report = diff(base, candidate)
    assert [v.rule for v in report.violations] == ["max-freq-shift"]
    assert all(abs(b - c) < 1e-12 for b, c in report.nodes[0].aggregate.values())
