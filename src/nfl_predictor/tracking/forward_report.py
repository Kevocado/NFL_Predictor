"""forward_report.py -- the weekly markdown record. This is the artifact Kevin
reads, not a dashboard.

It reports the record without flattering it: the hit rate against the 52.4%
breakeven, mean CLV with its sign, P(over) calibration buckets, and a one-line
verdict against the spec's gates. Picks and results are counted separately -- a
snapshotted-but-unresolved row is a pick in the log and not yet a result, and
folding it in as a miss would understate the hit rate.

Below `THIN_SAMPLE` picks a hit rate is not a measurement, and the report says
so instead of printing a confident percentage.
"""
from __future__ import annotations

import contextlib
import re
from pathlib import Path

import pandas as pd

from ..evaluate.walk_forward import calibration_report
from ..models.prop_probability import american_to_breakeven
from . import forward_tick, store

#: Breakeven for a -110 prop. Stated as a constant because the spec's gates are
#: written against it.
BREAKEVEN = american_to_breakeven(-110)

#: The confidence gate `forward_tick` logs a pick at, named here so the report
#: states the threshold that produced the log instead of implying a market edge.
#: Read from the tick rather than restated, for the reason `FORWARD_FEATURE_COLUMNS`
#: is imported rather than redeclared. Safe at module level: `forward_tick` imports
#: THIS module only inside `main()`, so there is no cycle -- and no try/except
#: either, because a fallback of 0.05 would silently misreport the threshold if the
#: import ever did break.
GATE = forward_tick.EDGE_GATE

#: Under this many graded picks, a hit rate is noise with a decimal point.
THIN_SAMPLE = 50

_FORWARD_COLUMNS = ["game_id", "player_id", "player_name", "market", "side",
                    "line_at_snapshot", "odds_at_snapshot", "model_p_over",
                    "edge_vs_breakeven", "closing_line", "clv", "hit",
                    "actual_value", "resolved"]


#: nflverse game ids are `<season>_<week>_<AWAY>_<HOME>` (e.g. `2026_05_TB_DAL`).
#: The season and week live in the id itself.
_GAME_ID = re.compile(r"^(?P<season>\d{4})_(?P<week>\d{2})_")


def graded_picks(season: int | None = None, week: int | None = None) -> pd.DataFrame:
    """Forward-test rows: those carrying a snapshot line, which is what
    distinguishes them from the pre-existing yardage projections.

    `player_prop_predictions` has no season or week column -- it is keyed by
    game_id -- so the scope is recovered from the game id, which encodes both
    (`2026_05_TB_DAL`).

    Deriving it from the id rather than joining `game_predictions` is not a
    shortcut. A standalone forward tick never writes `game_predictions` rows --
    it has no win probabilities to write, since those come from the game-level
    model -- so the join yields NULL season/week for every forward pick and the
    report renders an EMPTY week while the picks sit in the table. That is the
    worst shape of wrong for the one artifact a human reads, and it is silent.

    The join is kept as a fallback for any row whose id is not in nflverse form.
    """
    with contextlib.closing(store._connect()) as conn:
        frame = pd.read_sql(
            """
            SELECT p.*, g.season AS joined_season, g.week AS joined_week
            FROM player_prop_predictions AS p
            LEFT JOIN game_predictions AS g ON g.game_id = p.game_id
            WHERE p.line_at_snapshot IS NOT NULL
            """,
            conn,
        )

    parsed = frame["game_id"].astype(str).str.extract(_GAME_ID)
    frame["season"] = pd.to_numeric(parsed["season"], errors="coerce")
    frame["week"] = pd.to_numeric(parsed["week"], errors="coerce")
    # Fall back to the join only where the id is not nflverse-shaped.
    frame["season"] = frame["season"].fillna(pd.to_numeric(frame["joined_season"], errors="coerce"))
    frame["week"] = frame["week"].fillna(pd.to_numeric(frame["joined_week"], errors="coerce"))
    frame = frame.drop(columns=["joined_season", "joined_week"])

    if season is not None:
        frame = frame[frame["season"] == season]
    if week is not None:
        frame = frame[frame["week"] == week]
    return frame.reset_index(drop=True)


def _fmt(value, spec="{:.1f}") -> str:
    return "n/a" if value is None or pd.isna(value) else spec.format(value)


def _hit_rate(frame: pd.DataFrame) -> tuple[float | None, int]:
    graded = frame[frame["hit"].notna()]
    if graded.empty:
        return None, 0
    return float(graded["hit"].mean()), len(graded)


def _mean_clv(frame: pd.DataFrame) -> float | None:
    values = frame["clv"].dropna()
    return None if values.empty else float(values.mean())


def _verdict(frame: pd.DataFrame, rate: float | None, graded: int,
             clv: float | None) -> str:
    if rate is None:
        return "**Verdict:** no graded picks yet -- nothing to judge."
    if graded < THIN_SAMPLE:
        return (f"**Verdict:** {graded} graded picks, below the {THIN_SAMPLE} the spec's "
                f"forward gate needs. Not yet a measurement; keep logging.")
    if rate > BREAKEVEN:
        direction = "positive" if (clv or 0) >= 0 else "negative"
        return (f"**Verdict:** {rate:.1%} against a {BREAKEVEN:.1%} breakeven, "
                f"mean CLV {direction}. Holding above breakeven.")
    return (f"**Verdict:** {rate:.1%} against a {BREAKEVEN:.1%} breakeven -- below "
            f"breakeven on {graded} picks. Per the spec's kill rule, this is the point "
            f"to stop treating yardage props as a betting edge.")


def _picks_table(frame: pd.DataFrame) -> list[str]:
    """Every logged pick, with the number named for what it is.

    Without this the log is a count and the numbers only exist in the database,
    where `edge_vs_breakeven` reads as an inefficiency found in the market. It is
    not: it is P(side) minus breakeven, so both figures are shown side by side
    and the column is called confidence.
    """
    ordered = frame.sort_values("edge_vs_breakeven", ascending=False)
    lines = [
        "## Picks",
        "",
        "| player | market | side | line | P(side) | confidence vs breakeven | graded |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in ordered.itertuples():
        side = getattr(row, "side", None)
        # `model_p_over` is a misnomer in the schema: `_row` writes P(side taken)
        # into it, and `hit` is likewise side-relative (`_forward_verdict` covers
        # an under when actual < line). The two are therefore consistent with each
        # other and the calibration buckets below are sound; only the NAME lies.
        # Correcting the name here rather than migrating stored rows, which are
        # immutable by rule.
        p_side = getattr(row, "model_p_over", None)
        hit = getattr(row, "hit", None)
        lines.append(
            f"| {row.player_name} | {row.market} | {side} | "
            f"{_fmt(getattr(row, 'line_at_snapshot', None))} | "
            f"{_fmt(p_side, '{:.1%}')} | "
            f"{_fmt(getattr(row, 'edge_vs_breakeven', None), '{:+.1%}')} | "
            f"{'n/a' if hit is None or pd.isna(hit) else int(hit)} |")
    lines.append("")
    return lines


def render(season: int, week: int, frame: pd.DataFrame) -> str:
    picks = len(frame)
    rate, graded = _hit_rate(frame)
    clv = _mean_clv(frame)

    lines = [
        f"# Forward test -- {season} week {week}",
        "",
        f"- Picks logged: **{picks}**",
        f"- Graded: **{graded}**",
        f"- Hit rate: **{'n/a' if rate is None else f'{rate:.1%}'}** "
        f"(breakeven {BREAKEVEN:.1%} at -110)",
        f"- Mean CLV: **{_fmt(clv, '{:+.2f}')} yards**",
        "",
    ]

    if picks == 0:
        lines += [f"No picks this week: nothing cleared the {GATE:.0%} confidence gate.", ""]
        return "\n".join(lines) + "\n"

    lines += _picks_table(frame)
    lines += [
        "Confidence is the model's probability for the side taken, minus the "
        f"breakeven implied by the price. A pick is logged at {GATE:.0%} confidence "
        "against breakeven or better, and the table is ranked by it. It is a "
        "statement about the MODEL, not a demonstrated mispricing in the book's "
        "line: a large number means the model disagrees with the price, which is a "
        "hypothesis. What would settle it is beating the close, and that is "
        "measured separately by CLV.",
        "",
    ]

    if graded == 0:
        lines += ["Picks are logged but none are graded yet, so there is no hit rate "
                  "to report this week.", ""]
        lines.append(_verdict(frame, rate, graded, clv))
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.append("## By side")
    lines.append("")
    lines.append("| side | picks | graded | hit rate | mean CLV |")
    lines.append("|---|---|---|---|---|")
    for side in ("over", "under"):
        subset = frame[frame["side"] == side]
        side_rate, side_graded = _hit_rate(subset)
        lines.append(
            f"| {side} | {len(subset)} | {side_graded} | "
            f"{'n/a' if side_rate is None else f'{side_rate:.1%}'} | "
            f"{_fmt(_mean_clv(subset), '{:+.2f}')} |")
    lines.append("")

    # `calibration_report` reads the offline harness's column names (p_over /
    # covered); the store's are model_p_over / hit. Renamed rather than given a
    # second bucketing path, so both callers bucket identically.
    #
    # Both sides are side-relative (see `_picks_table`), so this is a
    # P(side)-vs-covered calibration, NOT a P(over) one. An earlier heading said
    # P(over) and was wrong for every under pick in the log.
    scored = frame[frame["model_p_over"].notna() & frame["hit"].notna()]
    buckets = calibration_report(
        scored.rename(columns={"model_p_over": "p_over", "hit": "covered"}), min_n=1)
    if buckets:
        lines += ["## P(side) calibration", "",
                  "| bucket | predicted | empirical | n | within +/-5pts |",
                  "|---|---|---|---|---|"]
        for label, bucket in buckets.items():
            lines.append(
                f"| {label} | {bucket['predicted']:.3f} | {bucket['empirical']:.3f} | "
                f"{bucket['n']} | {'yes' if bucket['within_tolerance'] else 'NO'} |")
        lines.append("")

    lines.append(_verdict(frame, rate, graded, clv))
    lines.append("")
    return "\n".join(lines)


def write_weekly_report(season: int, week: int, out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{season}-W{week}.md"
    path.write_text(render(season, week, graded_picks(season=season, week=week)))
    return path