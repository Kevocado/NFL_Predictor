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
from pathlib import Path

import pandas as pd

from ..evaluate.walk_forward import calibration_report
from ..models.prop_probability import american_to_breakeven
from . import store

#: Breakeven for a -110 prop. Stated as a constant because the spec's gates are
#: written against it.
BREAKEVEN = american_to_breakeven(-110)

#: Under this many graded picks, a hit rate is noise with a decimal point.
THIN_SAMPLE = 50

_FORWARD_COLUMNS = ["game_id", "player_id", "player_name", "market", "side",
                    "line_at_snapshot", "odds_at_snapshot", "model_p_over",
                    "edge_vs_breakeven", "closing_line", "clv", "hit",
                    "actual_value", "resolved"]


def graded_picks(season: int | None = None, week: int | None = None) -> pd.DataFrame:
    """Forward-test rows: those carrying a snapshot line, which is what
    distinguishes them from the pre-existing yardage projections.

    `player_prop_predictions` has no season or week column -- it is keyed by
    game_id -- so the scope comes from joining `game_predictions`, which carries
    both. Skipping that join is how a weekly report ends up rendering the
    all-time record under this week's filename, which is the worst shape of wrong
    for the one artifact a human reads.
    """
    with contextlib.closing(store._connect()) as conn:
        frame = pd.read_sql(
            """
            SELECT p.*, g.season AS season, g.week AS week
            FROM player_prop_predictions AS p
            LEFT JOIN game_predictions AS g ON g.game_id = p.game_id
            WHERE p.line_at_snapshot IS NOT NULL
            """,
            conn,
        )
    if season is not None and "season" in frame.columns:
        frame = frame[frame["season"] == season]
    if week is not None and "week" in frame.columns:
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
        lines += ["No picks this week: nothing cleared the 5% edge gate.", ""]
        return "\n".join(lines) + "\n"
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
    scored = frame[frame["model_p_over"].notna() & frame["hit"].notna()]
    buckets = calibration_report(
        scored.rename(columns={"model_p_over": "p_over", "hit": "covered"}), min_n=1)
    if buckets:
        lines += ["## P(over) calibration", "",
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