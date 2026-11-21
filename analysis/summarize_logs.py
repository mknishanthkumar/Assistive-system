"""
analysis/summarize_logs.py - Assistive Control System log summariser.

Usage:
  python analysis/summarize_logs.py logfile.csv [logfile2.csv ...]
  python analysis/summarize_logs.py --all        # all files in logs/runs/
  python analysis/summarize_logs.py --last N     # last N files sorted by name
"""

import csv
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_RUNS_DIR = _PROJECT_ROOT / "logs" / "runs"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class _ConditionStats:
    def __init__(self):
        self.task_seconds = 0.0      # sum of (last_stop - first_start) per file
        self.file_count = 0          # number of files contributing to this condition
        self.switch_count = 0
        self.wrong_switch_count = 0
        self.override_count = 0
        self.undo_count = 0

    def add_file(self, fs):
        """Merge a _FileStats object into this condition bucket."""
        if fs.task_seconds is not None:
            self.task_seconds += fs.task_seconds
        self.file_count += 1
        self.switch_count += fs.switch_count
        self.wrong_switch_count += fs.wrong_switch_count
        self.override_count += fs.override_count
        self.undo_count += fs.undo_count


class _FileStats:
    def __init__(self):
        self.total_rows = 0
        self.task_seconds = None      # None if no start/stop pair found
        self.switch_count = 0
        self.wrong_switch_count = 0
        self.override_count = 0
        self.undo_count = 0
        self.conditions_seen = set()  # all study_condition values in this file


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_ts(s):
    """Parse ISO timestamp string; return datetime or None."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _condition_label(row):
    """Return the study_condition for a row, or '(none)' if absent/blank."""
    val = row.get("study_condition", "").strip()
    return val if val else "(none)"


def _fmt_duration(seconds):
    """Format a float of seconds as 'Xm Ys'."""
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}m {s:02d}s"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_file(path: Path) -> tuple[_FileStats, dict]:
    """
    Parse one CSV file.
    Returns (_FileStats, condition_rows_dict) where condition_rows_dict maps
    condition_label -> list of parsed rows (dicts).
    """
    fs = _FileStats()
    rows_by_condition = defaultdict(list)

    # We need two passes conceptually, but to stay memory-light we do one pass
    # collecting rows, then post-process.
    all_rows = []

    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            try:
                row = dict(raw)
                fs.total_rows += 1
                ts = _parse_ts(row.get("timestamp_iso", ""))
                row["_ts"] = ts
                cond = _condition_label(row)
                row["_cond"] = cond
                fs.conditions_seen.add(cond)
                all_rows.append(row)
            except Exception:
                continue  # skip malformed rows

    if not all_rows:
        return fs, {}

    # ---- task time: first "start" → last "stop" ----
    start_ts = None
    stop_ts = None
    for row in all_rows:
        if row.get("event_type") == "start" and row["_ts"] is not None:
            if start_ts is None:
                start_ts = row["_ts"]
        if row.get("event_type") == "stop" and row["_ts"] is not None:
            stop_ts = row["_ts"]
    if start_ts is not None and stop_ts is not None and stop_ts > start_ts:
        fs.task_seconds = (stop_ts - start_ts).total_seconds()

    # ---- collect switch events (decision rows mentioning "MODE") ----
    switch_events = []  # list of (timestamp, row_index)
    for i, row in enumerate(all_rows):
        et = row.get("event_type", "")
        msg = row.get("message", "")
        if et == "decision" and "MODE" in msg.upper():
            fs.switch_count += 1
            if row["_ts"] is not None:
                switch_events.append((row["_ts"], i))

    # ---- override count ----
    for row in all_rows:
        et = row.get("event_type", "")
        msg = row.get("message", "")
        is_override_field = row.get("is_override", "").strip().lower() in ("true", "1", "yes")
        if et == "hotkey" or "manual override" in msg.lower() or is_override_field:
            fs.override_count += 1

    # ---- undo count ----
    for row in all_rows:
        if row.get("event_type") == "undo":
            fs.undo_count += 1
            # also count is_undo flag rows that aren't event_type="undo"
        elif row.get("is_undo", "").strip().lower() in ("true", "1", "yes"):
            if row.get("event_type") != "undo":
                fs.undo_count += 1

    # ---- wrong switch: switch followed within 5s by hotkey/undo/override ----
    # Build a set of timestamps of "corrective" actions
    corrective_ts = []
    for row in all_rows:
        et = row.get("event_type", "")
        is_hotkey = row.get("is_hotkey", "").strip().lower() in ("true", "1", "yes")
        is_undo = row.get("is_undo", "").strip().lower() in ("true", "1", "yes")
        is_override_field = row.get("is_override", "").strip().lower() in ("true", "1", "yes")
        msg = row.get("message", "")
        if et in ("hotkey", "undo") or is_hotkey or is_undo or is_override_field \
                or "manual override" in msg.lower():
            if row["_ts"] is not None:
                corrective_ts.append(row["_ts"])

    for sw_ts, _idx in switch_events:
        # Check if any corrective action falls within 5 seconds after the switch
        window_end = sw_ts + timedelta(seconds=5)
        for c_ts in corrective_ts:
            if sw_ts < c_ts <= window_end:
                fs.wrong_switch_count += 1
                break  # only count once per switch

    # ---- group rows by condition ----
    for row in all_rows:
        rows_by_condition[row["_cond"]].append(row)

    return fs, dict(rows_by_condition)


def analyse_files(paths):
    """
    Analyse a list of CSV paths.
    Returns (overall_stats_dict, condition_stats_dict, total_events, total_files).
    """
    condition_map = defaultdict(_ConditionStats)  # cond -> _ConditionStats
    total_events = 0
    total_task_seconds = 0.0
    files_ok = 0

    for path in paths:
        try:
            fs, _rows_by_cond = analyse_file(path)
        except Exception as exc:
            print(f"[warn] Could not read {path}: {exc}", file=sys.stderr)
            continue

        total_events += fs.total_rows
        if fs.task_seconds is not None:
            total_task_seconds += fs.task_seconds
        files_ok += 1

        # Bucket per condition.  A file may have multiple conditions; we
        # attribute the whole file's aggregates to each condition proportionally
        # — but the brief says "per condition" for the breakdown table.
        # Simpler & more faithful: attribute whole file stats to each distinct
        # condition found, counting file once per condition.
        # Actually, the clearest approach matching the spec: collect per-
        # condition stats by looking at rows within each condition bucket.
        for cond, rows in _rows_by_cond.items():
            cs = condition_map[cond]
            # re-derive per-condition stats from row subset
            # task time: not easily split per condition; attribute full file time
            # to the dominant condition (or "(none)")
            cs.switch_count += sum(
                1 for r in rows
                if r.get("event_type") == "decision" and "MODE" in r.get("message", "").upper()
            )
            cs.wrong_switch_count += _wrong_switches_in_rows(rows)
            cs.override_count += sum(
                1 for r in rows
                if r.get("event_type") == "hotkey"
                or "manual override" in r.get("message", "").lower()
                or r.get("is_override", "").strip().lower() in ("true", "1", "yes")
            )
            cs.undo_count += sum(
                1 for r in rows
                if r.get("event_type") == "undo"
                or (r.get("is_undo", "").strip().lower() in ("true", "1", "yes")
                    and r.get("event_type") != "undo")
            )
            # file count per condition
            cs.file_count += 1

        # Task time: attribute to the most-common condition in this file, or
        # "(none)" if no condition.
        if fs.task_seconds is not None:
            dominant = _dominant_condition(list(_rows_by_cond.keys()), _rows_by_cond)
            condition_map[dominant].task_seconds += fs.task_seconds

    return condition_map, total_events, total_task_seconds, files_ok


def _wrong_switches_in_rows(rows):
    """Count wrong switches in a subset of rows (same logic as analyse_file)."""
    switch_events = []
    for row in rows:
        if row.get("event_type") == "decision" and "MODE" in row.get("message", "").upper():
            if row["_ts"] is not None:
                switch_events.append(row["_ts"])

    corrective_ts = []
    for row in rows:
        et = row.get("event_type", "")
        is_hotkey = row.get("is_hotkey", "").strip().lower() in ("true", "1", "yes")
        is_undo = row.get("is_undo", "").strip().lower() in ("true", "1", "yes")
        is_override = row.get("is_override", "").strip().lower() in ("true", "1", "yes")
        msg = row.get("message", "")
        if et in ("hotkey", "undo") or is_hotkey or is_undo or is_override \
                or "manual override" in msg.lower():
            if row["_ts"] is not None:
                corrective_ts.append(row["_ts"])

    count = 0
    for sw_ts in switch_events:
        window_end = sw_ts + timedelta(seconds=5)
        for c_ts in corrective_ts:
            if sw_ts < c_ts <= window_end:
                count += 1
                break
    return count


def _dominant_condition(conds, rows_by_cond):
    """Return the condition with the most rows."""
    if not conds:
        return "(none)"
    return max(conds, key=lambda c: len(rows_by_cond.get(c, [])))


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def _pct(num, denom):
    if denom == 0:
        return " (0%)"
    return f" ({round(100 * num / denom)}%)"


def print_report(paths, condition_map, total_events, total_task_seconds, files_ok):
    # ---- Header ----
    print("=== Assistive System Log Summary ===")
    print(f"Files analysed: {files_ok}")
    print(f"Total duration: {_fmt_duration(total_task_seconds)}")
    print(f"Total events:   {total_events}")
    print()

    # ---- Per-condition breakdown ----
    print("--- Per-condition breakdown ---")
    # Column widths
    cw = max(16, *([len(c) for c in condition_map] or [0]))
    hdr = (f"{'Condition':<{cw}} | {'Tasks':>5} | {'Duration':>9} | "
           f"{'Switches':>8} | {'Wrong-switch':>14} | {'Override':>8} | {'Undo':>4}")
    print(hdr)
    print("-" * len(hdr))

    # Preferred order
    ORDER = ["face_only", "speech_only", "manual_switching", "auto_switching"]
    ordered_conds = [c for c in ORDER if c in condition_map]
    for c in sorted(condition_map.keys()):
        if c not in ordered_conds:
            ordered_conds.append(c)

    total_switches = 0
    total_wrong = 0
    total_overrides = 0
    total_undos = 0

    for cond in ordered_conds:
        cs = condition_map[cond]
        total_switches += cs.switch_count
        total_wrong += cs.wrong_switch_count
        total_overrides += cs.override_count
        total_undos += cs.undo_count
        ws_str = f"{cs.wrong_switch_count}{_pct(cs.wrong_switch_count, cs.switch_count)}"
        dur = _fmt_duration(cs.task_seconds)
        print(
            f"{cond:<{cw}} | {cs.file_count:>5} | {dur:>9} | "
            f"{cs.switch_count:>8} | {ws_str:>14} | {cs.override_count:>8} | {cs.undo_count:>4}"
        )

    print()

    # ---- Overall ----
    print("--- Overall ---")
    ws_overall = f"{round(100 * total_wrong / total_switches)}%" if total_switches else "0%"
    print(f"Task time:         {_fmt_duration(total_task_seconds)}")
    print(f"Mode switches:     {total_switches}")
    if total_switches:
        print(f"Wrong-switch rate: {ws_overall} ({total_wrong} of {total_switches} switches)")
    else:
        print(f"Wrong-switch rate: 0% (0 of 0 switches)")
    print(f"Manual overrides:  {total_overrides}")
    print(f"Undo count:        {total_undos}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_paths(argv):
    if not argv:
        print("Usage: summarize_logs.py [file.csv ...] | --all | --last N", file=sys.stderr)
        sys.exit(1)

    if argv[0] == "--all":
        paths = sorted(_RUNS_DIR.glob("*.csv"))
        if not paths:
            print(f"No CSV files found in {_RUNS_DIR}", file=sys.stderr)
            sys.exit(1)
        return paths

    if argv[0] == "--last":
        try:
            n = int(argv[1])
        except (IndexError, ValueError):
            print("--last requires an integer argument", file=sys.stderr)
            sys.exit(1)
        paths = sorted(_RUNS_DIR.glob("*.csv"))[-n:]
        if not paths:
            print(f"No CSV files found in {_RUNS_DIR}", file=sys.stderr)
            sys.exit(1)
        return paths

    # Explicit file list
    paths = []
    for arg in argv:
        p = Path(arg)
        if not p.is_absolute():
            # Try relative to project root first, then cwd
            candidates = [_PROJECT_ROOT / p, Path.cwd() / p, p]
            for c in candidates:
                if c.exists():
                    p = c
                    break
        if not p.exists():
            print(f"[warn] File not found: {arg}", file=sys.stderr)
            continue
        paths.append(p)
    if not paths:
        print("No valid CSV files specified.", file=sys.stderr)
        sys.exit(1)
    return paths


def main():
    argv = sys.argv[1:]
    paths = _resolve_paths(argv)
    condition_map, total_events, total_task_seconds, files_ok = analyse_files(paths)
    print_report(paths, condition_map, total_events, total_task_seconds, files_ok)


# Public alias used by tests / other modules
summarize = analyse_files


if __name__ == "__main__":
    main()
