#!/usr/bin/env python3
"""Fallback daily diagnosis collector — mechanical, no AI reasoning.

Walks every monitored Kalshi systemd service on this droplet, grabs the
last 24h of journalctl output, and writes a JSON report to
``/root/trading-dashboard/diagnosis/latest.json`` (plus a dated copy in
``diagnosis/history/``).  The trading-dashboard's ``/api/diagnosis/latest``
route reads the same file and the in-page Diagnosis button renders it.

This is the dumb-cron fallback for when the scheduled Claude agent can't
run.  It surfaces:

* per-service health (active / restart count / error count)
* unique tracebacks found in the last 24h as ``bugs`` entries
* repeated warning lines as ``streamlining`` entries

The Claude agent, when available, overwrites the same JSON with richer
analysis (root-cause, file:line, suggested fix).  Disable the systemd
timer (``systemctl disable --now diagnosis-collector.timer``) when
swapping over so there are no competing writers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

# Services covered by this collector.  Keep in sync with the Claude agent
# prompt — the dashboard renders whatever we put in ``services``.
#
# After the in-process bot fold-in (see src/trading_dashboard/bots/),
# every per-bot service was stopped + disabled. They live on as
# inactive units in /etc/systemd/system/ for rollback, but auditing
# them surfaces stale traceback noise + always-failing rows because
# ``systemctl is-active`` returns "inactive" → _classify returns
# "failing". So the live monitored set is just the dashboard
# (which now runs all 10 bot threads inside one process).
MONITORED_SERVICES: tuple[str, ...] = (
    "trading-dashboard.service",
)

# Disabled-on-purpose units we keep around for rollback. The
# collector ignores these — anything we'd want to know about them
# (a stray crash log, etc.) shows up under trading-dashboard.service
# in practice now that those bots run as threads inside the dashboard.
LEGACY_DISABLED_SERVICES: tuple[str, ...] = (
    "kalshi-nba.service",
    "kalshi-cpi.service",
    "kalshi-gas-bot.service",
    "kalshi-unemployment-bot.service",
    "baseline-break-monitor.service",
    "billboard-charts-monitor.service",
    "darts-monitor.service",
    "survivor-elimination-monitor.service",
    "table-tennis-monitor.service",
    "kalshi-natgas-bot.timer",
)

# Where the dashboard expects the report.  Dashboard runs with
# WorkingDirectory=/root/trading-dashboard and reads ``diagnosis/latest.json``
# relative to that.
DIAG_DIR = Path("/root/trading-dashboard/diagnosis")
LATEST_PATH = DIAG_DIR / "latest.json"
HISTORY_DIR = DIAG_DIR / "history"

# Pattern that opens a Python traceback block.  We grab from this line
# through the next blank line / next non-indented log line to capture
# the full stack as one "bug" item.
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):")

# Heuristic for "real" error lines — case-insensitive.  Used both to
# bump the per-service error count and to populate the ``notable`` field
# with the most recent example.
ERROR_RE = re.compile(r"\b(ERROR|CRITICAL|Exception|Traceback)\b")
WARN_RE = re.compile(r"\b(WARNING|WARN)\b")


def _run(cmd: list[str], timeout: int = 30) -> str:
    """Run a command and return stdout.  Returns empty string on failure
    so a single broken service doesn't abort the whole run.
    """
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
        return r.stdout
    except (subprocess.TimeoutExpired, OSError):
        return ""


def _service_active(name: str) -> str:
    """Return systemd's idea of the service state: ``active``, ``failed``,
    ``inactive``, etc.  Used to classify health.
    """
    return _run(["systemctl", "is-active", name], timeout=10).strip() or "unknown"


def _last_start_iso(name: str) -> str | None:
    """Return the ISO timestamp of the unit's last (re)start, suitable
    for passing to ``journalctl --since``.  Returns None if systemd
    can't tell us (e.g. unit never started).

    Used to scope the "degraded" classification: errors logged BEFORE
    the most recent restart almost always reflect a problem that has
    since been fixed by a deploy.  Counting them against the live
    service produces false-positive "degraded" badges for hours after
    shipping a fix — the worst time to look unreliable.
    """
    # ActiveEnterTimestamp format: "Fri 2026-05-22 20:56:24 UTC"
    out = _run(["systemctl", "show", name, "-p", "ActiveEnterTimestamp",
                "--value"], timeout=10).strip()
    if not out or out == "0" or out.startswith("n/a"):
        return None
    # journalctl --since accepts the systemd timestamp format directly,
    # but we strip the leading weekday for clarity.
    parts = out.split(" ", 1)
    return parts[1] if len(parts) == 2 else out


def _crash_restart_count(name: str) -> int:
    """How many times systemd has had to auto-restart this unit because
    the process exited with a non-zero status (i.e. crashed).  This is
    the alarming signal — it does NOT count clean ``systemctl restart``
    invocations, which are almost always deploys and shouldn't trigger
    a "service failing" classification.

    The original collector greped the journal for ``Started``/``Starting``
    lines, which over-counted in two ways:

    1. App-level logs occasionally contain those words (``hedge_monitor
       started``, etc.) — bumping the count without any real restart.
    2. Manual ``systemctl restart`` (= every deploy) also produces a
       ``systemd[1]: Started <unit>`` line, so a busy deploy day looked
       like a crash loop.

    ``systemctl show -p NRestarts`` returns systemd's own count of
    failure-driven auto-restarts and is exactly the signal we want.
    """
    out = _run(["systemctl", "show", name, "-p", "NRestarts", "--value"],
               timeout=10).strip()
    try:
        return int(out)
    except ValueError:
        return 0


def _manual_restart_count_24h(name: str) -> int:
    """Count clean (re)starts in the last 24h — most of these are deploys.
    Surfaced separately so the dashboard can distinguish "you deployed 5
    times today" from "the service crashed 5 times today".

    We restrict to log lines emitted by PID 1 (systemd itself), which
    excludes app-level "Started X" log lines that share the keyword.
    """
    out = _run([
        "journalctl", "-u", name, "--since", "24 hours ago",
        "--no-pager",
    ])
    return sum(1 for line in out.splitlines()
               if f"systemd[1]: Started {name}" in line)


def _journal_24h(name: str) -> list[str]:
    """Plain log lines for a service over the last 24h."""
    out = _run([
        "journalctl", "-u", name, "--since", "24 hours ago",
        "--no-pager", "-o", "short-iso",
    ], timeout=60)
    return out.splitlines()


def _journal_since(name: str, since: str) -> list[str]:
    """Plain log lines for a service since an arbitrary timestamp.
    Used to look only at log activity since the most recent restart
    when classifying current health (vs the full 24h window which
    keeps reporting stale, already-fixed errors).
    """
    out = _run([
        "journalctl", "-u", name, "--since", since,
        "--no-pager", "-o", "short-iso",
    ], timeout=60)
    return out.splitlines()


def _extract_tracebacks(lines: list[str]) -> list[str]:
    """Group consecutive traceback lines into single error blocks.  Each
    returned string is one full traceback (header + frames + exception).
    Caps at 12 frames per block so a runaway recursion doesn't blow up
    the JSON.
    """
    blocks: list[str] = []
    i = 0
    while i < len(lines):
        if TRACEBACK_RE.search(lines[i]):
            chunk = [lines[i]]
            j = i + 1
            # Traceback frames are indented; exception line is unindented
            # but starts with "<ExceptionName>:" — keep grabbing until we
            # see a clearly-new log entry (different timestamp prefix).
            while j < len(lines) and len(chunk) < 14:
                ln = lines[j]
                if not ln.strip():
                    break
                # A new systemd journal entry usually starts with a
                # timestamp.  If the line looks like a fresh log entry
                # (has an ISO date at column 0), stop.
                if re.match(r"^\d{4}-\d{2}-\d{2}T", ln) and j > i + 1:
                    break
                chunk.append(ln)
                # Exception line marks the end of a traceback.
                if re.match(r"^\S+(?:Error|Exception)[:\s]", ln.lstrip()):
                    chunk.append(ln) if False else None  # already appended
                    j += 1
                    break
                j += 1
            blocks.append("\n".join(chunk))
            i = j
        else:
            i += 1
    return blocks


def _dedup_keep_order(items: Iterable[str]) -> list[str]:
    """Stable de-dup so we report each unique traceback once but keep
    chronological order (oldest first).
    """
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = it[:200]  # rough fingerprint; tracebacks with same head dedup
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# journalctl short-iso prefix:
#   "2026-05-23T11:02:02+00:00 ubuntu-s-1vcpu-512mb-10gb-nyc1 python[627890]: "
# The collector reads this from every log line; the user doesn't
# care which PID or host it came from when they're trying to read
# what actually broke. Strip aggressively.
_JOURNAL_PREFIX_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+\-]\d{2}:\d{2}\s+\S+\s+\S+:\s*",
)


def _strip_journal_prefix(line: str) -> str:
    """Strip a single journalctl short-iso prefix from a log line."""
    return _JOURNAL_PREFIX_RE.sub("", line).strip()


def _strip_journal_prefixes(text: str) -> str:
    """Strip the journalctl prefix from every line of a multi-line
    block (e.g. a traceback's evidence). Used so the modal renders
    actual Python frames + exception messages instead of walls of
    timestamp / hostname / PID noise.
    """
    return "\n".join(_strip_journal_prefix(ln) for ln in text.splitlines())


def _classify(active_state: str, error_count: int, crash_restarts: int) -> str:
    """Map raw signals to the three statuses the dashboard renders.

    * failing  — systemd reports not-active, or the service has crashed
                 (systemd-detected auto-restart) at least once.  We treat
                 a single crash as failing because the previous heuristic
                 (>5 restarts) folded in clean deploys and made the whole
                 column meaningless.
    * degraded — at least one error in the last 24h.
    * healthy  — clean window.
    """
    if active_state != "active":
        return "failing"
    if crash_restarts > 0:
        return "failing"
    if error_count > 0:
        return "degraded"
    return "healthy"


def _notable(error_count: int, warn_count: int,
             crash_restarts: int, manual_restarts: int,
             last_error: str | None) -> str | None:
    """Pithy summary for the service-health table.  Prefer the most
    recent error line; fall back to counts when nothing severe happened
    but the window wasn't perfectly clean.  Crash restarts surface
    distinctly from manual ones so deploys don't look like incidents.
    """
    if last_error:
        # Strip the journalctl prefix (timestamp / hostname / python[PID]:)
        # before trimming — without that, the 120-char budget gets eaten
        # by log plumbing instead of the actual error text.
        text = _strip_journal_prefix(last_error.strip())
        if len(text) > 120:
            text = text[:117] + "..."
        return text
    bits = []
    if crash_restarts:
        bits.append(f"{crash_restarts} crash" + ("es" if crash_restarts != 1 else ""))
    if error_count:
        bits.append(f"{error_count} error" + ("s" if error_count != 1 else ""))
    if warn_count:
        bits.append(f"{warn_count} warning" + ("s" if warn_count != 1 else ""))
    if manual_restarts > 1:
        # Only mention deploys if there were several — one is just
        # "regular activity" and not worth surfacing.
        bits.append(f"{manual_restarts} deploys")
    return ", ".join(bits) if bits else None


def collect_service(name: str) -> tuple[dict, list[dict], list[dict]]:
    """Inspect one service.  Returns ``(service_row, bugs, streamlining)``
    where:

    * service_row goes in the dashboard's service-health table
    * bugs is one entry per unique traceback
    * streamlining is one entry per high-frequency warning pattern
    """
    active = _service_active(name)
    crash_restarts = _crash_restart_count(name)
    manual_restarts = _manual_restart_count_24h(name)

    # Window the bugs / streamlining lists to "since the unit was last
    # (re)started", capped at 24h. The previous version used the full
    # 24h for these lists even though classification already scoped to
    # since-restart — net effect was that the dashboard kept rendering
    # tracebacks from pre-fix code for up to a day after the deploy
    # that fixed them, plus warning counts that included pre-filter
    # spam. The collector is a "what's broken right now" tool, not an
    # incident archive, so all three signals should agree on the same
    # window.
    now = datetime.now(timezone.utc)
    h24_ago_iso = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S UTC")
    last_start = _last_start_iso(name)
    # Both timestamps are "YYYY-MM-DD HH:MM:SS UTC"; lexicographic
    # comparison matches chronological order in that format.
    window_since = (last_start
                     if last_start and last_start > h24_ago_iso
                     else h24_ago_iso)
    window_lines = _journal_since(name, window_since)
    errors = [ln for ln in window_lines if ERROR_RE.search(ln)]
    warns = [ln for ln in window_lines if WARN_RE.search(ln)]
    tracebacks = _dedup_keep_order(_extract_tracebacks(window_lines))

    last_error = errors[-1] if errors else None
    status = _classify(active, len(errors), crash_restarts)

    service_row = {
        "name": name,
        "status": status,
        # ``restarts_24h`` keeps the original key the dashboard renders
        # but now means "crashes" (the alarming signal).  Deploys live
        # in the notable line instead so the column stays meaningful.
        "restarts_24h": crash_restarts,
        "notable": _notable(len(errors), len(warns),
                            crash_restarts, manual_restarts, last_error),
    }

    # Group tracebacks by their (file:line, exception) pair so the
    # dashboard renders one row per UNIQUE failure with an occurrence
    # count + first/last-seen timestamps. The previous flat list
    # produced one row per stack-trace instance, so a single bug that
    # fires every 5 minutes turned into a dozen near-identical
    # entries and the user had no way to tell "12 separate bugs"
    # from "1 bug, 12 hits".
    bug_groups: dict[tuple[str, str], dict] = {}
    for tb in tracebacks:
        where = "(unknown — see evidence)"
        for ln in reversed(tb.splitlines()):
            m = re.search(r'File "([^"]+)", line (\d+)', ln)
            if m:
                where = f"{m.group(1)}:{m.group(2)}"
                break
        # Last line of the traceback IS the exception. Strip the
        # journalctl prefix ("YYYY-MM-DDTHH:MM:SS+TZ host python[PID]:
        # ") so the user sees the actual error text, not log
        # plumbing.
        exc_line = tb.splitlines()[-1].strip()
        clean_exc = _strip_journal_prefix(exc_line)
        # Pull the first line's timestamp out as the seen-at marker.
        first_line = tb.splitlines()[0]
        ts_match = re.match(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',
                             first_line)
        ts = ts_match.group(1) if ts_match else ""
        key = (where, clean_exc)
        if key in bug_groups:
            bug_groups[key]["count"] += 1
            if ts and ts > (bug_groups[key].get("last_seen") or ""):
                bug_groups[key]["last_seen"] = ts
        else:
            bug_groups[key] = {
                "service": name,
                "what": clean_exc[:240],
                "where": where,
                "count": 1,
                "first_seen": ts,
                "last_seen": ts,
                # Keep the trimmed traceback as evidence for the
                # "expand details" affordance, also with the journal
                # prefix stripped so it's actually readable.
                "evidence": _strip_journal_prefixes(tb[:1500]),
            }
    bugs: list[dict] = sorted(
        bug_groups.values(), key=lambda b: -b["count"],
    )[:5]

    # Streamlining: warnings that fire ≥10× in the current window
    # (capped at 24h, scoped down to since-last-restart) almost always
    # indicate a missed retry / chatty third-party / dead config
    # branch.  Surface the top three so the dashboard hints at them
    # without listing every noise line.
    streamlining: list[dict] = []
    warn_norm = [re.sub(r"\d+", "N", w.split("]", 1)[-1].strip()) for w in warns]
    common = Counter(warn_norm).most_common(3)
    for pattern, count in common:
        if count < 10:
            continue
        # The "evidence" pattern is the warning text with numbers
        # replaced by "N"; clean up the leading "WARNING " prefix
        # that the journal emits so it reads like a sentence.
        evidence = pattern[:400].lstrip(": ").strip()
        streamlining.append({
            "service": name,
            "what": f"{count}× since last restart",
            "where": "",  # not file:line-shaped; let the renderer hide
            "evidence": evidence,
            # suggested_fix omitted on purpose — the previous canned
            # "Consider rate-limiting, fixing root cause, or lowering
            # log level if expected" added zero information per item
            # and crowded out the actual evidence line.
        })

    return service_row, bugs, streamlining


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI surface so the live/sim split can run two collectors with
    different service lists + output paths. Defaults preserve today's
    sim-only behaviour, so the existing systemd timer keeps working
    unchanged.
    """
    p = argparse.ArgumentParser(
        description="Mechanical droplet diagnosis collector.",
    )
    p.add_argument(
        "--services", nargs="+", default=list(MONITORED_SERVICES),
        help=("Systemd services to audit. Defaults to "
              f"{', '.join(MONITORED_SERVICES)} (the sim dashboard). "
              "The live timer should pass "
              "``--services trading-dashboard-live.service``."),
    )
    p.add_argument(
        "--diag-dir", type=Path, default=DIAG_DIR,
        help=("Directory to write latest.json + history/ into. "
              "Defaults to /root/trading-dashboard/diagnosis/. "
              "The live timer should point this at "
              "/root/trading-dashboard/diagnosis-live/."),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    diag_dir = Path(args.diag_dir)
    latest_path = diag_dir / "latest.json"
    history_dir = diag_dir / "history"
    diag_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    services: list[dict] = []
    bugs: list[dict] = []
    streamlining: list[dict] = []
    for name in args.services:
        try:
            srv, srv_bugs, srv_stream = collect_service(name)
        except Exception as e:  # noqa: BLE001 — never let one service abort the run
            srv = {
                "name": name,
                "status": "failing",
                "restarts_24h": 0,
                "notable": f"collector error: {e}",
            }
            srv_bugs = []
            srv_stream = []
        services.append(srv)
        bugs.extend(srv_bugs)
        streamlining.extend(srv_stream)

    healthy = sum(1 for s in services if s["status"] == "healthy")
    report = {
        "generated_at": now,
        "last_checked_at": now,
        "status": "completed",
        "services_audited": len(services),
        "services_healthy": healthy,
        "issues_found": len(bugs) + len(streamlining),
        # No GitHub issue from the fallback — the cron is local-only.
        # The Claude agent will populate this when it takes over.
        "github_issue_url": None,
        "generator": "fallback-cron",
        "services": services,
        "bugs": bugs,
        "recommended_changes": [],
        "streamlining": streamlining,
    }

    # Atomic write — open a tempfile in the same dir, fsync, rename.
    # The dashboard reads this on every API hit; a partial JSON would
    # break the modal until the next run.
    fd, tmp_path = tempfile.mkstemp(prefix=".latest-", suffix=".json",
                                     dir=str(diag_dir))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(report, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, latest_path)
    except Exception:
        # Best-effort cleanup if the rename never happened.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # History copy — useful when /schedule comes back and we want to
    # compare the cron's view to Claude's view of the same day.
    shutil.copy2(latest_path, history_dir / f"{today}.json")
    print(f"wrote {latest_path} ({len(bugs)} bugs, "
          f"{len(streamlining)} streamlining, {healthy}/{len(services)} healthy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
