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

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Services covered by this collector.  Keep in sync with the Claude agent
# prompt — the dashboard renders whatever we put in ``services``.
MONITORED_SERVICES: tuple[str, ...] = (
    "trading-dashboard.service",
    "kalshi-nba.service",
    "kalshi-cpi.service",
    "kalshi-gas-bot.service",
    "kalshi-unemployment-bot.service",
    "baseline-break-monitor.service",
    "billboard-charts-monitor.service",
    "darts-monitor.service",
    "survivor-elimination-monitor.service",
    "table-tennis-monitor.service",
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


def _restart_count_24h(name: str) -> int:
    """Count restart events in the last 24h by greping the unit's journal
    for the marker systemd writes on every (re)start.  Works for both
    crash-loops and clean restarts.
    """
    out = _run([
        "journalctl", "-u", name, "--since", "24 hours ago",
        "--no-pager", "-o", "cat",
    ])
    return sum(1 for line in out.splitlines()
               if "Started " in line or "Starting " in line)


def _journal_24h(name: str) -> list[str]:
    """Plain log lines for a service over the last 24h."""
    out = _run([
        "journalctl", "-u", name, "--since", "24 hours ago",
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


def _classify(active_state: str, error_count: int, restarts: int) -> str:
    """Map raw signals to the three statuses the dashboard renders.

    * failing  — systemd reports not-active, or the service has been
                 restarted more than 5× in 24h (crash loop).
    * degraded — at least one error in the last 24h.
    * healthy  — clean window.
    """
    if active_state != "active" or restarts > 5:
        return "failing"
    if error_count > 0:
        return "degraded"
    return "healthy"


def _notable(error_count: int, warn_count: int, restarts: int,
             last_error: str | None) -> str | None:
    """Pithy summary for the service-health table.  Prefer the most
    recent error line; fall back to counts when nothing severe happened
    but the window wasn't perfectly clean.
    """
    if last_error:
        # Trim to keep the table tidy; the full evidence lives in bugs[].
        text = last_error.strip()
        if len(text) > 120:
            text = text[:117] + "..."
        return text
    bits = []
    if error_count:
        bits.append(f"{error_count} error" + ("s" if error_count != 1 else ""))
    if warn_count:
        bits.append(f"{warn_count} warning" + ("s" if warn_count != 1 else ""))
    if restarts > 1:
        bits.append(f"{restarts} restarts")
    return ", ".join(bits) if bits else None


def collect_service(name: str) -> tuple[dict, list[dict], list[dict]]:
    """Inspect one service.  Returns ``(service_row, bugs, streamlining)``
    where:

    * service_row goes in the dashboard's service-health table
    * bugs is one entry per unique traceback
    * streamlining is one entry per high-frequency warning pattern
    """
    active = _service_active(name)
    restarts = _restart_count_24h(name)
    lines = _journal_24h(name)
    errors = [ln for ln in lines if ERROR_RE.search(ln)]
    warns = [ln for ln in lines if WARN_RE.search(ln)]
    tracebacks = _dedup_keep_order(_extract_tracebacks(lines))

    last_error = errors[-1] if errors else None
    status = _classify(active, len(errors), restarts)

    service_row = {
        "name": name,
        "status": status,
        "restarts_24h": restarts,
        "notable": _notable(len(errors), len(warns), restarts, last_error),
    }

    bugs: list[dict] = []
    for tb in tracebacks[:5]:  # cap per-service so the JSON stays readable
        # Pick the last "File ..." frame as a where-hint; the in-app
        # source is often a thin wrapper, so the deepest user frame is
        # usually the most informative.
        where = None
        for ln in reversed(tb.splitlines()):
            m = re.search(r'File "([^"]+)", line (\d+)', ln)
            if m:
                where = f"{m.group(1)}:{m.group(2)}"
                break
        # Last line of the traceback is the exception itself — use it
        # as the "what" so the dashboard shows the actual error type.
        exc_line = tb.splitlines()[-1].strip()
        bugs.append({
            "service": name,
            "what": exc_line[:200],
            "where": where or "(unknown — see evidence)",
            "evidence": tb[:1200],
            "suggested_fix": "(automated collector — no fix proposed; "
                             "review the traceback)",
        })

    # Streamlining: warnings that fire ≥10× in 24h almost always indicate
    # a missed retry / chatty third-party / dead config branch.  Surface
    # the top three so the dashboard hints at them without listing every
    # noise line.
    streamlining: list[dict] = []
    warn_norm = [re.sub(r"\d+", "N", w.split("]", 1)[-1].strip()) for w in warns]
    common = Counter(warn_norm).most_common(3)
    for pattern, count in common:
        if count < 10:
            continue
        streamlining.append({
            "service": name,
            "what": f"Warning fires {count}× in 24h",
            "where": "(repeated log pattern)",
            "evidence": pattern[:400],
            "suggested_fix": "Consider rate-limiting, fixing root cause, "
                             "or lowering log level if expected.",
        })

    return service_row, bugs, streamlining


def main() -> int:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    services: list[dict] = []
    bugs: list[dict] = []
    streamlining: list[dict] = []
    for name in MONITORED_SERVICES:
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
                                     dir=str(DIAG_DIR))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(report, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, LATEST_PATH)
    except Exception:
        # Best-effort cleanup if the rename never happened.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # History copy — useful when /schedule comes back and we want to
    # compare the cron's view to Claude's view of the same day.
    shutil.copy2(LATEST_PATH, HISTORY_DIR / f"{today}.json")
    print(f"wrote {LATEST_PATH} ({len(bugs)} bugs, "
          f"{len(streamlining)} streamlining, {healthy}/{len(services)} healthy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
