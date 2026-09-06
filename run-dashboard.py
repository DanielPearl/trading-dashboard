#!/usr/bin/env python3
"""Start the local dashboard with the shared Kalshi creds loaded.

Why this exists
---------------
The dashboard's in-process bot daemons call
``bots/_base.require_kalshi_creds()``, which reads ``KALSHI_API_KEY_ID``
and ``KALSHI_PRIVATE_KEY_PATH`` from the process environment. On the
droplet systemd injects them via ``EnvironmentFile=``. Nothing did that
locally, so every creds-requiring bot (world-cup, tennis, the macro
bots) died at startup with "bot needs Kalshi creds in env" while
public-data-only bots like reality-leaks carried on — which looks
exactly like "trading is broken for some bots".

Everything here is resolved RELATIVE to this file. In August 2026 the
whole tree moved from ``~/Documents/Kalshi`` to
``~/Documents/Documents (Local)/Kalshi`` (iCloud Drive does this when
"Desktop & Documents" sync is switched on) and every hardcoded absolute
path on the machine broke at once — including the ``kalshi_sdk``
editable install, which took down every bot. Keep this script
path-relative so a future move is a non-event.

Loading is done with python-dotenv rather than shell sourcing on
purpose: the current path contains ``(Local)``, and parentheses are
shell metacharacters. ``. .env.shared`` silently dropped
KALSHI_PRIVATE_KEY_PATH.

Usage
-----
    python3 run-dashboard.py                  # local config, port 8080
    python3 run-dashboard.py --config X.yaml --port 9000
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # Trading dashboard/
ROOT = HERE.parent                              # Kalshi/
ENV_FILE = ROOT / "Secret Keys" / ".env.shared"
DEFAULT_CONFIG = HERE / "config" / "dashboard.local.yaml"


def _load_env() -> None:
    """Merge .env.shared into os.environ WITHOUT clobbering real env.

    Matches kalshi_sdk.load_kalshi_env's documented precedence:
    shared baseline → per-bot .env → process env (later wins).
    """
    if not ENV_FILE.exists():
        print(f"warn: {ENV_FILE} not found — bots that need Kalshi creds "
              f"will fail their startup check", file=sys.stderr, flush=True)
        return
    try:
        from dotenv import dotenv_values
    except ImportError:
        print("warn: python-dotenv not installed; skipping env load",
              file=sys.stderr, flush=True)
        return
    loaded = 0
    for key, val in dotenv_values(ENV_FILE).items():
        if val is not None and key not in os.environ:
            os.environ[key] = val
            loaded += 1
    print(f"loaded {loaded} vars from {ENV_FILE.name}", flush=True)

    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    if key_path and not Path(key_path).exists():
        print(f"warn: KALSHI_PRIVATE_KEY_PATH does not exist: {key_path}\n"
              f"      (stale absolute path? it should live under {ROOT})",
              file=sys.stderr, flush=True)


def _ensure_sdk() -> None:
    """Fall back to the sibling checkout if the editable install is stale.

    A moved tree leaves ``__editable___kalshi_sdk_*_finder.py`` pointing
    at a directory that no longer exists, and every import fails. Rather
    than have the dashboard die, put the checkout on the path and say
    what the permanent fix is.
    """
    try:
        import kalshi_sdk  # noqa: F401
        return
    except ImportError:
        pass
    sdk = ROOT / "kalshi_sdk"
    if (sdk / "kalshi_sdk" / "__init__.py").exists():
        sys.path.insert(0, str(sdk))
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(sdk), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        print(f"warn: kalshi_sdk editable install is stale — falling back to "
              f"{sdk}\n"
              f"      permanent fix: python3 -m pip install --user -e "
              f"'{sdk}'", file=sys.stderr, flush=True)
    else:
        print(f"error: cannot import kalshi_sdk and no checkout at {sdk}",
              file=sys.stderr, flush=True)
        raise SystemExit(1)


def main() -> int:
    _load_env()
    _ensure_sdk()

    args = sys.argv[1:]
    if not any(a == "--config" for a in args):
        args = ["--config", str(DEFAULT_CONFIG)] + args
    if not any(a == "--port" for a in args):
        args += ["--port", "8080"]

    sys.argv = [str(HERE / "dashboard.py")] + args
    sys.path.insert(0, str(HERE / "src"))
    print(f"starting: dashboard.py {' '.join(args)}", flush=True)

    from trading_dashboard.dashboard import main as dashboard_main
    return dashboard_main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
