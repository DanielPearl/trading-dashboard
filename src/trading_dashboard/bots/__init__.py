"""In-process trading bots.

Each module here exposes a ``start_daemon(...)`` that spawns a background
thread inside the dashboard process. The thread owns the bot's polling
loop, calls into the bot's predict/decide functions, and writes its own
state files. The dashboard's HTTP server, the bot toggle on the Home
tab (``bot_state.is_bot_enabled``), and the hedge_monitor / regime_monitor
daemons all coexist with these bot threads in the same process.

This consolidation replaces the previous topology of one systemd unit
per bot on the droplet — the dashboard is now the single source of
truth for both viewing and trading.
"""
