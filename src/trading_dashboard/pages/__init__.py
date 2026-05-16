"""Page renderers for bot-specific dashboard views.

Each module exposes a ``render`` function returning a fully-formed HTML
string. The top-level ``dashboard.do_GET`` dispatches into here based on
the bot's ``dashboard_type`` / requested ``?tab=``.
"""
