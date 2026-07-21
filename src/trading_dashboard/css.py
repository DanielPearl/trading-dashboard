"""The dashboard's single shared stylesheet."""
from __future__ import annotations



# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #

CSS = """
* { box-sizing: border-box; }
body { background: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; line-height: 1.4; }
h1, h2 { color: #f0f6fc; margin: 0 0 8px 0; }
h1 { font-size: 22px; font-weight: 600; }
h2 { font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #8b949e; margin-top: 28px; margin-bottom: 8px; }
.meta { color: #8b949e; font-size: 12px; margin-bottom: 20px; }
/* ──────────────────────────────────────────────────────────────────
   Mode-pill — header anchor that switches between the sim dashboard
   (port 8080) and the live dashboard (port 8081). JS at the bottom
   of the page rewrites its href on load to point at the current
   window.location.host with the peer port, preserving the user's
   tab / bot / period query string so a click never loses context.
   ────────────────────────────────────────────────────────────── */
.mode-pill {
  display: inline-block;
  margin-left: 14px;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-decoration: none;
  border: 1px solid;
  vertical-align: middle;
  transition: background 0.15s, color 0.15s;
}
body[data-mode="sim"] .mode-pill {
  background: rgba(248, 81, 73, 0.10);
  color: #f85149;
  border-color: #f85149;
}
body[data-mode="sim"] .mode-pill:hover { background: #f85149; color: #fff; }
body[data-mode="live"] .mode-pill {
  background: rgba(63, 185, 80, 0.10);
  color: #3fb950;
  border-color: #3fb950;
}
body[data-mode="live"] .mode-pill:hover { background: #3fb950; color: #fff; }
/* ──────────────────────────────────────────────────────────────────
   Sim-mode chrome. Live is intentionally the "default" look (clean
   GitHub-dark, normal h1 colour) so it feels professional rather
   than alarming. Sim wears a subtle blue accent — a thin gradient
   bar pinned across the top of every page and a muted-blue h1 —
   to signal "paper-trading playground" without using the danger
   colour wheel. Blue was chosen over green because green would
   read as "go / trade enabled" against a financial dashboard.

   Combined with the existing header text + meta line + mode-pill
   differences, sim has a coherent identity you can read at a
   glance with both tabs open side-by-side.
   ────────────────────────────────────────────────────────────── */
body[data-mode="sim"]::before {
  content: "";
  display: block;
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, #1f6feb 0%, #58a6ff 50%, #1f6feb 100%);
  z-index: 100;
  pointer-events: none;
}
body[data-mode="sim"] h1 {
  color: #79c0ff;
}
.row { display: flex; gap: 18px; flex-wrap: wrap; margin-bottom: 14px; }
/* Cards sit on a slightly lighter, slightly cooler shade than the
   section background (#161b22). Subtle border + soft drop-shadow gives
   them a gentle elevated appearance against the section panel. */
.card { background: #1d232c; border: 1px solid #30363d; border-radius: 8px; padding: 14px 18px; flex: 1; min-width: 180px; box-shadow: 0 1px 2px rgba(0,0,0,0.35); text-align: center; }
.card .label { font-size: 11px; text-transform: uppercase; color: #9ca5b3; letter-spacing: 0.05em; }
.card .value { font-size: 22px; font-weight: 600; color: #f0f6fc; margin-top: 4px; }
/* Color modifiers — must be more specific than .card .value (0,2,0) so
   the green/red/gray classes actually paint summary card values. */
.card .value.green, .green { color: #56d364; }
.card .value.red, .red { color: #f85149; }
.card .value.gray, .gray { color: #8b949e; }
.card .value.yellow, .yellow { color: #e3b341; }
table { width: 100%; border-collapse: collapse; background: transparent; font-size: 13px; margin: 4px 0; }
th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #21262d; }
tr:last-child td { border-bottom: none; }
th { background: #161b22; color: #8b949e; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
/* Right-align numeric column headers so they sit directly over their
   values. Without this override th inherits the global "text-align:
   left" and the label floats off the left edge of a right-aligned
   number column. Descendant flex/grid layouts inside a th (the split
   YES|NO stacks) get their own explicit alignment further down and
   are unaffected. */
th.num { text-align: right; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.badge-yes { background: rgba(86, 211, 100, 0.2); color: #56d364; }
.badge-no { background: rgba(248, 81, 73, 0.2); color: #f85149; }
/* Sport-bot active-bet side cell: same team-tricode / "vs opp"
   layout the watchlist uses underneath. Player / team names render
   in the default cell colour (no YES/NO accent) so the column reads
   as identity, not direction. */
.active-side-team strong { font-weight: 700; }
.badge-skip { background: rgba(139, 148, 158, 0.2); color: #8b949e; }
.badge-hedge { background: rgba(227, 179, 65, 0.2); color: #e3b341; margin-left: 4px; }
.empty { color: #8b949e; padding: 14px; text-align: center; font-style: italic; }
/* EV diagnostic banner — loud when the trade has gone NEGATIVE EV. */
.ev-warning { background: rgba(248, 81, 73, 0.12); border: 1px solid #f85149;
   color: #ffa6a1; padding: 10px 14px; border-radius: 6px;
   margin: 12px 0; font-size: 13px; line-height: 1.45; }
.ev-warning strong { color: #f85149; }
/* "Why this trade was taken" — two-column grid of audit rows. */
.why-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 24px;
   margin: 8px 0 12px; max-width: 720px; }
.why-row { display: flex; justify-content: space-between;
   border-bottom: 1px dashed #30363d; padding: 4px 0; font-size: 13px; }
.why-row span:first-child { color: #8b949e; }
.why-row span:last-child { color: #c9d1d9; font-variant-numeric: tabular-nums; }
.why-gates { margin: 6px 0 14px; line-height: 1.6; }
.why-gates .mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
   font-size: 11px; color: #c9d1d9; }
/* Status pill — used on watchlist verdict + diagnostics scorecards. */
.status-pill { display: inline-block; padding: 2px 8px; border-radius: 12px;
   font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
   text-transform: uppercase; }
.status-pill.green { background: rgba(86, 211, 100, 0.2); color: #56d364; }
.status-pill.yellow { background: rgba(227, 179, 65, 0.2); color: #e3b341; }
.status-pill.red { background: rgba(248, 81, 73, 0.2); color: #f85149; }
.status-pill.gray { background: rgba(139, 148, 158, 0.2); color: #8b949e; }
/* Brief highlight on cells whose value just updated via the live JS
   poll. Pulses then fades — keeps changes visible without being loud. */
@keyframes cell-flash-fade {
  0%   { background-color: rgba(88, 166, 255, 0.35); }
  100% { background-color: transparent; }
}
.cell-flash { animation: cell-flash-fade 0.8s ease-out; }
/* Buy-criteria reference card. Compact two-column variant + a wider
   three-column variant (with descriptions) used in Section 2. */
table.criteria { max-width: 560px; font-size: 12px; }
table.criteria.criteria-wide { max-width: 1100px; width: 100%; }
table.criteria td { padding: 6px 10px; border-bottom: 1px solid #1f2530;
    vertical-align: top; }
table.criteria td:first-child { color: #c9d1d9; font-weight: 500;
    white-space: nowrap; }
table.criteria.criteria-wide td:nth-child(2) {
    white-space: nowrap; color: #8b949e; }
table.criteria td.criteria-why {
    color: #8b949e; line-height: 1.5; font-size: 12px; }
table.criteria tr.criteria-group td {
    background: #1c2128; color: #c9d1d9; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.05em;
    padding-top: 6px; padding-bottom: 6px;
}
table.criteria code { background: transparent; color: #c9d1d9; padding: 0; }
.section { background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; margin-bottom: 24px; }
.section h2 { padding: 14px 22px 10px; margin: 0; }
/* Default inner padding so card rows / tables / paragraphs don't touch
   the section edge. Sections that needed different padding (.summary-body,
   .rules) override below. */
.section .body { padding: 14px 22px 18px; }
.bar { display: flex; align-items: baseline; gap: 8px; }
.bar .small, .small { font-size: 11px; color: #8b949e; }
td.mono, code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
/* Watchlist ticker/title links — keep the cell looking like the rest
   of the table at rest, only flip color + underline on hover so the
   affordance is discoverable without making the table feel like a
   wall of links. Applies whether the link sits in the Ticker cell
   (td.mono, non-sport bots) or the Title cell (sport bots, where the
   ticker column was dropped and the Title itself became the click
   target). */
a.ticker-link { color: inherit; text-decoration: none; }
a.ticker-link:hover { color: #58a6ff; text-decoration: underline; }
/* Bot-name link in the active-bets / bet-history tables — same
   restraint as the ticker links so the table stays readable. */
a.bot-link { color: inherit; text-decoration: none; }
a.bot-link:hover { color: #58a6ff; text-decoration: underline; }
code { background: #161b22; padding: 1px 6px; border-radius: 3px; color: #c9d1d9; }
/* hero-card sits inside the body which already has padding; only need
   internal vertical breathing room for multi-card scenarios. */
.hero-card { padding: 4px 0 6px 0; border-bottom: 1px solid #21262d; }
.hero-card:last-child { border-bottom: none; padding-bottom: 0; }
.hero-question { font-size: 22px; font-weight: 600; color: #f0f6fc; margin-bottom: 4px; }
.hero-question .badge { font-size: 13px; padding: 4px 10px; vertical-align: middle; }
.hero-question .hero-q-text { margin-left: 6px; }
.hero-question .hero-event-title { color: #f0f6fc; margin-right: 10px; }
.hero-ticker { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; color: #8b949e; margin-bottom: 14px; }
.hero-stats { margin-bottom: 14px; }
.hero-chart { padding: 4px 0; }
.rules { padding: 0; }
.rules ol { margin: 0; padding-left: 20px; line-height: 1.8; }
.rules p { margin-top: 0; }
.rules li { color: #c9d1d9; font-size: 13px; }
.rules code { font-size: 12px; }
.summary-body { padding: 18px 22px; }
/* Compact cards: applied wherever we want a tight row of equal-width
   stat cards that fit on one line at desktop widths. Used by Summary
   and the Model-strength row. Centered labels/values for visual alignment.
*/
.summary-body .row,
.row.compact { gap: 10px; flex-wrap: nowrap; }
.summary-body .card,
.row.compact > .card { padding: 14px 14px; min-width: 0; flex: 1 1 0;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; text-align: center; min-height: 78px; }
.summary-body .card .label,
.row.compact > .card .label { font-size: 10px; margin-bottom: 6px; }
.summary-body .card .value,
.row.compact > .card .value { font-size: 22px; line-height: 1.2; }
.summary-body .card .small,
.row.compact > .card .small { font-size: 10px; margin-top: 4px; }
@media (max-width: 1100px) {
    .summary-body .row,
    .row.compact { flex-wrap: wrap; }
    .summary-body .card,
    .row.compact > .card { flex: 1 1 30%; min-width: 150px; }
}
.subhead { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #8b949e; margin: 16px 0 8px 0; font-weight: 600; }
.subsec { padding: 0 0 14px 0; }
.subsec h3 { margin-top: 12px; }
/* Bot filter bar — slim, sits between sections like a real filter, not
   like another content section. Pill-style links per bot. */
.bot-filter-bar { display: flex; align-items: center; gap: 10px;
    padding: 4px 0 18px 0; margin-bottom: 8px; flex-wrap: wrap;
    border-bottom: 1px solid #21262d; margin-top: -8px; }
.section .hdr-flex { display: flex; align-items: center;
    justify-content: space-between; gap: 12px;
    padding: 14px 22px 10px; }
.section .hdr-flex h2 { padding: 0; margin: 0; }
.section .hdr-flex .bot-filter-bar { padding: 0; margin: 0;
    border-bottom: none; }
.bot-filter-bar .filter-label {
    color: #8b949e; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.06em; font-weight: 600; margin-right: 4px;
}
/* Bot dropdown — native <select> styled to match the rest of the
   dashboard. The chevron is drawn via background-image so the look
   stays consistent across browsers. */
.bot-select {
    background: #0d1117 url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'><path d='M2 3.5l3 3 3-3' fill='none' stroke='%238b949e' stroke-width='1.5'/></svg>") no-repeat right 10px center;
    color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px;
    padding: 6px 30px 6px 12px;
    font-size: 13px; line-height: 1.4;
    appearance: none; -webkit-appearance: none; -moz-appearance: none;
    cursor: pointer; min-width: 200px;
    transition: border-color 120ms, background-color 120ms;
}
.bot-select:hover { border-color: #40464d; background-color: #161b22; }
.bot-select:focus { outline: none; border-color: #1f6feb;
    box-shadow: 0 0 0 3px rgba(31, 111, 235, 0.18); }
.bot-select option { background: #0d1117; color: #c9d1d9; }
.filter-pill { background: #21262d; color: #c9d1d9; text-decoration: none;
    padding: 6px 14px; border-radius: 999px; font-size: 13px;
    border: 1px solid #30363d; transition: background 120ms, border-color 120ms;
    line-height: 1.4; }
.filter-pill:hover { background: #2d333b; border-color: #40464d; }
.filter-pill-active { background: #1f6feb; color: #fff; border-color: #1f6feb; }
.filter-pill-active:hover { background: #1f6feb; border-color: #1f6feb; }
.filter-pill-disabled { color: #6e7681; cursor: not-allowed; opacity: 0.7; }
.filter-pill-disabled:hover { background: #21262d; border-color: #30363d; }
/* Diagnosis button — sits inline in the bot-filter-bar to the right of
   the bot dropdown. Opens the shared diagnosis modal which fetches
   /api/diagnosis/latest and renders the latest scheduled droplet
   health report (bugs, recommended changes, streamlining ideas).
   Status dot reflects the latest report's health at a glance. */
.diagnosis-btn {
    background: transparent; color: #8b949e; border: 1px solid #30363d;
    border-radius: 6px; padding: 6px 12px; font-size: 12px;
    cursor: pointer; line-height: 1.4; margin-left: auto;
    display: inline-flex; align-items: center; gap: 6px;
    transition: background 120ms, border-color 120ms, color 120ms;
}
.diagnosis-btn:hover { background: #1c2128; border-color: #40464d;
    color: #c9d1d9; }
.diagnosis-btn .diagnosis-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #6e7681;
    flex-shrink: 0;
}
.diagnosis-btn .diagnosis-dot.has-issues { background: #f85149; }
.diagnosis-btn .diagnosis-dot.healthy    { background: #3fb950; }
.diagnosis-btn .diagnosis-dot.stale      { background: #d29922; }
/* Diagnosis modal — wider than the criteria modal because the body
   shows a service-health table plus three sections (bugs / recommended /
   streamlining). Same overlay-and-X-to-close idiom as criteria-modal. */
.diagnosis-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.55); z-index: 100;
}
.diagnosis-overlay[hidden] { display: none !important; }
.diagnosis-modal {
    position: fixed; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
    width: min(900px, 92vw); max-height: 85vh;
    display: flex; flex-direction: column;
    z-index: 101; box-shadow: 0 12px 48px rgba(0,0,0,0.6);
}
.diagnosis-modal[hidden] { display: none !important; }
.diagnosis-modal-head {
    display: flex; align-items: flex-start;
    justify-content: space-between; gap: 12px;
    padding: 14px 18px; border-bottom: 1px solid #21262d;
}
.diagnosis-modal-head h3 { margin: 0; font-size: 15px; font-weight: 700;
    color: #f0f6fc; }
.diagnosis-modal-head .diagnosis-meta {
    color: #8b949e; font-size: 12px; margin-top: 4px;
}
.diagnosis-modal-close {
    background: transparent; border: none; color: #8b949e;
    font-size: 24px; line-height: 1; cursor: pointer; padding: 0 4px;
}
.diagnosis-modal-close:hover { color: #f0f6fc; }
.diagnosis-modal-body {
    padding: 14px 18px; overflow-y: auto; flex: 1;
}
.diagnosis-section { margin-bottom: 18px; }
.diagnosis-section h4 {
    margin: 0 0 8px 0; font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.06em; color: #8b949e; font-weight: 600;
}
.diagnosis-item {
    border: 1px solid #30363d; border-radius: 6px;
    padding: 10px 12px; margin-bottom: 6px; background: #161b22;
}
.diagnosis-item .diagnosis-where {
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 11px; color: #8b949e; margin-bottom: 4px;
}
.diagnosis-item .diagnosis-evidence {
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 11px; color: #c9d1d9; white-space: pre-wrap;
    margin: 6px 0 0 0; padding: 6px 8px; background: #0d1117;
    border-radius: 4px; border-left: 2px solid #30363d;
    max-height: 180px; overflow-y: auto;
}
.diagnosis-item .diagnosis-evidence-wrap { margin-top: 6px; }
.diagnosis-item .diagnosis-evidence-wrap summary {
    cursor: pointer; color: #58a6ff; font-size: 11px;
    user-select: none; padding: 2px 0;
}
.diagnosis-item .diagnosis-evidence-wrap summary:hover { color: #79c0ff; }
.diagnosis-item .diagnosis-fix {
    color: #3fb950; font-size: 12px; margin-top: 6px;
}
.diagnosis-item .diagnosis-what-row {
    display: flex; align-items: baseline; flex-wrap: wrap;
    gap: 8px; margin-bottom: 4px;
}
.diagnosis-item .diagnosis-what {
    color: #f0f6fc; font-size: 13px; line-height: 1.4;
    font-weight: 500; flex: 1; min-width: 0; margin-bottom: 0;
    word-break: break-word;
}
.diagnosis-item .diagnosis-count {
    background: rgba(248, 81, 73, 0.15); color: #f85149;
    border: 1px solid rgba(248, 81, 73, 0.5);
    padding: 1px 8px; border-radius: 10px; font-size: 11px;
    font-weight: 700; font-variant-numeric: tabular-nums;
}
.diagnosis-item .diagnosis-last-seen {
    color: #6e7681; font-size: 11px; font-variant-numeric: tabular-nums;
}
/* Top-of-modal headline. Three palettes: ok (green), warn (amber),
   bad (red). The icon is a single character ("✓" / "!" / "—") so
   the headline reads cleanly in any rendering context including
   text-only screen readers. */
.diagnosis-headline {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 14px; margin-bottom: 16px;
    border-radius: 6px; font-size: 13px; line-height: 1.4;
    border: 1px solid; font-weight: 500;
}
.diagnosis-headline-icon {
    font-size: 18px; font-weight: 700; line-height: 1;
}
.diagnosis-headline-ok {
    background: rgba(63, 185, 80, 0.10);
    color: #56d364; border-color: rgba(63, 185, 80, 0.45);
}
.diagnosis-headline-warn {
    background: rgba(227, 179, 65, 0.10);
    color: #e3b341; border-color: rgba(227, 179, 65, 0.45);
}
.diagnosis-headline-bad {
    background: rgba(248, 81, 73, 0.10);
    color: #f85149; border-color: rgba(248, 81, 73, 0.45);
}
.diagnosis-headline-neutral {
    background: #161b22; color: #8b949e;
    border-color: #30363d;
}
.diagnosis-empty {
    text-align: center; padding: 32px 20px; color: #6e7681;
    font-size: 13px; line-height: 1.5;
}
.diagnosis-services-table {
    width: 100%; border-collapse: collapse; font-size: 12px;
}
.diagnosis-services-table th, .diagnosis-services-table td {
    padding: 6px 10px; text-align: left; border-bottom: 1px solid #21262d;
}
.diagnosis-services-table th {
    color: #8b949e; font-weight: 600; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.diagnosis-services-table td.status-healthy  { color: #3fb950; }
.diagnosis-services-table td.status-degraded { color: #d29922; }
.diagnosis-services-table td.status-failing  { color: #f85149; }
.diagnosis-github-link {
    display: inline-block; margin-top: 8px; padding: 6px 12px;
    background: #21262d; border: 1px solid #30363d; border-radius: 6px;
    color: #58a6ff; font-size: 12px; text-decoration: none;
}
.diagnosis-github-link:hover {
    background: #2d333b; border-color: #40464d; text-decoration: underline;
}
/* Tab bar for the per-bot detail panes. Same pill idiom as the bot/
   period filters above, slightly slimmer so the visual hierarchy reads
   "filter > tab > content". */
.tab-bar { display: flex; align-items: center; gap: 6px;
    padding: 0 0 10px 0; margin: 4px 0 12px;
    border-bottom: 1px solid #21262d; flex-wrap: wrap; }
.tab-pill { background: transparent; color: #8b949e; cursor: pointer;
    padding: 6px 14px; border-radius: 6px 6px 0 0; font-size: 13px;
    border: 1px solid transparent; line-height: 1.4;
    text-decoration: none; transition: color 120ms, background 120ms; }
.tab-pill:hover { color: #c9d1d9; background: #1c2128; }
.tab-pill-active { color: #f0f6fc; background: #21262d;
    border-color: #30363d; border-bottom-color: #21262d;
    margin-bottom: -1px; font-weight: 600; }
.tab-panel { display: none; }
.tab-panel-active { display: block; }
/* Contracts sub-tabs (Watchlist / Model / Training Data). Same pill
   idiom as the top-level bar, slightly smaller so the hierarchy reads
   "tab > sub-tab > filter > content". The bot filter renders directly
   below this bar and scopes all three sub-pages. */
.subtab-bar { display: flex; align-items: center; gap: 6px;
    padding: 0 0 8px 0; margin: 0 0 10px;
    border-bottom: 1px solid #21262d; flex-wrap: wrap; }
.subtab-pill { background: transparent; color: #8b949e; cursor: pointer;
    padding: 4px 12px; border-radius: 6px 6px 0 0; font-size: 12px;
    border: 1px solid transparent; line-height: 1.4;
    text-decoration: none; transition: color 120ms, background 120ms; }
.subtab-pill:hover { color: #c9d1d9; background: #1c2128; }
.subtab-pill-active { color: #f0f6fc; background: #21262d;
    border-color: #30363d; border-bottom-color: #21262d;
    margin-bottom: -1px; font-weight: 600; }
.subtab-panel { display: none; }
.subtab-panel-active { display: block; }
/* Seasons tab — one card per league. Fixed-width slots (auto-fill
   so a single card never stretches to fill its row) keep the grid
   uniform regardless of how many cards are on the page. */
.season-grid { display: grid; gap: 14px;
   grid-template-columns: repeat(auto-fill, minmax(280px, 320px));
   justify-content: start; }
.season-card { background: #1d232c; border: 1px solid #30363d;
   border-radius: 8px; padding: 14px 16px;
   box-shadow: 0 1px 2px rgba(0,0,0,0.35); display: flex;
   flex-direction: column; gap: 10px; }
.season-card-head { display: flex; align-items: center;
   justify-content: space-between; gap: 8px; }
.season-bot { color: #f0f6fc; font-weight: 600; font-size: 14px;
   text-decoration: none; }
.season-bot:hover { color: #58a6ff; text-decoration: underline; }
.season-name { color: #c9d1d9; font-size: 12px; }
.season-countdown { display: flex; align-items: baseline; gap: 8px;
   margin-top: 4px; flex-wrap: wrap; }
.season-countdown-label { font-size: 11px; text-transform: uppercase;
   letter-spacing: 0.05em; color: #8b949e; }
.season-countdown-value { font-size: 18px; font-weight: 600;
   font-variant-numeric: tabular-nums; }
/* Progress bar from start → end. Empty before start, fills as time
   passes, stays full once the season is over. */
.season-progress { background: #161b22; border: 1px solid #30363d;
   border-radius: 999px; height: 6px; overflow: hidden; }
.season-progress-fill { background: #58a6ff; height: 100%;
   transition: width 1s linear; }
.season-meta { display: grid; grid-template-columns: repeat(3, 1fr);
   gap: 6px; margin-top: 4px; }
.season-meta > div { display: flex; flex-direction: column; gap: 2px; }
.season-meta-label { font-size: 10px; text-transform: uppercase;
   letter-spacing: 0.05em; color: #8b949e; }
.season-meta-value { font-size: 13px; color: #f0f6fc;
   font-variant-numeric: tabular-nums; }
/* "Why?" button on each active-bets row + the criteria modal it
   opens. Single shared modal at page bottom; JS populates the body
   from data-criteria on the clicked button. */
/* Per-row info button — circle with an italic "i" inside, mirroring
   common information-icon affordances. */
.criteria-btn {
    background: #21262d; color: #8b949e; border: 1px solid #30363d;
    border-radius: 50%; width: 22px; height: 22px; padding: 0;
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic; font-weight: 700;
    font-size: 13px; line-height: 1; cursor: pointer;
    display: inline-flex; align-items: center; justify-content: center;
    transition: background 120ms, border-color 120ms, color 120ms; }
.criteria-btn:hover { background: #2d333b; border-color: #1f6feb;
    color: #f0f6fc; }
/* Used for the "what does the bot need before it'll buy" reference
   popup, rendered inline next to the Active-bet h3 as the same
   circle-i info-icon affordance as the per-row criteria-btn. */
.criteria-rules-btn {
    background: #21262d; color: #8b949e; border: 1px solid #30363d;
    border-radius: 50%; width: 22px; height: 22px; padding: 0;
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic; font-weight: 700;
    font-size: 13px; line-height: 1; cursor: pointer;
    display: inline-flex; align-items: center; justify-content: center;
    transition: background 120ms, border-color 120ms, color 120ms; }
.criteria-rules-btn:hover { background: #2d333b; border-color: #1f6feb;
    color: #f0f6fc; }
/* Same circle-i affordance for the Kalshi-rules section header.
   Clicking opens the shared modal with the extended contract rules
   (primary + secondary paragraphs from Kalshi). */
.contract-rules-btn {
    background: #21262d; color: #8b949e; border: 1px solid #30363d;
    border-radius: 50%; width: 22px; height: 22px; padding: 0;
    font-family: Georgia, "Times New Roman", serif;
    font-style: italic; font-weight: 700;
    font-size: 13px; line-height: 1; cursor: pointer;
    display: inline-flex; align-items: center; justify-content: center;
    transition: background 120ms, border-color 120ms, color 120ms;
    vertical-align: 4px; }
.contract-rules-btn:hover { background: #2d333b; border-color: #1f6feb;
    color: #f0f6fc; }
.criteria-overlay {
    position: fixed; inset: 0; background: rgba(0,0,0,0.55);
    z-index: 100; }
.criteria-modal {
    position: fixed; top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    background: #0d1117; border: 1px solid #30363d; border-radius: 10px;
    /* Sized to look like a proper panel — wide enough for two-column
       label/value rows to breathe, capped so it never overruns the
       viewport on narrow screens. */
    width: 640px; max-width: 92vw;
    max-height: 82vh;
    display: flex; flex-direction: column;
    z-index: 101; box-shadow: 0 16px 56px rgba(0,0,0,0.65); }
/* Fee suffix on the entry-cost cell — same red as the base amount
   (it's also a cash outflow). Keep the cell on one line so the
   "−$0.26 + $0.02" pattern stays scannable horizontally. */
.entry-fee { color: #f85149; font-weight: 400; margin-left: 2px; }
td.num.red, td.num.green { white-space: nowrap; }
/* Slash separator inside the combined Kalshi/My/Edge/EV cells —
   muted so the per-side numbers (which keep their own colour
   spans) stay the visual focus, with the "/" reading as a divider. */
.cell-sep { color: #6e7681; padding: 0 2px; }
/* Align the YES | NO split inside watchlist .num cells AND the
   matching header sub-row so the pipe character lands on the
   same x-coordinate across header + every row. Each side becomes
   a fixed-width inline-block; YES right-aligned, NO left-aligned,
   separator fixed in the middle. */
td.num [data-side='yes'],
th.num [data-side='yes'] {
    display: inline-block; min-width: 3.5em;
    text-align: right; font-variant-numeric: tabular-nums; }
td.num [data-side='no'],
th.num [data-side='no'] {
    display: inline-block; min-width: 3.5em;
    text-align: left; font-variant-numeric: tabular-nums; }
/* Stack the header label on top and the "yes | no" sub-row
   beneath so the sub-row's pipe column-aligns with the data
   pipes in the same column. Lowercase + small + gray so the
   header label visually dominates. */
th.num .th-side-row { display: block; line-height: 1.3;
    margin-top: 2px; font-weight: 400; text-transform: none;
    letter-spacing: 0; }
/* Vertical YES-on-top / NO-on-bottom layout for the side-paired
   columns (My %, Kalshi %, Edge, EV). YES always renders green,
   NO always renders red — the side is conveyed by colour AND
   position, replacing the old horizontal "yes | no" rendering.
   ``.side-yes`` / ``.side-no`` use !important to override the
   per-row tinting rules (.row-bought etc.) that previously
   dimmed cells inside acted-on rows — the side colour should
   stay legible regardless of row state. */
td.num.cell-stack { padding-top: 2px; padding-bottom: 2px;
    line-height: 1.2; }
td.num.cell-stack .side-yes,
td.num.cell-stack .side-no,
td.num.cell-stack .inv-earn,
td.num.cell-stack .inv-cost {
    display: block; text-align: right;
    font-variant-numeric: tabular-nums;
    /* drop the inline-block min-width set by the [data-side]
       rules above — vertical cells don't need horizontal
       alignment between YES and NO. */
    min-width: 0; }
td.num.cell-stack .side-yes { color: #3fb950 !important; }  /* green */
td.num.cell-stack .side-no  { color: #f85149 !important; }  /* red   */
/* Investment column (Active bets) — Potential earnings on top in
   green, Kalshi total cost beneath in red, consolidated into one
   cell per user 2026-07-13. ``.neg`` flips a (rare) negative
   potential-earnings figure to red so the colour still tracks the
   sign, not the row position. */
td.num.cell-stack .inv-earn { color: #3fb950 !important; }
td.num.cell-stack .inv-earn.neg { color: #f85149 !important; }
td.num.cell-stack .inv-cost { color: #f85149 !important; }
/* Bot card drift badge — amber pill that lights up when the model's
   training accuracy and live actual-win-% diverge by >10pp on n≥10
   closed bets. Surfaces "this model may have drifted" as a one-look
   signal without forcing users to compare two cells. */
/* Models panel header — the section title sits on a flex row
   that also accommodates the Pre-game / In-game toggle for sport
   bots. ALL model pages use this header so the title + body sit
   at the same vertical position regardless of whether the toggle
   is present. min-height matches the toggle's natural height so
   the row is the same size with or without it. */
.section .section-header { display: flex; align-items: center;
    justify-content: space-between; gap: 12px;
    padding: 14px 22px 10px; min-height: 32px; }
.section .section-header h2 { padding: 0; margin: 0; }
/* Pre-game / In-game toggle that lives in the Models panel header
   for sport bots. Pills mimic the existing tab-pill idiom but
   live inside one section instead of the page-level tab bar. */
.model-view-toggle { display: inline-flex; gap: 4px;
    padding: 4px; background: #0d1117;
    border: 1px solid #21262d; border-radius: 8px; }
.model-view-toggle .model-view-pill {
    text-decoration: none; color: #8b949e; font-size: 12px;
    font-weight: 600; padding: 6px 14px; border-radius: 5px;
    text-transform: none; letter-spacing: 0.02em; }
.model-view-toggle .model-view-pill:hover { color: #c9d1d9; }
.model-view-toggle .model-view-pill.model-view-active {
    background: #1d232c; color: #f0f6fc;
    box-shadow: 0 1px 2px rgba(0,0,0,0.4); }
/* In-game model pill — appears in the active-bets table next to the
   YES/NO badge when the live model has a confident view of the
   position. EXIT = red (model expects loss), RUN = green (model
   expects win and you should hold), HOLD = yellow (market may be
   overreacting; defer to thresholds). */
.in-game-pill { display: inline-block; margin-left: 6px;
    padding: 1px 6px; border-radius: 4px;
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; line-height: 1.5;
    vertical-align: 2px; cursor: help; }
.in-game-pill.ig-green { background: rgba(63, 185, 80, 0.18);
    color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.35); }
.in-game-pill.ig-red { background: rgba(248, 81, 73, 0.18);
    color: #f85149; border: 1px solid rgba(248, 81, 73, 0.35); }
.in-game-pill.ig-yellow { background: rgba(227, 179, 65, 0.18);
    color: #e3b341; border: 1px solid rgba(227, 179, 65, 0.35); }
.in-game-pill.ig-gray { background: rgba(139, 148, 158, 0.15);
    color: #8b949e; border: 1px solid rgba(139, 148, 158, 0.30); }
.drift-badge { display: inline-block; margin-left: 6px;
    padding: 1px 6px; border-radius: 4px;
    background: rgba(212, 153, 0, 0.18);
    color: #d49900; border: 1px solid rgba(212, 153, 0, 0.35);
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; line-height: 1.5;
    vertical-align: 2px; }
/* Forecast-staleness badge — fires when the bot's stored
   current_gas_price has drifted away from the live Kalshi-implied
   spot by more than $0.20, which usually means the bot is reading
   a stale upstream data feed (EIA publishing lag, missed retrain,
   etc.). Shares the drift-badge typography so the two pills sit
   visually consistent next to the bot name. */
.stale-badge { display: inline-block; margin-left: 6px;
    padding: 1px 6px; border-radius: 4px;
    background: rgba(227, 179, 65, 0.18);
    color: #e3b341; border: 1px solid rgba(227, 179, 65, 0.35);
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; line-height: 1.5;
    vertical-align: 2px; }
/* "×N" badge on history rows where the same ticker was traded
   multiple times (flap-trades collapsed into one row). Small,
   muted-gray so it doesn't compete with WON/LOST coloring. */
.merged-badge { display: inline-block; margin-left: 6px;
    padding: 0 5px; border-radius: 3px;
    background: rgba(139, 148, 158, 0.18);
    color: #8b949e; border: 1px solid rgba(139, 148, 158, 0.3);
    font-size: 9px; font-weight: 700; line-height: 1.4;
    vertical-align: 1px; cursor: help; }
/* Auto-pause notifications panel — surfaced above the bot-card
   grid on Home when the regime monitor has flipped a bot off in the
   recent past. Silent (no DOM) when the audit log is empty so the
   page stays calm on the happy path. */
.notifications-panel { margin-bottom: 14px;
    background: #1d1f24; border: 1px solid #3d342a;
    border-left: 3px solid #e3934d; border-radius: 6px;
    padding: 10px 14px; }
.notifications-head { display: flex; gap: 10px;
    flex-wrap: wrap; align-items: baseline; margin-bottom: 6px; }
.notifications-title { color: #e3934d; font-weight: 700;
    font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.06em; }
.notifications-list { margin: 0; padding: 0; list-style: none;
    display: flex; flex-direction: column; gap: 4px; }
.notifications-list li { display: grid;
    grid-template-columns: 130px 160px 1fr;
    gap: 10px; font-size: 12px; color: #c9d1d9; }
.notification-ts { color: #8b949e; font-family: monospace;
    font-size: 11px; }
.notification-bot { color: #f0f6fc; font-weight: 600; }
.notification-reason { color: #8b949e; }
/* Regime-status pill — sits inline with the bot name on the Home
   tab cards. Three states map to existing summary colours so the
   palette stays consistent: green = edge confirmed, yellow = edge
   eroding, red = anti-edge. */
.regime-pill { display: inline-block; margin-left: 6px;
    padding: 1px 6px; border-radius: 4px;
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; line-height: 1.5;
    vertical-align: 2px; }
.regime-pill.regime-green { background: rgba(63, 185, 80, 0.18);
    color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.35); }
.regime-pill.regime-yellow { background: rgba(227, 179, 65, 0.18);
    color: #e3b341; border: 1px solid rgba(227, 179, 65, 0.35); }
.regime-pill.regime-red { background: rgba(248, 81, 73, 0.18);
    color: #f85149; border: 1px solid rgba(248, 81, 73, 0.35); }
/* The HTML `hidden` attribute applies `display: none` via the UA
   stylesheet (specificity 0,1,0). Our `.criteria-modal { display:
   flex }` rule shares that specificity and wins by source order, so
   the modal kept showing even after JS set `.hidden = true`. These
   attribute selectors (specificity 0,2,0) restore the expected
   behaviour for both the modal and the overlay. */
.criteria-modal[hidden]   { display: none !important; }
.criteria-overlay[hidden] { display: none !important; }
.criteria-modal-head {
    display: flex; align-items: baseline;
    justify-content: space-between;
    padding: 14px 18px; border-bottom: 1px solid #21262d; }
.criteria-modal-head h3 { margin: 0; font-size: 15px; font-weight: 700;
    color: #f0f6fc; }
.criteria-modal-head .ticker { font-family: ui-monospace, SFMono-Regular,
    Consolas, monospace; font-size: 11px; color: #8b949e; }
.criteria-modal-close {
    background: transparent; border: none; color: #8b949e;
    font-size: 20px; cursor: pointer; padding: 0 4px; line-height: 1;
    margin-left: 8px; }
.criteria-modal-close:hover { color: #f0f6fc; }
.criteria-modal-body {
    padding: 18px 22px 22px 22px; overflow-y: auto; font-size: 13px;
    color: #c9d1d9; line-height: 1.55; }
.criteria-modal-body dl { margin: 0; display: grid;
    grid-template-columns: max-content 1fr; gap: 6px 16px; }
.criteria-modal-body dt { color: #8b949e; }
.criteria-modal-body dd { margin: 0; color: #c9d1d9;
    font-variant-numeric: tabular-nums; }
.criteria-modal-body dd.green { color: #3fb950; }
.criteria-modal-body dd.red   { color: #f85149; }
.criteria-modal-body dd.gray  { color: #6e7681; }
.criteria-modal-body .crit-section { margin-top: 14px; }
.criteria-modal-body .crit-section h4 {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
    color: #8b949e; font-weight: 600; margin: 0 0 8px 0; }

/* Redesigned rule card + rule row for the sport-bot criteria modal.
   Each rule renders as a two-line row: label + value chip on line 1,
   muted description on line 2. Cards group related rules under a
   coloured header so the four sections (edge / market / risk / exit)
   scan at a glance. */
.criteria-modal-body .crit-card {
    background: #11161d; border: 1px solid #21262d; border-radius: 8px;
    padding: 14px 16px; margin: 0 0 12px 0; }
.criteria-modal-body .crit-card:last-child { margin-bottom: 0; }
.criteria-modal-body .crit-card-head {
    display: flex; align-items: center; gap: 8px;
    margin: 0 0 12px 0; padding: 0 0 10px 0;
    border-bottom: 1px solid #21262d; }
.criteria-modal-body .crit-card-icon {
    width: 22px; height: 22px; border-radius: 5px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; letter-spacing: 0; }
.criteria-modal-body .crit-card-icon.edge   { background: rgba(63,185,80,0.15);  color: #3fb950; }
.criteria-modal-body .crit-card-icon.market { background: rgba(88,166,255,0.15); color: #58a6ff; }
.criteria-modal-body .crit-card-icon.risk   { background: rgba(210,153,34,0.15); color: #e3b341; }
.criteria-modal-body .crit-card-icon.exit   { background: rgba(163,113,247,0.18); color: #a371f7; }
.criteria-modal-body .crit-card-title {
    font-size: 13px; font-weight: 600; color: #f0f6fc;
    letter-spacing: 0.01em; }
.criteria-modal-body .crit-card-sub {
    margin-left: auto; font-size: 11px; color: #8b949e; }
.criteria-modal-body .crit-rule {
    padding: 8px 0; border-top: 1px solid transparent; }
.criteria-modal-body .crit-rule + .crit-rule {
    border-top-color: #1a1f26; }
.criteria-modal-body .crit-rule-head {
    display: flex; align-items: baseline; gap: 10px;
    justify-content: space-between; }
.criteria-modal-body .crit-rule-label {
    font-size: 12.5px; color: #f0f6fc; font-weight: 600; }
.criteria-modal-body .crit-rule-desc {
    margin: 3px 0 0 0; font-size: 11.5px; color: #8b949e; line-height: 1.5; }
.criteria-modal-body .crit-chip {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    background: #21262d; border: 1px solid #30363d;
    font-size: 12px; font-weight: 600; color: #f0f6fc;
    font-variant-numeric: tabular-nums; white-space: nowrap; }
.criteria-modal-body .crit-chip.pos  { background: rgba(63,185,80,0.14); border-color: rgba(63,185,80,0.35); color: #3fb950; }
.criteria-modal-body .crit-chip.neg  { background: rgba(248,81,73,0.14); border-color: rgba(248,81,73,0.35); color: #f85149; }
.criteria-modal-body .crit-chip.info { background: rgba(88,166,255,0.14); border-color: rgba(88,166,255,0.35); color: #58a6ff; }

/* Tennis-specific hero at the top of the modal — three stat pills
   showing the essence of the bot's approach so the user gets the
   headline before they read any bullets. */
.criteria-modal-body .crit-hero {
    background: linear-gradient(135deg, rgba(63,185,80,0.10),
        rgba(88,166,255,0.08));
    border: 1px solid #21262d; border-radius: 8px;
    padding: 14px 16px; margin: 0 0 14px 0; }
.criteria-modal-body .crit-hero-lead {
    font-size: 12px; color: #8b949e; margin: 0 0 10px 0;
    letter-spacing: 0.02em; }
.criteria-modal-body .crit-hero-lead b { color: #f0f6fc; }
.criteria-modal-body .crit-hero-stats {
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
.criteria-modal-body .crit-hero-stat {
    background: #0d1117; border: 1px solid #21262d; border-radius: 6px;
    padding: 8px 10px; }
.criteria-modal-body .crit-hero-stat-label {
    font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em;
    color: #8b949e; margin: 0 0 3px 0; }
.criteria-modal-body .crit-hero-stat-value {
    font-size: 15px; font-weight: 700; color: #f0f6fc;
    font-variant-numeric: tabular-nums; }
/* Compact source pill in the modal head (was a full-width banner). */
.criteria-modal-body .crit-source-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; padding: 3px 8px; border-radius: 4px;
    margin: 0 0 12px 0; }
.criteria-modal-body .crit-source-pill.live {
    color: #3fb950; background: rgba(63,185,80,0.10);
    border: 1px solid rgba(63,185,80,0.30); }
.criteria-modal-body .crit-source-pill.fallback {
    color: #e3b341; background: rgba(227,179,65,0.10);
    border: 1px solid rgba(227,179,65,0.30); }
.criteria-modal-body .crit-source-pill .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: currentColor; }
.criteria-modal-body .crit-foot {
    margin-top: 12px; padding: 10px 12px; font-size: 11px;
    color: #8b949e; background: #0d1117; border: 1px solid #21262d;
    border-radius: 6px; line-height: 1.55; }

/* Per-row Kalshi-rules info button — small circle-i sitting in the
   Rules column. Same visual as the ev-info-btn but centered inside
   its own td so the popover anchor is predictable. */
.rules-info-btn {
    display: inline-flex; align-items: center; justify-content: center;
    width: 18px; height: 18px; padding: 0;
    border-radius: 50%; border: 1px solid #30363d;
    background: #0d1117; color: #8b949e;
    font-family: ui-serif, Georgia, serif;
    font-size: 11px; font-style: italic; font-weight: 700;
    line-height: 1; cursor: pointer; }
.rules-info-btn:hover { background: #1c222b; color: #58a6ff;
    border-color: #1f6feb; }
.rules-info-popover {
    position: fixed; z-index: 90; max-width: 420px;
    background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
    padding: 14px 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    color: #c9d1d9; font-size: 12.5px; line-height: 1.6;
    font-family: inherit; letter-spacing: 0; text-transform: none;
    text-align: left; font-weight: normal;
    max-height: 60vh; overflow-y: auto; }
.rules-info-popover[hidden] { display: none !important; }
.rules-info-popover h5 { margin: 0 0 8px 0; font-size: 12.5px;
    font-weight: 600; color: #f0f6fc; letter-spacing: 0.02em;
    text-transform: uppercase; }
.rules-info-popover .rules-body { white-space: pre-wrap; }

/* Inline info button next to the EV column header — opens a small
   popover explaining how EV is derived. Same visual language as the
   criteria-rules-btn but smaller / thinner so it sits happily next to
   a th label. */
.ev-info-btn {
    display: inline-flex; align-items: center; justify-content: center;
    width: 14px; height: 14px; padding: 0; margin: 0 0 0 5px;
    border-radius: 50%; border: 1px solid #30363d;
    background: #0d1117; color: #8b949e;
    font-family: ui-serif, Georgia, serif;
    font-size: 10px; font-style: italic; font-weight: 700;
    line-height: 1; cursor: pointer; vertical-align: 1px; }
.ev-info-btn:hover { background: #1c222b; color: #58a6ff;
    border-color: #1f6feb; }
.ev-info-popover {
    position: fixed; z-index: 90; max-width: 320px;
    background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
    padding: 12px 14px; box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    color: #c9d1d9; font-size: 12px; line-height: 1.55;
    font-family: inherit; letter-spacing: 0; text-transform: none;
    text-align: left; font-weight: normal; }
.ev-info-popover[hidden] { display: none !important; }
.ev-info-popover h5 { margin: 0 0 8px 0; font-size: 12.5px;
    font-weight: 600; color: #f0f6fc; letter-spacing: 0; }
.ev-info-popover code { background: #161b22; padding: 1px 5px;
    border-radius: 3px; color: #c9d1d9; font-size: 11.5px; }
.ev-info-popover .ev-info-formula {
    margin: 8px 0; padding: 8px 10px; background: #161b22;
    border: 1px solid #21262d; border-radius: 5px;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 11.5px; line-height: 1.5; color: #f0f6fc;
    white-space: pre-wrap; }
.ev-info-popover .gray { color: #8b949e; }
/* Per-bot performance cards on the Performance tab. Cards align in a
   grid (auto-fit so they reflow at narrow widths) and are clickable —
   the whole card is an anchor to that bot's Watchlist tab. */
.bot-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    grid-auto-rows: 1fr;
    gap: 14px;
}
.bot-card { display: flex; flex-direction: column;
    background: #0d1117; border: 1px solid #21262d;
    border-radius: 8px; padding: 14px 16px;
    color: inherit; text-decoration: none;
    transition: border-color 120ms, background 120ms,
                transform 120ms; }
.bot-card:hover {
    border-color: #1f6feb; background: #11161d;
    transform: translateY(-1px);
}
.bot-card-head { display: flex; align-items: flex-start;
    justify-content: space-between; gap: 12px;
    border-bottom: 1px solid #21262d; padding-bottom: 10px;
    margin-bottom: 10px;
    /* Reserve a fixed height for the name + ticker block. Without
       this, adding / removing the PAUSED badge bumps the card by
       ~18px when the toggle flips. */
    min-height: 48px; }
.bot-card-head-left { display: flex; flex-direction: column;
    gap: 2px; min-width: 0; }
.bot-card-head .bot-name { font-size: 14px; font-weight: 700;
    color: #f0f6fc; letter-spacing: -0.2px;
    display: inline-flex; align-items: center; gap: 6px;
    /* PAUSED badge inserts/removes inline next to the name — line
       height clamp keeps the row a constant height regardless. */
    line-height: 22px; }
.bot-card-head .bot-meta { font-size: 10px; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.04em;
    margin-top: 2px; }
/* On/off toggle in the card header. The track + knob is pure CSS
   styled to feel like the iOS-style switches the rest of the
   industry uses — green when on, gray when off, with a 200ms slide
   on the knob so the click feels responsive. */
.bot-toggle { all: unset; cursor: pointer; display: inline-flex;
    align-items: center; gap: 6px; padding: 4px 8px;
    border-radius: 999px; background: transparent;
    border: 1px solid transparent; }
.bot-toggle:hover { background: #1d232c; border-color: #30363d; }
.bot-toggle .bot-toggle-track { position: relative;
    width: 32px; height: 18px; border-radius: 999px;
    background: #30363d; transition: background 160ms; }
.bot-toggle .bot-toggle-knob { position: absolute;
    top: 2px; left: 2px; width: 14px; height: 14px;
    border-radius: 50%; background: #f0f6fc;
    transition: transform 200ms cubic-bezier(0.4, 0, 0.2, 1); }
.bot-toggle[data-enabled='1'] .bot-toggle-track { background: #2da44e; }
.bot-toggle[data-enabled='1'] .bot-toggle-knob { transform: translateX(14px); }
.bot-toggle .bot-toggle-label { font-size: 10px; font-weight: 700;
    color: #8b949e; letter-spacing: 0.05em;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.bot-toggle[data-enabled='1'] .bot-toggle-label { color: #2da44e; }
/* Paused card — dim the content + drop the hover lift so the bot
   reads as "off" at a glance without disappearing entirely. */
.bot-card-paused { opacity: 0.55; border-style: dashed; }
.bot-card-paused:hover { transform: none; border-color: #30363d;
    background: #0d1117; }
.paused-badge { display: inline-block; padding: 1px 6px;
    border-radius: 4px; background: rgba(139, 148, 158, 0.2);
    color: #c9d1d9; font-size: 9px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; }
.bot-card dl { margin: 0; display: grid;
    grid-template-columns: max-content 1fr max-content 1fr;
    gap: 4px 12px;
    font-size: 12px; line-height: 1.45; }
.bot-card dt { color: #8b949e; }
.bot-card dd { margin: 0; color: #c9d1d9;
    font-variant-numeric: tabular-nums; text-align: right;
    font-weight: 500; }
/* High-specificity + !important so the green/red gain-loss colors
   land regardless of any other .green/.red cascade rules. */
.bot-card dl dd.green { color: #3fb950 !important; font-weight: 600; }
.bot-card dl dd.red   { color: #f85149 !important; font-weight: 600; }
.bot-card dl dd.gray  { color: #6e7681 !important; }
.bot-card-foot {
    margin-top: auto; padding-top: 10px;
    border-top: 1px solid #21262d;
    font-size: 10px; color: #6e7681;
    display: flex; justify-content: space-between;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.bot-card-foot .arrow { color: #8b949e; }
/* Watchlist row that fails one or more validations (horizon mismatch,
   wide spread, edge<cost, etc.). Rendered visible but de-emphasized.
   EV used to escape this rule via a nth-last-child(2) exception —
   per user request the EV column now dims with the rest of the row
   so only the held positions pop in full brightness. */
tr.row-suspect td { opacity: 0.55; }
/* Watchlist row matching the strike the bot currently holds an open
   position on. Non-coloured cells go pure white; the side-paired
   cells (My %, Kalshi %, Edge, EV) keep their full-brightness
   green/red so YES vs NO stays legible. Subtle left rail + bold
   ticker stay as the secondary cue; the Verdict column's HOLDING
   YES / HOLDING NO badge conveys the bet direction. Wins
   specificity over row-suspect so a held position is never dimmed. */
tr.row-bought td { opacity: 1 !important; color: #ffffff !important;
    font-weight: 600 !important; }
tr.row-bought td:first-child { border-left: 3px solid #8b949e; }
tr.row-bought td a.ticker-link,
tr.row-bought td.mono { color: #ffffff; }
/* Preserve the red / green colouring on cost / earnings cells even
   inside a row-bought row — otherwise the row's white-color override
   above wins and Kalshi total cost / Potential earnings both render
   white. Higher-specificity selector with the same !important cast
   flips them back to their intended colours. */
tr.row-bought td.num.red   { color: #f85149 !important; }
tr.row-bought td.num.green { color: #56d364 !important; }
/* Watchlist table: fixed scrolling viewport so the strike list never
   pushes the rest of the page off-screen. Sticky header keeps the
   column labels in view as the user scrolls. */
.watchlist-scroll { max-height: 360px; overflow-y: auto;
    border: 1px solid #21262d; border-radius: 6px;
    margin-top: 4px; }
.watchlist-scroll table { margin: 0; border: none; }
.watchlist-scroll thead th {
    position: sticky; top: 0; z-index: 1;
    background: #161b22; box-shadow: 0 1px 0 #30363d;
}
.section h2 .small { text-transform: none; letter-spacing: 0; font-size: 11px; font-weight: 400; }
/* Watchlist hero — Kalshi-style market header above the strikes table.
   Layout mirrors the live Kalshi market page: title + countdown on top,
   then a big current-value, % change, and total volume row, then the
   underlying chart. */
.wl-hero { background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
    padding: 16px 18px; margin-bottom: 18px; }
.wl-hero-top { display: flex; align-items: baseline; justify-content: space-between;
    gap: 12px; margin-bottom: 12px; }
.wl-hero-stats { display: flex; align-items: baseline; gap: 14px;
    flex-wrap: wrap; }
/* Watchlist chart hero top-left — the forecast price + change
   indicator that lived here previously were removed per user
   request. The replacement is the volume of the contract the
   chart line represents (atm market), matching the same large-
   number + small-label visual rhythm. */
.wl-hero-volume { font-size: 24px; font-weight: 700; color: #f0f6fc;
    letter-spacing: -0.3px; }
.wl-hero-volume-label { font-size: 12px; font-weight: 500; color: #8b949e;
    text-transform: lowercase; margin-left: 4px; letter-spacing: 0.02em; }
.wl-hero-mtc { font-size: 12px; color: #8b949e; flex: 0 0 auto; }
.wl-hero-mtc .label { color: #8b949e; text-transform: uppercase;
    letter-spacing: 0.04em; margin-right: 6px; font-size: 10px; }
.wl-hero-mtc .value { color: #c9d1d9; font-weight: 600; font-size: 13px; }
/* Hover crosshair on the underlying chart. JS draws the vertical line
   inside the SVG and positions this tooltip via inline `left:`. */
.wl-chart-wrap { position: relative; }
.wl-chart-tooltip {
    position: absolute; top: -8px; transform: translateX(-50%);
    background: #161b22; color: #c9d1d9;
    border: 1px solid #30363d; border-radius: 4px;
    padding: 4px 9px; font-size: 11px; font-weight: 500;
    pointer-events: none; white-space: nowrap; z-index: 2;
    text-align: center; line-height: 1.35;
}
.wl-chart-tooltip .wl-chart-tip-time { color: #8b949e; font-size: 10px; }
.wl-chart-tooltip .wl-chart-tip-value { color: #f0f6fc; font-size: 13px;
    font-weight: 600; }
/* Scroll container around the Summary's "Active bets" table. The
   per-bot active-bets list lower on the Watchlist tab keeps its
   natural height — only the global aggregate at the top of Home
   gets clamped. Matches the .watchlist-scroll idiom used for the
   strike-ladder table. */
.summary-active-scroll { max-height: 280px; overflow-y: auto;
    border: 1px solid #30363d; border-radius: 6px;
    background: #0d1117; }
.summary-active-scroll table { margin: 0; }
.summary-active-scroll thead th { position: sticky; top: 0;
    background: #161b22; z-index: 1; }
/* Watchlist-tab Active bets scroller. Mirrors .watchlist-scroll
   (the strike-ladder table below) so the two read as a stacked
   pair, but capped at a smaller height since it's the bet list
   not the full ladder. Section-grey background contrasts the
   near-black chart panel directly above it. */
.watchlist-active-scroll { max-height: 220px; overflow-y: auto;
    border: 1px solid #21262d; border-radius: 6px;
    background: #161b22; margin-top: 4px;
    margin-bottom: 14px; }
.watchlist-active-scroll table { margin: 0; border: none; }
.watchlist-active-scroll thead th { position: sticky; top: 0;
    z-index: 1; background: #1d232c;
    box-shadow: 0 1px 0 #30363d; }
/* History tab scroll container — taller than the Summary's active
   bets scroll since the History tab is dedicated to this table.
   ~14 rows visible before the user scrolls. */
.history-scroll { max-height: 640px; overflow-y: auto;
    border: 1px solid #30363d; border-radius: 6px;
    background: #0d1117; margin-top: 10px; }
.history-scroll table { margin: 0; }
.history-scroll thead th { position: sticky; top: 0;
    background: #161b22; z-index: 1; }
/* HTML <details> wrappers inside the scroll container — the
   collapsed "show more" rows are invisible until expanded; keep the
   summary line sticky so it stays accessible when scrolled. */
.history-scroll details > summary { position: sticky; bottom: 0;
    background: #161b22; padding: 6px 10px; cursor: pointer;
    border-top: 1px solid #30363d; }
/* History tab P&L attribution — small breakdown tables in a two-up
   grid. Each panel has its own h3 subhead and a compact table. */
.attribution-grid { display: grid; gap: 14px;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    margin-bottom: 14px; }
.attribution-panel { background: #0d1117;
    border: 1px solid #21262d; border-radius: 6px;
    padding: 10px 14px; }
.attribution-panel h3.subhead { margin-top: 0; margin-bottom: 6px; }
.attribution-panel table { font-size: 12px; }
.attribution-panel th, .attribution-panel td { padding: 6px 8px; }
/* History tab P&L line chart — sits between the headline cards and
   the ledger table. The wrap is `position: relative` so the empty-
   state overlay can be absolute-positioned over the SVG frame. */
.history-chart-section { margin-top: 14px; }
/* Inline toolbar above the chart: chart title on the left, period
   selector on the right. Suppress the .bot-filter-bar divider so
   the filter reads as a chart control, not a section break. */
.history-chart-toolbar { display: flex; align-items: center;
    justify-content: space-between; gap: 12px;
    margin: 0 0 8px 0; flex-wrap: wrap; }
.history-chart-toolbar .history-chart-title {
    color: #c9d1d9; font-size: 14px; font-weight: 600; }
.history-chart-toolbar .bot-filter-bar { padding: 0; margin: 0;
    border-bottom: none; }
.history-chart-wrap { position: relative;
    border: 1px solid #30363d; border-radius: 6px;
    background: #0d1117; padding: 8px 4px 4px 4px; }
"""
