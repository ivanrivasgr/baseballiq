"""
dashboard/matchup_board.py
===========================
Slate matchup board — the Kasper-style grid: pick a game from the slate
strip, see every projected hitter vs the opposing probable starter with
color-graded Statcast columns, FORM arrows, and (when a model-scores file
exists) our calibrated P(HR) where their proprietary KHR sits.

Reads the parquet produced by pipeline/gold/matchup_board.py — no database
at runtime, same deploy story as dashboard/app.py.

Run:
    streamlit run dashboard/matchup_board.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

GOLD_DIR = Path(__file__).resolve().parents[1] / "data" / "gold"

st.set_page_config(page_title="BaseballIQ — Matchups", page_icon="⚾", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
div[data-testid="stHorizontalBlock"] button {width: 100%;}
</style>
""", unsafe_allow_html=True)


# ── Load ────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_boards() -> dict[str, pd.DataFrame]:
    boards = {}
    for path in sorted(GOLD_DIR.glob("matchup_board_*.parquet")):
        slate_date = path.stem.replace("matchup_board_", "")
        boards[slate_date] = pd.read_parquet(path)
    return boards


boards = load_boards()
st.title("⚾ Matchups")

if not boards:
    st.info(
        "No matchup boards found. Build one with:\n\n"
        "```\npython -m pipeline.gold.matchup_board --date YYYY-MM-DD\n```"
    )
    st.stop()

slate_date = st.selectbox("Slate", sorted(boards, reverse=True))
board = boards[slate_date]

# ── Game strip ──────────────────────────────────────────────────────────────────

games = (
    board.groupby("game_pk")
         .agg(away=("batter_team", lambda s: s[~board.loc[s.index, "is_home"]].iloc[0]
                    if (~board.loc[s.index, "is_home"]).any() else "?"),
              home=("batter_team", lambda s: s[board.loc[s.index, "is_home"]].iloc[0]
                    if board.loc[s.index, "is_home"].any() else "?"),
              day_night=("day_night", "first"))
         .reset_index()
)

labels = {row.game_pk: f"{row.away} @ {row.home}" for row in games.itertuples()}
cols = st.columns(min(len(labels), 15))
if "selected_game" not in st.session_state:
    st.session_state.selected_game = games.iloc[0].game_pk
for col, (pk, label) in zip(cols, labels.items()):
    if col.button(label, key=f"game_{pk}"):
        st.session_state.selected_game = pk

game_pk = st.session_state.selected_game
view = board[board["game_pk"] == game_pk].copy()

side = st.radio("Lineup", sorted(view["batter_team"].unique()), horizontal=True)
view = view[view["batter_team"] == side]

starter = view["opp_starter_name"].iloc[0] if len(view) else "TBD"
throws = view["opp_starter_throws"].iloc[0] if len(view) else None
st.subheader(f"vs {starter}" + (f" ({throws}HP)" if isinstance(throws, str) else ""))

# ── Grid ────────────────────────────────────────────────────────────────────────

DISPLAY = {
    "batter_name": "HITTER", "stand": "B", "recent_slot": "SLOT",
    "hr_prob": "P(HR)", "form_pct": "FORM", "form_arrow": "",
    "pit_365d": "PIT", "bip_365d": "BIP",
    "iso_365d": "ISO", "xwoba_365d": "XWOBA", "xwobacon_365d": "XWOBAC",
    "sws_365d": "SWS%", "pbrl_365d": "PBRL%", "brl_365d": "BRL%",
    "swsp_365d": "SWSP%", "fb_365d": "FB%", "hh_365d": "HH%", "la_365d": "LA",
}
PCT_COLS = ["sws_365d", "pbrl_365d", "brl_365d", "swsp_365d", "fb_365d", "hh_365d"]
GRADIENT_UP = ["hr_prob", "iso_365d", "xwoba_365d", "xwobacon_365d",
               "pbrl_365d", "brl_365d", "swsp_365d", "fb_365d", "hh_365d", "form_pct"]
GRADIENT_DOWN = ["sws_365d"]        # more whiffs = worse for the hitter

grid = view[[c for c in DISPLAY if c in view.columns]].rename(columns=DISPLAY)

styler = grid.style
for src, shown in DISPLAY.items():
    if shown not in grid.columns or grid[shown].dropna().empty:
        continue
    if src in GRADIENT_UP:
        styler = styler.background_gradient(cmap="RdYlGn", subset=[shown])
    elif src in GRADIENT_DOWN:
        styler = styler.background_gradient(cmap="RdYlGn_r", subset=[shown])

fmt = {DISPLAY[c]: "{:.1%}" for c in PCT_COLS if DISPLAY[c] in grid.columns}
fmt |= {"ISO": "{:.3f}", "XWOBA": "{:.3f}", "XWOBAC": "{:.3f}",
        "P(HR)": "{:.1%}", "FORM": "{:.0f}%", "LA": "{:.1f}",
        "PIT": "{:,.0f}", "BIP": "{:,.0f}"}
styler = styler.format(fmt, na_rep="–")

st.dataframe(styler, use_container_width=True, height=520, hide_index=True)

st.caption(
    "All columns computed from public Statcast/MLB data as of the morning of "
    f"{slate_date} (strictly prior days only). P(HR) is the calibrated model "
    "probability — the number the edge and Voss Edge tier layers consume. "
    "FORM = 30-day xwOBAcon vs the hitter's own 365-day baseline (50% = steady)."
)
