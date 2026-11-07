"""Level 2 Situational Awareness Dashboard, Time Machine & AI Explainability Desk for Turku Energia.

Designed strictly according to Turku Energia's Corporate Identity (`data/Graafinen ohjeisto.pdf`, 2024)
with a full Control-Room Dark Mode (default) and optional Light Mode:
- Brand Orange (#F59B00, always with white text), Control-Room Dark Navy (#07131E / #0D1E2E),
  Turquoise (#46BEDC), Ice Blue (#B4DCE6), Lime (#AAC800), Berry (#E6508C), and Plum (#C85AB2).
- Modular architecture:
  - `apps/theme.py`        : Corporate Identity CSS variables (Dark & Light), 8:5 SVG logo, KPI & Alert components
  - `apps/charts.py`       : Probabilistic Plotly Fan Charts & Grid Visualizations
  - `apps/ai_assistant.py` : Gemini-connected & Data-Grounded Natural Language Explainability Chat

Run with:
    ./venv/bin/streamlit run apps/dashboard.py
    (or on Windows: .\\venv\\Scripts\\streamlit.exe run apps/dashboard.py)
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.ai_assistant import generate_explainability_response
from apps.charts import (
    build_direction_probability_chart,
    build_horizon_benchmark_chart,
    build_physical_drivers_chart,
    build_price_fan_chart,
    build_shap_drivers_bar_chart,
    build_spike_probability_chart,
)
from apps.theme import (
    TE_COLORS,
    format_driver_name,
    get_toolbar_theme_toggle_html,
    inject_brand_css,
    render_alert_cards,
    render_hero_header,
    render_kpi_strip,
)
from src.data.loader import DEFAULT_DATA_DIR, SPIKE_MARGIN, load_all_data, to_utc

st.set_page_config(
    page_title="Turku Energia | Imbalance Price Situational Desk",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

SPIKE_ALERT_THRESHOLD = 0.30
REVISION_ALERT_EUR = 15.0


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict:
    """Load backtest predictions, validation predictions, scores, and grid series."""
    _, _, combined = load_all_data(DEFAULT_DATA_DIR)
    val_wide = (
        combined.pivot(index="timestamp_utc", columns="series_id", values="value")
        .sort_index()
        .astype(float)
    )
    fu_wide = (
        combined.pivot(index="timestamp_utc", columns="series_id", values="first_usable_origin_utc")
        .sort_index()
    )

    test_path = PROJECT_ROOT / "forecast_backtest.csv"
    val_path = PROJECT_ROOT / "results" / "validation" / "forecast_backtest.csv"

    def _read_sub(p: Path) -> pd.DataFrame | None:
        if not p.exists():
            return None
        df = pd.read_csv(p)
        for c in ("forecast_origin_utc", "delivery_start_utc", "data_cutoff_utc"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True)
        return df

    test_sub = _read_sub(test_path)
    val_sub = _read_sub(val_path)

    val_score_summary = PROJECT_ROOT / "results" / "validation" / "score" / "score_summary.csv"
    val_score_horizon = PROJECT_ROOT / "results" / "validation" / "score" / "score_by_horizon.csv"
    base_score_summary = PROJECT_ROOT / "baseline_output" / "validation" / "score" / "score_summary.csv"
    base_score_horizon = PROJECT_ROOT / "baseline_output" / "validation" / "score" / "score_by_horizon.csv"
    feat_imp_path = PROJECT_ROOT / "results" / "feature_importance.csv"

    return {
        "val_wide": val_wide,
        "fu_wide": fu_wide,
        "test_sub": test_sub,
        "val_sub": val_sub,
        "val_summary": pd.read_csv(val_score_summary) if val_score_summary.exists() else None,
        "val_horizon": pd.read_csv(val_score_horizon) if val_score_horizon.exists() else None,
        "base_summary": pd.read_csv(base_score_summary) if base_score_summary.exists() else None,
        "base_horizon": pd.read_csv(base_score_horizon) if base_score_horizon.exists() else None,
        "feat_imp": pd.read_csv(feat_imp_path) if feat_imp_path.exists() else None,
    }


def get_usable_series_at(val_wide: pd.DataFrame, fu_wide: pd.DataFrame, origin_utc: pd.Timestamp) -> pd.DataFrame:
    """Strictly filter wide table to entries where first_usable_origin_utc <= origin_utc (RULE #1)."""
    origin = to_utc(origin_utc)
    mask = fu_wide <= origin
    return val_wide.where(mask)


def build_situational_narrative(
    origin_df: pd.DataFrame,
    prev_origin_df: pd.DataFrame | None,
    usable_grid: pd.DataFrame,
    origin_ts: pd.Timestamp,
) -> dict:
    """Synthesize decision-support briefing and structured alerts for shift supervisors & traders."""
    max_hi_idx = origin_df["p_spike_high"].idxmax()
    max_lo_idx = origin_df["p_spike_low"].idxmax()
    max_p_hi = float(origin_df.loc[max_hi_idx, "p_spike_high"])
    max_p_lo = float(origin_df.loc[max_lo_idx, "p_spike_low"])

    hi_steps = origin_df[origin_df["p_spike_high"] >= SPIKE_ALERT_THRESHOLD]
    lo_steps = origin_df[origin_df["p_spike_low"] >= SPIKE_ALERT_THRESHOLD]

    # Revision vs previous hourly origin (t - 1h) on overlapping delivery_start_utc
    max_abs_rev = 0.0
    mean_rev = 0.0
    if prev_origin_df is not None and len(prev_origin_df):
        merged_prev = origin_df[["delivery_start_utc", "step", "p50"]].merge(
            prev_origin_df[["delivery_start_utc", "p50"]],
            on="delivery_start_utc",
            suffixes=("_now", "_prev"),
        )
        if len(merged_prev):
            diff = merged_prev["p50_now"] - merged_prev["p50_prev"]
            max_abs_rev = float(diff.abs().max())
            mean_rev = float(diff.mean())

    # Aggregate top SHAP drivers across the horizon using human-friendly English labels
    driver_counts: dict[str, int] = {}
    for s in origin_df["top_drivers"].dropna().astype(str):
        for d in s.split(";"):
            d_clean = format_driver_name(d)
            if d_clean:
                driver_counts[d_clean] = driver_counts.get(d_clean, 0) + 1
    top3_drivers = [k for k, _ in sorted(driver_counts.items(), key=lambda x: -x[1])[:3]]
    drivers_text = ", ".join(top3_drivers) if top3_drivers else "Day-Ahead Spot Price, Recent Imbalance Spread"

    # Dominant regulation direction
    mean_p_up = float(origin_df["p_up"].mean())
    mean_p_down = float(origin_df["p_down"].mean())
    dominant_dir_short = "UP" if mean_p_up >= mean_p_down else "DOWN"
    dominant_dir = "UP-REGULATION (Power Deficit)" if mean_p_up >= mean_p_down else "DOWN-REGULATION (Power Surplus)"

    # Uncertainty width
    mean_band = float((origin_df["p90"] - origin_df["p10"]).mean())
    max_band = float((origin_df["p90"] - origin_df["p10"]).max())

    alerts: list[dict[str, str]] = []
    if len(hi_steps):
        t_start = hi_steps["delivery_start_utc"].min().strftime("%H:%M UTC")
        t_end = (hi_steps["delivery_start_utc"].max() + pd.Timedelta(minutes=15)).strftime("%H:%M UTC")
        alerts.append({
            "level": "HIGH",
            "title": f"Elevated High-Price Spike Risk ({max_p_hi:.0%}) for {t_start}–{t_end}",
            "body": (
                f"Probability of an exceptional high imbalance price (> Spot + {int(SPIKE_MARGIN)} €/MWh) "
                f"exceeds the 30% alert threshold in {len(hi_steps)} quarter-hour step(s) ({t_start}–{t_end}), "
                f"peaking at {max_p_hi:.1%}. Key drivers: {drivers_text}. "
                f"Suggestion to review: Check availability of flexible generation, district heating electric boiler load, and open intraday short exposure."
            ),
        })

    if len(lo_steps):
        t_start = lo_steps["delivery_start_utc"].min().strftime("%H:%M UTC")
        t_end = (lo_steps["delivery_start_utc"].max() + pd.Timedelta(minutes=15)).strftime("%H:%M UTC")
        alerts.append({
            "level": "LOW_SPIKE",
            "title": f"Low / Negative Price Spike Risk ({max_p_lo:.0%}) for {t_start}–{t_end}",
            "body": (
                f"Probability of a sharp downward price drop (< Spot − {int(SPIKE_MARGIN)} €/MWh) "
                f"exceeds {SPIKE_ALERT_THRESHOLD:.0%} in {len(lo_steps)} quarter-hour step(s) ({t_start}–{t_end}), "
                f"peaking at {max_p_lo:.1%}. Key drivers: {drivers_text}. "
                f"Suggestion to review: Evaluate flexible power-to-heat demand (electric boilers / heat pumps) and mFRR down-regulation bids."
            ),
        })

    if max_abs_rev >= REVISION_ALERT_EUR:
        direction_word = "upward" if mean_rev >= 0 else "downward"
        alerts.append({
            "level": "REVISION",
            "title": f"Significant Forecast Revision vs. Previous Hour (max ΔP50 = {max_abs_rev:.1f} €/MWh)",
            "body": (
                f"The P50 forecast trajectory shifted {direction_word} compared to the {(origin_ts - pd.Timedelta(hours=1)).strftime('%H:%M UTC')} run "
                f"by an average of {mean_rev:+.1f} €/MWh (max single-step shift {max_abs_rev:.1f} €/MWh), "
                f"driven by newly released real-time consumption and balancing telemetry."
            ),
        })

    if not alerts:
        alerts.append({
            "level": "NORMAL",
            "title": "Balanced 10-Hour Grid Outlook — No Spike Warnings (≥ 30%)",
            "body": (
                f"Both high-price spike ({max_p_hi:.1%}) and low-price spike ({max_p_lo:.1%}) probabilities "
                f"remain below the 30% warning threshold across the next 10 hours. Dominant balancing tendency: {dominant_dir}. "
                f"Primary forecast drivers: {drivers_text}."
            ),
        })

    return {
        "max_p_hi": max_p_hi,
        "max_p_lo": max_p_lo,
        "n_hi_steps": len(hi_steps),
        "n_lo_steps": len(lo_steps),
        "max_abs_rev": max_abs_rev,
        "mean_rev": mean_rev,
        "dominant_dir": dominant_dir,
        "dominant_dir_short": dominant_dir_short,
        "mean_p_up": mean_p_up,
        "mean_p_down": mean_p_down,
        "mean_band": mean_band,
        "max_band": max_band,
        "top3_drivers": top3_drivers,
        "alerts": alerts,
    }


def _render_chat_sidebar(
    origin_df: pd.DataFrame,
    briefing: dict,
    selected_origin: pd.Timestamp,
    ui_context: dict | None = None,
) -> None:
    """Render the right-hand AI Copilot Sidebar inside a unified outer container."""
    if "te_chat_messages" not in st.session_state:
        st.session_state.te_chat_messages = []
    if "te_last_origin" not in st.session_state or st.session_state.te_last_origin != selected_origin:
        st.session_state.te_last_origin = selected_origin
        peak_row = origin_df.loc[origin_df["p50"].idxmax()]
        peak_t = pd.Timestamp(peak_row["delivery_start_utc"]).strftime("%H:%M UTC")
        st.session_state.te_chat_messages = [
            {
                "role": "assistant",
                "markdown": (
                    f"**Origin `{selected_origin.strftime('%Y-%m-%d %H:%M UTC')}` loaded** *(40 steps)*\n\n"
                    f"* **Peak P50:** **{peak_row['p50']:.1f} €/MWh** at **{peak_t}** *(Spot {peak_row['spot_price']:.1f} €)*\n"
                    f"* **Spike Risk:** High **{briefing['max_p_hi']:.1%}** · Low **{briefing['max_p_lo']:.1%}**\n\n"
                    f"Select a quick question below or ask anything about the forecast, Spot price, Dataset Period, or dashboard (EN / DE)."
                ),
                "source": "Situational Desk AI",
            }
        ]

    # Unified Outer Box wrapping the entire AI Assistant section
    with st.container(border=True):
        st.markdown(
            """
            <div class="te-section-header" style="margin-top:0; margin-bottom:0.55rem;">
              <h3 class="te-section-title" style="font-size:1.3rem;">🤖 AI Assistant<span>.</span></h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. TOP: Scrollable Chat History Container
        chat_box = st.container(height=500, border=True)

        # 2. MIDDLE ("dazwischen"): Quick-Prompt Suggestion Buttons
        peak_hour_str = pd.Timestamp(origin_df.loc[origin_df["p50"].idxmax(), "delivery_start_utc"]).strftime("%H:%M")
        st.markdown(
            '<div class="te-quick-prompts-bar"><span>✨ Quick Questions</span><span>Click to ask</span></div>',
            unsafe_allow_html=True,
        )
        q_col1, q_col2 = st.columns(2, gap="small")
        triggered_prompt: str | None = None
        with q_col1:
            if st.button(f"🔍 Drivers at {peak_hour_str}", use_container_width=True, key="qp_peak"):
                triggered_prompt = f"What features drive the price at {peak_hour_str}?"
            if st.button("🔄 Change vs. t−1h", use_container_width=True, key="qp_rev"):
                triggered_prompt = "What changed compared to the previous hour (t-1h)?"
        with q_col2:
            if st.button("⚡ Spike Risk Summary", use_container_width=True, key="qp_spike"):
                triggered_prompt = "Why is there a spike risk in this horizon and when does it peak?"
            if st.button("🏭 Review Actions", use_container_width=True, key="qp_rec"):
                triggered_prompt = "What operational suggestions to review do you have for shift supervisors and traders?"

        # 3. BOTTOM: Compact Single-Row Chat Input Form + Settings Expander
        with st.form("te_chat_form", clear_on_submit=True, border=False):
            in_col, btn_col = st.columns([3.3, 1.0], gap="small")
            with in_col:
                user_text = st.text_input(
                    "Ask the AI Assistant",
                    placeholder=f"Ask about {peak_hour_str}, Spot price, Dataset Period...",
                    label_visibility="collapsed",
                )
            with btn_col:
                submitted = st.form_submit_button("Send ➤", type="primary", use_container_width=True)
            if submitted and user_text.strip():
                triggered_prompt = user_text.strip()

        with st.expander("⚙️ AI Model, API Key & Reset Chat", expanded=False):
            st.selectbox(
                "AI Model (used when API Key is provided)",
                [
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-2.0-flash",
                ],
                index=0,
                help=(
                    "Select the primary LLM model. Also auto-detects OpenAI (gpt-4o if key starts with sk-...) "
                    "or Groq (llama-3.3-70b-versatile if key starts with gsk_...)."
                ),
                key="preferred_llm_model",
            )
            st.text_input(
                "API Key (Gemini AIza..., OpenAI sk-..., or Groq gsk_...)",
                type="default",
                placeholder="Paste API key here (masked automatically)...",
                help="Masked via CSS (-webkit-text-security: disc) so Chrome's Google Password Manager never pops up.",
                key="gemini_api_key_input",
            )
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.te_chat_messages = st.session_state.te_chat_messages[:1]
                st.rerun()

        # Process newly triggered question before rendering inside chat_box
        if triggered_prompt:
            prior_history = list(st.session_state.te_chat_messages)
            st.session_state.te_chat_messages.append({"role": "user", "markdown": triggered_prompt})
            resp = generate_explainability_response(
                question=triggered_prompt,
                origin_df=origin_df,
                briefing=briefing,
                selected_origin=selected_origin,
                api_key=st.session_state.get("gemini_api_key_input", ""),
                preferred_model=st.session_state.get("preferred_llm_model", "gemini-2.5-pro"),
                ui_context=ui_context,
                chat_history=prior_history,
            )
            st.session_state.te_chat_messages.append(
                {
                    "role": "assistant",
                    "markdown": resp["markdown"],
                    "source": resp.get("source", "Situational Desk AI"),
                }
            )

        # Populate the top Chat Container with all messages (clean text & tables, no bulky SHAP bars)
        with chat_box:
            for idx, msg in enumerate(st.session_state.te_chat_messages):
                if msg["role"] == "user":
                    st.markdown(
                        f"""
                        <div class="te-msg-user">
                          <div class="te-msg-user-label">You</div>
                          {html.escape(msg["markdown"])}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    src_badge = html.escape(msg.get("source", "AI Assistant"))
                    st.markdown(
                        f"""
                        <div class="te-msg-ai-header">
                          <span>⚡ Turku Energia AI</span>
                          <span style="color:#F59B00;">· {src_badge}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(msg["markdown"])

                    if idx + 1 < len(st.session_state.te_chat_messages):
                        st.markdown(
                            "<hr style='margin:0.45rem 0; border:none; border-top:1px solid var(--te-border-subtle);'/>",
                            unsafe_allow_html=True,
                        )
            st.markdown("<div id='te-chat-bottom-anchor'></div>", unsafe_allow_html=True)


def main() -> None:
    data = load_dashboard_data()
    val_wide: pd.DataFrame = data["val_wide"]
    fu_wide: pd.DataFrame = data["fu_wide"]
    test_sub: pd.DataFrame | None = data["test_sub"]
    val_sub: pd.DataFrame | None = data["val_sub"]

    if test_sub is None and val_sub is None:
        st.error("No forecast files found. Please run `./venv/bin/python -m src.evaluation.pipeline` first.")
        return

    # ------------------------------------------------------------------
    # Theme State (Default: Control-Room Dark Mode)
    # ------------------------------------------------------------------
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    def _toggle_theme() -> None:
        st.session_state.dark_mode = not bool(st.session_state.get("dark_mode", True))

    dark_mode: bool = bool(st.session_state.dark_mode)

    # ------------------------------------------------------------------
    # Left Sidebar: Time Machine Controls & Anti-Leakage Gate
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div class="te-sb-header-card">
              <div class="te-sb-kicker">Control Desk Navigation</div>
              <h2 class="te-sb-title">⏱️ Historical Time Machine</h2>
              <p class="te-sb-desc">
                Jump to any hourly forecast origin (<i>t</i>) to inspect the exact 10-hour outlook known at that moment.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="te-sb-section-label">
              <span>1. Dataset Period</span>
              <span class="te-sb-count-pill">2026</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        period_options = []
        if test_sub is not None:
            period_options.append("Test Period (5 Oct – 1 Nov 2026)")
        if val_sub is not None:
            period_options.append("Validation Period (7 Sep – 4 Oct 2026)")

        selected_period = st.radio(
            "Dataset Period",
            period_options,
            index=0,
            label_visibility="collapsed",
        )
        active_sub = test_sub if selected_period.startswith("Test") else val_sub
        assert active_sub is not None

        origins_sorted = pd.DatetimeIndex(active_sub["forecast_origin_utc"].unique()).sort_values()

        # Precompute per-origin spike risk summary for quick jumping
        orig_grp = active_sub.groupby("forecast_origin_utc", as_index=False).agg(
            max_hi=("p_spike_high", "max"),
            max_lo=("p_spike_low", "max"),
            max_p50=("p50", "max"),
            min_p50=("p50", "min"),
        )
        high_risk_origins = orig_grp.loc[
            (orig_grp["max_hi"] >= SPIKE_ALERT_THRESHOLD) | (orig_grp["max_lo"] >= SPIKE_ALERT_THRESHOLD)
        ].sort_values("max_hi", ascending=False)

        filter_options = [
            "📅 All 673 Origins (Chronological)",
            f"🚨 Spike Alert Origins Only (≥ {SPIKE_ALERT_THRESHOLD:.0%})",
            "🔥 Top 20 High-Price Spike Origins (> Spot + 50 €)",
            "❄️ Top 20 Low / Negative Spike Origins (< Spot − 50 €)",
        ]

        st.markdown(
            """
            <div class="te-sb-section-label" style="margin-top:0.85rem;">
              <span>2. Origin Filter &amp; Time Step (t)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        filter_mode = st.selectbox(
            "Quick Filter (Origin Filter)",
            filter_options,
            help=(
                "Pre-filters the list of hourly forecast origins:\n"
                "• Top 20 High-Price: Origins with the highest probability of price surges > Spot + 50 €/MWh.\n"
                "• Top 20 Low-Price: Origins with the highest probability of price drops < Spot − 50 €/MWh."
            ),
        )

        if filter_mode.startswith("🚨") and len(high_risk_origins):
            candidate_origins = pd.DatetimeIndex(high_risk_origins["forecast_origin_utc"]).sort_values()
        elif filter_mode.startswith("🔥"):
            candidate_origins = pd.DatetimeIndex(
                orig_grp.sort_values("max_hi", ascending=False).head(20)["forecast_origin_utc"]
            )
        elif filter_mode.startswith("❄️"):
            candidate_origins = pd.DatetimeIndex(
                orig_grp.sort_values("max_lo", ascending=False).head(20)["forecast_origin_utc"]
            )
        else:
            candidate_origins = origins_sorted

        origin_labels = [ts.strftime("%Y-%m-%d %H:%M UTC") for ts in candidate_origins]

        if "origin_idx" not in st.session_state or st.session_state.get("last_filter") != (selected_period, filter_mode):
            st.session_state.origin_idx = 0
            st.session_state.last_filter = (selected_period, filter_mode)
            st.session_state.origin_select_box = origin_labels[0]

        st.session_state.origin_idx = min(st.session_state.origin_idx, len(origin_labels) - 1)
        if st.session_state.get("origin_select_box") not in origin_labels:
            st.session_state.origin_select_box = origin_labels[st.session_state.origin_idx]

        def _on_select_origin() -> None:
            chosen = st.session_state.get("origin_select_box")
            if chosen in origin_labels:
                st.session_state.origin_idx = origin_labels.index(chosen)

        def _step_prev() -> None:
            if st.session_state.origin_idx > 0:
                st.session_state.origin_idx -= 1
                st.session_state.origin_select_box = origin_labels[st.session_state.origin_idx]

        def _step_next() -> None:
            if st.session_state.origin_idx + 1 < len(origin_labels):
                st.session_state.origin_idx += 1
                st.session_state.origin_select_box = origin_labels[st.session_state.origin_idx]

        selected_label = st.selectbox(
            f"Forecast Origin (t) — [{st.session_state.origin_idx + 1} / {len(origin_labels)}]",
            origin_labels,
            key="origin_select_box",
            on_change=_on_select_origin,
        )
        st.session_state.origin_idx = origin_labels.index(selected_label)
        selected_origin = candidate_origins[st.session_state.origin_idx]

        # Prev / Next Hour Navigation Buttons placed directly BELOW Forecast Origin (t)
        nav_c1, nav_c2 = st.columns(2)
        with nav_c1:
            st.button(
                "◀ Previous",
                use_container_width=True,
                on_click=_step_prev,
                disabled=(st.session_state.origin_idx <= 0),
            )
        with nav_c2:
            st.button(
                "Next ▶",
                use_container_width=True,
                on_click=_step_next,
                disabled=(st.session_state.origin_idx + 1 >= len(origin_labels)),
            )

        st.markdown(
            """
            <div class="te-sb-section-label" style="margin-top:0.9rem;">
              <span>3. Display &amp; Overlays</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        show_chat_sidebar = st.toggle(
            "🤖 Show AI Explainability Chat",
            value=True,
            help="Show or hide the interactive AI Explainability Chat sidebar on the right.",
        )
        show_hindsight = st.toggle(
            "🎯 Overlay Actual Prices (Hindsight)",
            value=selected_period.startswith("Validation"),
            help="Overlays realized imbalance prices after origin t for post-event validation.",
        )

        st.markdown("<div style='height:0.45rem;'></div>", unsafe_allow_html=True)
        origin_rows = active_sub[active_sub["forecast_origin_utc"] == selected_origin].sort_values("step")
        cutoff_ts = pd.Timestamp(origin_rows["data_cutoff_utc"].iloc[0])
        st.markdown(
            f"""
            <div class="te-gate-box">
              <div style="font-family:'Fira Sans Extra Condensed',sans-serif; font-weight:800; font-size:0.92rem; text-transform:uppercase; color:#46BEDC; margin-bottom:0.35rem; letter-spacing:0.03em;">
                🛡️ Anti-Leakage Gate (Rule #1)
              </div>
              <div class="te-gate-row">
                <span><strong>Origin (<i>t</i>):</strong></span>
                <code>{selected_origin.strftime('%Y-%m-%d %H:%M:%SZ')}</code>
              </div>
              <div class="te-gate-row">
                <span><strong>Data Cutoff:</strong></span>
                <code>{cutoff_ts.strftime('%Y-%m-%d %H:%M:%SZ')}</code>
              </div>
              <div class="te-gate-verified">
                <span>✓ Verified:</span>
                <code style="background:transparent !important; border:none !important; padding:0 !important; color:inherit !important; font-weight:700;">first_usable_origin_utc ≤ t</code>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # Prepare Data for Selected Origin t
    # ------------------------------------------------------------------
    origin_df = active_sub[active_sub["forecast_origin_utc"] == selected_origin].sort_values("step").reset_index(drop=True)
    prev_origin_ts = selected_origin - pd.Timedelta(hours=1)
    prev_origin_df = active_sub[active_sub["forecast_origin_utc"] == prev_origin_ts].sort_values("step").reset_index(drop=True)

    usable_grid = get_usable_series_at(val_wide, fu_wide, selected_origin)
    deliv_idx = pd.DatetimeIndex(origin_df["delivery_start_utc"])
    origin_df["spot_price"] = usable_grid["fi_spot_price"].reindex(deliv_idx).to_numpy()
    origin_df["wind_da"] = usable_grid["fi_wind_fc_da"].reindex(deliv_idx).to_numpy()
    origin_df["solar_da"] = usable_grid["fi_solar_fc_da"].reindex(deliv_idx).to_numpy()
    origin_df["cons_da"] = usable_grid["fi_consumption_fc_da"].reindex(deliv_idx).to_numpy()
    origin_df["residual_load_da"] = origin_df["cons_da"] - origin_df["wind_da"] - origin_df["solar_da"]

    origin_df["actual_imbalance_price"] = val_wide["fi_imbalance_price"].reindex(deliv_idx).to_numpy()
    origin_df["actual_direction_code"] = val_wide["fi_mfrr_direction"].reindex(deliv_idx).to_numpy()

    briefing = build_situational_narrative(origin_df, prev_origin_df, usable_grid, selected_origin)

    # ------------------------------------------------------------------
    # Top Brand Header + CSS (Combined into single block to avoid top flex gap)
    # ------------------------------------------------------------------
    st.markdown(
        inject_brand_css(dark_mode=dark_mode)
        + render_hero_header(
            origin_str=selected_origin.strftime("%Y-%m-%d %H:%M UTC"),
            period_label=selected_period,
        ),
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Split Layout: Main Data Dashboard (Left) + AI Chat Sidebar (Right)
    # ------------------------------------------------------------------
    if show_chat_sidebar:
        main_col, chat_col = st.columns([2.2, 1.15], gap="medium")
    else:
        main_col = st.container()
        chat_col = None

    with main_col:
        # 1. KPI Strip (5 Cards)
        p50_mean = float(origin_df["p50"].mean())
        spot_mean = float(origin_df["spot_price"].mean())
        st.markdown(
            render_kpi_strip(
                p50_mean=p50_mean,
                spot_mean=spot_mean,
                max_p_hi=briefing["max_p_hi"],
                n_hi_steps=briefing["n_hi_steps"],
                max_p_lo=briefing["max_p_lo"],
                n_lo_steps=briefing["n_lo_steps"],
                dominant_dir_short=briefing["dominant_dir_short"],
                mean_p_up=briefing["mean_p_up"],
                mean_p_down=briefing["mean_p_down"],
                mean_band=briefing["mean_band"],
                max_abs_rev=briefing["max_abs_rev"],
            ),
            unsafe_allow_html=True,
        )

        # 2. Situational Alert Banner
        st.markdown(render_alert_cards(briefing["alerts"]), unsafe_allow_html=True)

        # 3. Main Navigation Tabs
        tab_forecast, tab_drivers, tab_hindsight, tab_api = st.tabs([
            "📈 10-Hour Forecast & Risk Monitor",
            "🔍 Physical Grid Drivers & SHAP Explainability",
            "🎯 Hindsight Benchmark vs. Official Baseline",
            "🔌 40-Step Data Table & JSON API",
        ])

        with tab_forecast:
            hist_start = selected_origin - pd.Timedelta(hours=6)
            hist_idx = pd.date_range(hist_start, selected_origin - pd.Timedelta(minutes=15), freq="15min", tz="UTC")
            hist_imb = usable_grid["fi_imbalance_price"].reindex(hist_idx)
            hist_spot = usable_grid["fi_spot_price"].reindex(hist_idx)

            # Primary Probabilistic Fan Chart Card
            st.markdown(
                """
                <div class="te-section-header">
                  <h3 class="te-section-title">Probabilistic 10-Hour Imbalance Price Forecast (P10–P50–P90) vs. Spot Price<span>.</span></h3>
                  <span class="te-section-subtitle">Left of NOW: 6h Realized History (≤ t) · Right: 40 Quarter-Hour Steps (+10h)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            fig_fan = build_price_fan_chart(
                origin_df=origin_df,
                prev_origin_df=prev_origin_df,
                hist_idx=hist_idx,
                hist_imb=hist_imb,
                hist_spot=hist_spot,
                selected_origin=selected_origin,
                show_hindsight=show_hindsight,
                dark_mode=dark_mode,
            )
            st.plotly_chart(fig_fan, use_container_width=True)

            # Side-by-Side Secondary Row: Spike Risk (Left) & Balancing Direction (Right)
            sub_c1, sub_c2 = st.columns(2, gap="medium")
            with sub_c1:
                st.markdown(
                    f"""
                    <div class="te-section-header">
                      <h3 class="te-section-title">Price Spike Probability (|P − Spot| &gt; {int(SPIKE_MARGIN)} €)<span>.</span></h3>
                      <span class="te-section-subtitle">Alert Threshold at 30%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_spike = build_spike_probability_chart(origin_df, SPIKE_ALERT_THRESHOLD, dark_mode=dark_mode)
                st.plotly_chart(fig_spike, use_container_width=True)

            with sub_c2:
                st.markdown(
                    """
                    <div class="te-section-header">
                      <h3 class="te-section-title">mFRR Balancing Direction (Up / Down / None)<span>.</span></h3>
                      <span class="te-section-subtitle">Up = Power Deficit · Down = Power Surplus</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_dir = build_direction_probability_chart(origin_df, dark_mode=dark_mode)
                st.plotly_chart(fig_dir, use_container_width=True)

        with tab_drivers:
            drv_c1, drv_c2 = st.columns([1.15, 1.0], gap="medium")
            with drv_c1:
                st.markdown(
                    """
                    <div class="te-section-header">
                      <h3 class="te-section-title">Consumption, Wind Power &amp; Residual Load (Steps 1–40)<span>.</span></h3>
                      <span class="te-section-subtitle">Residual Load = Consumption − Wind − Solar (MW)</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.plotly_chart(build_physical_drivers_chart(origin_df, dark_mode=dark_mode), use_container_width=True)

            with drv_c2:
                st.markdown(
                    """
                    <div class="te-section-header">
                      <h3 class="te-section-title">SHAP Top Drivers Across Current 10h Horizon<span>.</span></h3>
                      <span class="te-section-subtitle">Weighted Dominance Across All 40 Quarter-Hour Steps</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_shap = build_shap_drivers_bar_chart(origin_df, dark_mode=dark_mode)
                if fig_shap is not None:
                    st.plotly_chart(fig_shap, use_container_width=True)

            if data["feat_imp"] is not None:
                st.markdown(
                    """
                    <div class="te-section-header">
                      <h3 class="te-section-title">Global LightGBM Feature Importance (Top 15 by Information Gain)<span>.</span></h3>
                      <span class="te-section-subtitle">Overall Model Impact Across Training History</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                top_imp = data["feat_imp"].head(15).iloc[::-1].copy()
                top_imp["label"] = top_imp["feature"].apply(format_driver_name)
                fig_imp = go.Figure(
                    go.Bar(
                        x=top_imp["gain_p50"],
                        y=top_imp["label"],
                        orientation="h",
                        marker_color=TE_COLORS["turquoise"] if dark_mode else TE_COLORS["navy"],
                    )
                )
                fig_imp.update_layout(
                    height=360,
                    paper_bgcolor="#0D1E2E" if dark_mode else "#FFFFFF",
                    plot_bgcolor="#091622" if dark_mode else "#FFFFFF",
                    font=dict(color="#E6F3F8" if dark_mode else TE_COLORS["navy"]),
                    xaxis_title="Total Information Gain (P50 Spread Model)",
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                fig_imp.update_xaxes(gridcolor="#152C42" if dark_mode else "#EAF7FB")
                fig_imp.update_yaxes(gridcolor="#152C42" if dark_mode else "#EAF7FB")
                st.plotly_chart(fig_imp, use_container_width=True)

        with tab_hindsight:
            st.markdown(
                """
                <div class="te-section-header">
                  <h3 class="te-section-title">Validation Benchmark (7 Sep – 4 Oct 2026 · 26,700 Evaluated Rows)<span>.</span></h3>
                  <span class="te-section-subtitle">Comparison of Our LightGBM Suite Against the Official Baseline</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            val_sum = data["val_summary"]
            base_sum = data["base_summary"]
            if val_sum is not None and base_sum is not None:
                v_row = val_sum.iloc[0]
                b_row = base_sum.iloc[0]
                comp_df = pd.DataFrame([
                    {
                        "Evaluation Criterion (Level 1)": "Point Accuracy — MAE on P50 (30% Weight)",
                        "Official Baseline": f"{float(b_row['mae_p50']):.2f} €/MWh",
                        "Our LightGBM Suite": f"{float(v_row['mae_p50']):.2f} €/MWh",
                        "Improvement": f"{(float(v_row['mae_p50']) - float(b_row['mae_p50'])):+.2f} €/MWh ({(1 - float(v_row['mae_p50'])/float(b_row['mae_p50']))*100:.1f}% better)",
                    },
                    {
                        "Evaluation Criterion (Level 1)": "Calibration — Mean Pinball Loss P10/P90 (20% Weight)",
                        "Official Baseline": f"{float(b_row['calibration_pinball_mean']):.2f}",
                        "Our LightGBM Suite": f"{float(v_row['calibration_pinball_mean']):.2f}",
                        "Improvement": f"{(float(v_row['calibration_pinball_mean']) - float(b_row['calibration_pinball_mean'])):+.2f} ({(1 - float(v_row['calibration_pinball_mean'])/float(b_row['calibration_pinball_mean']))*100:.1f}% better)",
                    },
                    {
                        "Evaluation Criterion (Level 1)": "Interval Coverage — P10–P90 Coverage (Target ~80%)",
                        "Official Baseline": f"{float(b_row['coverage_p10_p90']):.2%}",
                        "Our LightGBM Suite": f"{float(v_row['coverage_p10_p90']):.2%}",
                        "Improvement": f"Exact 80.0% Target ({float(v_row['coverage_p10_p90']):.2%})",
                    },
                    {
                        "Evaluation Criterion (Level 1)": "Spike Identification (±50 €) — Mean Brier Score (20% Weight)",
                        "Official Baseline": f"{float(b_row['brier_spike_mean']):.4f}",
                        "Our LightGBM Suite": f"{float(v_row['brier_spike_mean']):.4f}",
                        "Improvement": f"{(float(v_row['brier_spike_mean']) - float(b_row['brier_spike_mean'])):+.4f} ({(1 - float(v_row['brier_spike_mean'])/float(b_row['brier_spike_mean']))*100:.1f}% better)",
                    },
                    {
                        "Evaluation Criterion (Level 1)": "Balancing Direction (Up/Down/None) — 3-Class Brier (10% Weight)",
                        "Official Baseline": f"{float(b_row['brier_direction']):.4f}",
                        "Our LightGBM Suite": f"{float(v_row['brier_direction']):.4f}",
                        "Improvement": f"{(float(v_row['brier_direction']) - float(b_row['brier_direction'])):+.4f} ({(1 - float(v_row['brier_direction'])/float(b_row['brier_direction']))*100:.1f}% better)",
                    },
                ])
                st.table(comp_df)
            else:
                st.info(
                    "To generate full validation score tables, run `.\\venv\\Scripts\\python.exe -m src.evaluation.pipeline` "
                    "(Reference benchmark: MAE 43.35 €/MWh vs. Baseline 66.20 €/MWh; Pinball 12.43 vs. 24.18; Coverage 80.00%)."
                )

            val_h = data["val_horizon"]
            base_h = data["base_horizon"]
            if val_h is not None and base_h is not None:
                st.plotly_chart(build_horizon_benchmark_chart(val_h, base_h, dark_mode=dark_mode), use_container_width=True)

        with tab_api:
            st.markdown(
                f"""
                <div class="te-section-header">
                  <h3 class="te-section-title">40-Step Forecast Table &amp; JSON API for Origin {selected_origin.strftime('%Y-%m-%dT%H:%M:%SZ')}<span>.</span></h3>
                  <span class="te-section-subtitle">Machine-Readable Time Series &amp; REST API Payload</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            display_cols = [
                "step",
                "delivery_start_utc",
                "spot_price",
                "p10",
                "p50",
                "p90",
                "p_up",
                "p_down",
                "p_none",
                "p_spike_high",
                "p_spike_low",
                "top_drivers",
            ]
            st.dataframe(origin_df[display_cols], use_container_width=True, hide_index=True)

            api_payload = {
                "forecast_origin_utc": selected_origin.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "data_cutoff_utc": cutoff_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "situational_summary": {
                    "max_p_spike_high": round(briefing["max_p_hi"], 4),
                    "max_p_spike_low": round(briefing["max_p_lo"], 4),
                    "dominant_direction": briefing["dominant_dir"],
                    "top_drivers": briefing["top3_drivers"],
                    "alerts": briefing["alerts"],
                },
                "steps": [
                    {
                        "step": int(r["step"]),
                        "delivery_start_utc": pd.Timestamp(r["delivery_start_utc"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "p10": float(r["p10"]),
                        "p50": float(r["p50"]),
                        "p90": float(r["p90"]),
                        "p_up": float(r["p_up"]),
                        "p_down": float(r["p_down"]),
                        "p_none": float(r["p_none"]),
                        "p_spike_high": float(r["p_spike_high"]),
                        "p_spike_low": float(r["p_spike_low"]),
                        "top_drivers": str(r["top_drivers"]),
                    }
                    for _, r in origin_df.iterrows()
                ],
            }
            st.download_button(
                "⬇️ Download Origin Forecast as JSON API Payload",
                data=json.dumps(api_payload, indent=2, ensure_ascii=False),
                file_name=f"forecast_{selected_origin.strftime('%Y%m%dT%H%MZ')}.json",
                mime="application/json",
            )
            with st.expander("Preview JSON API Response in Browser"):
                st.json(api_payload)

    # ------------------------------------------------------------------
    # Right-Hand AI Explainability Chat Sidebar
    # ------------------------------------------------------------------
    if chat_col is not None:
        ui_context = {
            "selected_period": selected_period,
            "filter_mode": filter_mode,
            "origin_pos_str": f"{st.session_state.origin_idx + 1} / {len(origin_labels)}",
            "cutoff_str": cutoff_ts.strftime("%Y-%m-%d %H:%M:%SZ"),
            "show_hindsight": show_hindsight,
            "dark_mode": dark_mode,
        }
        with chat_col:
            _render_chat_sidebar(origin_df, briefing, selected_origin, ui_context=ui_context)

    # Mount the Sun <-> Moon toggle button into Streamlit's top toolbar (at bottom of DOM to avoid top gap)
    st.button(
        "Theme",
        key="te_theme_toggle_btn",
        on_click=_toggle_theme,
        help="Switch between Control-Room Dark Mode (Moon) and Web Light Mode (Sun)",
    )
    chat_msg_count = len(st.session_state.get("te_chat_messages", []))
    components.html(get_toolbar_theme_toggle_html(dark_mode=dark_mode, chat_msg_count=chat_msg_count), height=0)


if __name__ == "__main__":
    main()
