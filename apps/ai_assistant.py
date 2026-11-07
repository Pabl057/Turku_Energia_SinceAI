"""Integrated AI Explainability Chat Module (Gemini API + Data-Grounded Analytical Engine).

Allows shift supervisors and electricity traders to ask natural-language questions
about the currently displayed 10-hour forecast (e.g., "What features drive the price
at 14:00?", "Why is spike risk elevated?", "What changed vs. t-1h?").

Supports:
1. Live Google Gemini API (`GEMINI_API_KEY` / `GOOGLE_API_KEY`) grounded with the exact
   40-step forecast, physical grid series, and TreeSHAP attributions at origin `t`.
2. Deterministic Data-Grounded Explainability Engine (instant structured Markdown tables
   and SHAP feature weightings even without an external API key).
"""

from __future__ import annotations

import os
import re
import pandas as pd
import requests

from apps.theme import format_driver_name


def compute_horizon_driver_weights(origin_df: pd.DataFrame, target_rows: pd.DataFrame | None = None) -> list[dict]:
    """Compute normalized percentage weights of TreeSHAP top drivers for the given rows."""
    df_use = target_rows if (target_rows is not None and len(target_rows)) else origin_df
    scores: dict[str, float] = {}
    for s in df_use["top_drivers"].dropna().astype(str):
        parts = [p.strip() for p in s.split(";") if p.strip()]
        for idx, raw_drv in enumerate(parts[:3]):
            # Rank 1 gets weight 3, Rank 2 gets 2, Rank 3 gets 1
            w = float(3 - idx)
            label = format_driver_name(raw_drv)
            scores[label] = scores.get(label, 0.0) + w

    total = sum(scores.values()) or 1.0
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:5]
    return [
        {
            "driver": name,
            "pct": round((val / total) * 100.0, 1),
        }
        for name, val in ranked
    ]


def _extract_requested_hour_rows(question: str, origin_df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """Check if the user asked about a specific hour (e.g. 'at 14:00', '14:30', 'Step 8')."""
    q_low = question.lower()

    # Check for explicit step number: "step 12" or "schritt 12"
    m_step = re.search(r"\b(?:step|schritt)\s*(\d{1,2})\b", q_low)
    if m_step:
        step_num = int(m_step.group(1))
        sub = origin_df[origin_df["step"] == step_num]
        if len(sub):
            ts_str = sub["delivery_start_utc"].iloc[0].strftime("%H:%M UTC")
            return sub, f"Step {step_num} ({ts_str})"

    # Check for time pattern like "14:15", "at 14", "14 utc", "14h", "14 uhr"
    m_time = re.search(r"\b([01]?\d|2[0-3])(?::([0-5]\d))?\s*(?:utc|uhr|h|o'clock)?\b", q_low)
    if m_time and (
        "at " in q_low
        or "around " in q_low
        or "utc" in q_low
        or "uhr" in q_low
        or ":" in m_time.group(0)
        or "um " in q_low
    ):
        hour = int(m_time.group(1))
        minute = int(m_time.group(2)) if m_time.group(2) is not None else None
        deliv_dt = pd.to_datetime(origin_df["delivery_start_utc"], utc=True)
        if minute is not None:
            sub = origin_df[(deliv_dt.dt.hour == hour) & (deliv_dt.dt.minute == minute)]
            label = f"{hour:02d}:{minute:02d} UTC"
        else:
            sub = origin_df[deliv_dt.dt.hour == hour]
            label = f"{hour:02d}:00–{hour:02d}:45 UTC"
        if len(sub):
            return sub, label

    return None, None


def _build_grounded_context_prompt(
    question: str,
    origin_df: pd.DataFrame,
    briefing: dict,
    selected_origin: pd.Timestamp,
) -> str:
    """Create a compact, leak-free English data context prompt for the Gemini LLM."""
    peak_hi_row = origin_df.loc[origin_df["p_spike_high"].idxmax()]
    peak_lo_row = origin_df.loc[origin_df["p_spike_low"].idxmax()]
    peak_p50_row = origin_df.loc[origin_df["p50"].idxmax()]
    min_p50_row = origin_df.loc[origin_df["p50"].idxmin()]

    sample_steps = origin_df[origin_df["step"].isin([1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40])].copy()
    rows_txt = []
    for _, r in sample_steps.iterrows():
        ts = pd.Timestamp(r["delivery_start_utc"]).strftime("%H:%M")
        rows_txt.append(
            f"Step {int(r['step'])} ({ts} UTC): Spot={r['spot_price']:.1f}, "
            f"P10={r['p10']:.1f}, P50={r['p50']:.1f}, P90={r['p90']:.1f}, "
            f"P_Up={r['p_up']:.2f}, P_Down={r['p_down']:.2f}, "
            f"SpikeHi={r['p_spike_high']:.2f}, SpikeLo={r['p_spike_low']:.2f}, "
            f"WindDA={r['wind_da']:.0f}MW, ConsDA={r['cons_da']:.0f}MW, "
            f"Drivers={r['top_drivers']}"
        )

    return f"""You are the AI Explainability Assistant inside Turku Energia's Situational Control Desk.
Answer the user's question in clear, concise, and authoritative English following Turku Energia's Tone of Voice:
"Asiantunteva, mutta ei insinöörimäinen" (Expert and trustworthy, yet clear and solution-oriented).
Use structured Markdown, bullet points, and a compact Markdown table for relevant quarter-hour steps.
IMPORTANT: Always frame operational actions as "Suggestions to Review" (decision support based on public market data), never as automated plant dispatch commands.

Current Forecast Origin (t): {selected_origin.strftime('%Y-%m-%d %H:%M UTC')}
Dominant Balancing Direction: {briefing['dominant_dir']} (Avg P_Up={briefing['mean_p_up']:.1%}, Avg P_Down={briefing['mean_p_down']:.1%})
Peak High-Spike Risk: {briefing['max_p_hi']:.1%} at {pd.Timestamp(peak_hi_row['delivery_start_utc']).strftime('%H:%M UTC')} (P50={peak_hi_row['p50']:.1f} €/MWh vs Spot={peak_hi_row['spot_price']:.1f} €/MWh)
Peak Low-Spike Risk: {briefing['max_p_lo']:.1%} at {pd.Timestamp(peak_lo_row['delivery_start_utc']).strftime('%H:%M UTC')} (P50={peak_lo_row['p50']:.1f} €/MWh)
Highest P50 Forecast: {peak_p50_row['p50']:.1f} €/MWh at {pd.Timestamp(peak_p50_row['delivery_start_utc']).strftime('%H:%M UTC')}
Lowest P50 Forecast: {min_p50_row['p50']:.1f} €/MWh at {pd.Timestamp(min_p50_row['delivery_start_utc']).strftime('%H:%M UTC')}
Max Revision vs. Previous Hour (t-1h): {briefing['max_abs_rev']:.1f} €/MWh (Mean {briefing['mean_rev']:+.1f} €/MWh)

Sample of the 40 Quarter-Hour Steps:
{chr(10).join(rows_txt)}

User Question: {question}
"""


def generate_explainability_response(
    question: str,
    origin_df: pd.DataFrame,
    briefing: dict,
    selected_origin: pd.Timestamp,
    api_key: str | None = None,
) -> dict:
    """Return structured English AI response with Markdown text, table, and feature weights."""
    matched_rows, matched_label = _extract_requested_hour_rows(question, origin_df)
    driver_weights = compute_horizon_driver_weights(origin_df, matched_rows)

    effective_key = (api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()

    # 1. Try live Gemini API if key is available
    if effective_key:
        try:
            prompt = _build_grounded_context_prompt(question, origin_df, briefing, selected_origin)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={effective_key}"
            resp = requests.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 700},
                },
                timeout=12,
            )
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    text = candidates[0]["content"]["parts"][0]["text"]
                    return {
                        "markdown": text,
                        "driver_weights": driver_weights,
                        "source": "Gemini 2.0 Flash · Grounded Data",
                    }
        except Exception:
            pass  # Fall back seamlessly to local grounded engine

    # 2. Local Data-Grounded Explainability Engine (always works, zero latency)
    q_low = question.lower()

    if matched_rows is not None and matched_label is not None:
        # User asked about a specific time / hour / step!
        table_lines = [
            "| Time (UTC) | Step | Spot (€) | P50 (€) | P10–P90 Band | P(Spike Hi/Lo) | Key SHAP Drivers |",
            "| :--- | :---: | ---: | ---: | :---: | :---: | :--- |",
        ]
        for _, r in matched_rows.iterrows():
            t_str = pd.Timestamp(r["delivery_start_utc"]).strftime("%H:%M")
            raw_drvs = [format_driver_name(d) for d in str(r["top_drivers"]).split(";")[:2] if d.strip()]
            drv_short = ", ".join(raw_drvs)
            table_lines.append(
                f"| **{t_str}** | {int(r['step'])} | {r['spot_price']:.1f} | **{r['p50']:.1f}** | "
                f"[{r['p10']:.1f}, {r['p90']:.1f}] | {r['p_spike_high']:.0%} / {r['p_spike_low']:.0%} | {drv_short} |"
            )

        mean_p50 = float(matched_rows["p50"].mean())
        mean_spot = float(matched_rows["spot_price"].mean())
        mean_wind = float(matched_rows["wind_da"].mean())
        mean_cons = float(matched_rows["cons_da"].mean())
        mean_res = float(matched_rows["residual_load_da"].mean())
        top_drv_names = ", ".join(d["driver"] for d in driver_weights[:3])

        md = (
            f"### Breakdown for **{matched_label}** (Origin `{selected_origin.strftime('%H:%M UTC')}`)\n\n"
            f"Across the requested window, the mean **P50 forecast is {mean_p50:.2f} €/MWh** "
            f"(Spread vs. Day-Ahead Spot: **{mean_p50 - mean_spot:+.2f} €/MWh**).\n\n"
            f"* **Primary SHAP Drivers:** {top_drv_names}.\n"
            f"* **Physical Grid Fundamentals:** Scheduled consumption **{mean_cons:,.0f} MW**, "
            f"wind generation forecast **{mean_wind:,.0f} MW**, resulting residual load **{mean_res:,.0f} MW**.\n\n"
            + "\n".join(table_lines)
        )
        return {
            "markdown": md,
            "driver_weights": driver_weights,
            "source": "TreeSHAP & FeatureStore Engine",
        }

    if any(k in q_low for k in ("previous", "revision", "changed", "change", "drift", "t-1", "t−1", "vorstunde")):
        md = (
            f"### Forecast Revision vs. Previous Hour (`t − 1h`)\n\n"
            f"Compared to the previous hourly run, the **P50 trajectory** shifted by an average of "
            f"**{briefing['mean_rev']:+.2f} €/MWh** (maximum single-step revision: "
            f"**Δ {briefing['max_abs_rev']:.2f} €/MWh**).\n\n"
            f"* **Root Cause of Update:** Newly published real-time telemetry prior to `{selected_origin.strftime('%H:%M UTC')}` "
            f"(30-minute publication lag for realized load & cross-border flows, plus 60-minute lag for aFRR/mFRR balancing activations).\n"
            f"* **Assessment:** {'🔄 Significant revision (≥ 15 €/MWh) — review open intraday exposure.' if briefing['max_abs_rev'] >= 15.0 else '✅ Minor revision (< 15 €/MWh) — dispatch outlook remains stable.'}"
        )
        return {
            "markdown": md,
            "driver_weights": driver_weights,
            "source": "TreeSHAP & FeatureStore Engine",
        }

    if any(k in q_low for k in ("action", "recommend", "suggestion", "review", "supervisor", "trader", "dispatch", "empfehlung")):
        hi_steps = origin_df[origin_df["p_spike_high"] >= 0.30]
        lo_steps = origin_df[origin_df["p_spike_low"] >= 0.30]
        if len(hi_steps):
            t0 = hi_steps["delivery_start_utc"].min().strftime("%H:%M UTC")
            t1 = (hi_steps["delivery_start_utc"].max() + pd.Timedelta(minutes=15)).strftime("%H:%M UTC")
            rec = (
                f"1. **Review High-Price Window ({t0}–{t1}):** Probability of prices `> Spot + 50 €/MWh` peaks at **{briefing['max_p_hi']:.1%}**.\n"
                f"2. **Generation & District Heating:** Review availability of flexible generation assets and evaluate reducing electric boiler load during peak risk steps.\n"
                f"3. **Intraday Trading Desk:** Consider hedging open short positions ahead of the high-risk window."
            )
        elif len(lo_steps):
            t0 = lo_steps["delivery_start_utc"].min().strftime("%H:%M UTC")
            t1 = (lo_steps["delivery_start_utc"].max() + pd.Timedelta(minutes=15)).strftime("%H:%M UTC")
            rec = (
                f"1. **Review Low / Negative Price Window ({t0}–{t1}):** Probability of prices `< Spot − 50 €/MWh` reaches **{briefing['max_p_lo']:.1%}**.\n"
                f"2. **Flexible Demand (Power-to-Heat):** Evaluate ramping up electric boilers or district heat pumps to capture low or negative imbalance prices.\n"
                f"3. **mFRR Down-Regulation Bids:** Verify curtailment readiness."
            )
        else:
            rec = (
                "1. **Standard Operations:** No quarter-hour step exceeds the 30% spike alert threshold.\n"
                "2. **Grid Monitoring:** Dominant balancing tendency is **" + briefing["dominant_dir"] + "** — maintain baseline schedule."
            )
        md = (
            f"### Operational Suggestions to Review (Shift Supervisors & Traders)\n\n"
            f"*(Note per Challenge Guidelines: Decision support grounded in public market data — not automated plant switching commands)*\n\n"
            f"{rec}"
        )
        return {
            "markdown": md,
            "driver_weights": driver_weights,
            "source": "Turku Energia Decision Support",
        }

    # Default: Comprehensive driver & peak analysis across the 10-hour horizon
    peak_idx = origin_df["p50"].idxmax()
    min_idx = origin_df["p50"].idxmin()
    hi_idx = origin_df["p_spike_high"].idxmax()
    r_peak = origin_df.loc[peak_idx]
    r_min = origin_df.loc[min_idx]
    r_hi = origin_df.loc[hi_idx]

    top_drv_names = ", ".join(d["driver"] for d in driver_weights[:3])
    md = (
        f"### Driver & Risk Summary for Origin `{selected_origin.strftime('%Y-%m-%d %H:%M UTC')}`\n\n"
        f"Across the upcoming 10-hour horizon (Steps 1–40), the trajectory is primarily driven by **{top_drv_names}**:\n\n"
        f"| Key Point / Metric | Time (UTC) | Step | Spot (€) | P50 (€) | P10–P90 (€) | Spike Risk |\n"
        f"| :--- | :---: | :---: | ---: | ---: | :---: | :---: |\n"
        f"| **Highest Price (P50 Max)** | {pd.Timestamp(r_peak['delivery_start_utc']).strftime('%H:%M')} | {int(r_peak['step'])} | {r_peak['spot_price']:.1f} | **{r_peak['p50']:.1f}** | [{r_peak['p10']:.1f}, {r_peak['p90']:.1f}] | Hi: {r_peak['p_spike_high']:.0%} |\n"
        f"| **Lowest Price (P50 Min)** | {pd.Timestamp(r_min['delivery_start_utc']).strftime('%H:%M')} | {int(r_min['step'])} | {r_min['spot_price']:.1f} | **{r_min['p50']:.1f}** | [{r_min['p10']:.1f}, {r_min['p90']:.1f}] | Lo: {r_min['p_spike_low']:.0%} |\n"
        f"| **Peak Spike Risk Step** | {pd.Timestamp(r_hi['delivery_start_utc']).strftime('%H:%M')} | {int(r_hi['step'])} | {r_hi['spot_price']:.1f} | **{r_hi['p50']:.1f}** | [{r_hi['p10']:.1f}, {r_hi['p90']:.1f}] | **Hi: {r_hi['p_spike_high']:.1%}** |\n\n"
        f"💡 *Tip: Ask me about a specific delivery hour (e.g., “What features drive the price at {pd.Timestamp(r_peak['delivery_start_utc']).strftime('%H:%M')}?”).* "
    )
    return {
        "markdown": md,
        "driver_weights": driver_weights,
        "source": "TreeSHAP & FeatureStore Engine",
    }
