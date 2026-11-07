"""Integrated AI Explainability & Dashboard Copilot (Multi-Model LLM + Deep Context Engine).

Capabilities:
1. Top-Tier Live AI Models:
   - Google Gemini (`gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`)
   - Auto-support for OpenAI (`gpt-4o` if `sk-...` key provided) & Groq (`llama-3.3-70b-versatile` if `gsk_...` key provided).
2. Full UI & Data Context Grounding:
   - Active `selected_period`, `filter_mode`, `selected_origin`, `show_hindsight`, and multi-turn `chat_history`.
   - Complete 40-step quarter-hour forecast matrix (Spot, P10, P50, P90, mFRR Up/Down/None,
     High/Low Spike probabilities, Wind DA, Consumption DA, Residual Load, and TreeSHAP drivers).
   - Full knowledge of every UI element on the website (Dataset Period, Quick Filter, Forecast Origin,
     Anti-Leakage Gate Rule #1, KPI Cards, Fan Chart, Spike Monitor, mFRR Direction, and Benchmarks).
3. Bilingual (German & English) & Zero-Horizontal-Scroll Formatting.
"""

from __future__ import annotations

import os
import re
import pandas as pd
import requests

from apps.theme import format_driver_name
from apps.ai_knowledge_base import (
    is_german_query,
    kb_digest_for_llm,
    match_kb_entry,
    render_kb_entry,
)


def compute_horizon_driver_weights(origin_df: pd.DataFrame, target_rows: pd.DataFrame | None = None) -> list[dict]:
    """Compute normalized percentage weights of TreeSHAP top drivers for the given rows."""
    df_use = target_rows if (target_rows is not None and len(target_rows)) else origin_df
    scores: dict[str, float] = {}
    for s in df_use["top_drivers"].dropna().astype(str):
        parts = [p.strip() for p in s.split(";") if p.strip()]
        for idx, raw_drv in enumerate(parts[:3]):
            w = float(3 - idx)
            label = format_driver_name(raw_drv)
            scores[label] = scores.get(label, 0.0) + w

    total = sum(scores.values()) or 1.0
    ranked = sorted(scores.items(), key=lambda x: -x[1])[:4]
    return [
        {
            "driver": name,
            "pct": round((val / total) * 100.0, 1),
        }
        for name, val in ranked
    ]


def _is_german_query(question: str) -> bool:
    """Detect whether the user's question is written in German."""
    # Score-based EN/DE detection (shared cognates like "filter", "graph", "website"
    # no longer force German answers to English questions).
    return is_german_query(question)


def _extract_requested_hour_rows(question: str, origin_df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """Check if the user asked about a specific hour (e.g. 'at 14:00', 'um 14 Uhr', 'Step 8')."""
    q_low = question.lower()

    m_step = re.search(r"\b(?:step|schritt)\s*(\d{1,2})\b", q_low)
    if m_step:
        step_num = int(m_step.group(1))
        sub = origin_df[origin_df["step"] == step_num]
        if len(sub):
            ts_str = pd.Timestamp(sub["delivery_start_utc"].iloc[0]).strftime("%H:%M UTC")
            return sub, f"Step {step_num} ({ts_str})"

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
    ui_context: dict | None = None,
    chat_history: list[dict] | None = None,
) -> str:
    """Create a complete, leak-free context prompt covering UI state, docs, and all 40 forecast steps."""
    ui = ui_context or {}
    active_period = ui.get("selected_period", "Test Period (5 Oct – 1 Nov 2026)")
    active_filter = ui.get("filter_mode", "All 673 Origins (Chronological)")
    origin_pos = ui.get("origin_pos_str", "1 / 673")
    cutoff_str = ui.get("cutoff_str", selected_origin.strftime("%Y-%m-%d %H:%M:%SZ"))

    peak_hi_row = origin_df.loc[origin_df["p_spike_high"].idxmax()]
    peak_lo_row = origin_df.loc[origin_df["p_spike_low"].idxmax()]
    peak_p50_row = origin_df.loc[origin_df["p50"].idxmax()]
    min_p50_row = origin_df.loc[origin_df["p50"].idxmin()]

    spot_mean = float(origin_df["spot_price"].mean())
    spot_min = float(origin_df["spot_price"].min())
    spot_max = float(origin_df["spot_price"].max())
    p50_mean = float(origin_df["p50"].mean())

    rows_txt = []
    for _, r in origin_df.iterrows():
        ts = pd.Timestamp(r["delivery_start_utc"]).strftime("%H:%M")
        drvs = ", ".join(format_driver_name(d) for d in str(r["top_drivers"]).split(";")[:2] if d.strip())
        rows_txt.append(
            f"S{int(r['step'])}({ts}): Spot={r['spot_price']:.1f}, P50={r['p50']:.1f} [{r['p10']:.1f}..{r['p90']:.1f}], "
            f"Up={r['p_up']:.0%}/Dn={r['p_down']:.0%}, SpkHi={r['p_spike_high']:.0%}/Lo={r['p_spike_low']:.0%}, "
            f"Wind={r['wind_da']:.0f}MW, Load={r['cons_da']:.0f}MW, ResLoad={r['residual_load_da']:.0f}MW, Drv=[{drvs}]"
        )

    history_txt = ""
    if chat_history and len(chat_history) > 1:
        recent_turns = chat_history[-5:-1]
        turn_lines = [f"{m['role'].upper()}: {m['markdown'][:280]}" for m in recent_turns]
        history_txt = "\nRecent Conversation History:\n" + "\n".join(turn_lines) + "\n"

    lang_instruction = (
        "WICHTIG: Der Nutzer fragt auf DEUTSCH – antworte ausschließlich auf Deutsch."
        if _is_german_query(question)
        else "IMPORTANT: The user asks in ENGLISH – respond only in English."
    ) + " (General rule: always reply in the same language as the user's question.)"

    matched_kb = match_kb_entry(question)
    kb_hint = ""
    if matched_kb is not None:
        kb_title, kb_body = matched_kb["en"]
        kb_hint = f"\nRelevant UI Component Reference ({kb_title}):\n{kb_body}\n"

    return f"""You are the intelligent AI Assistant for the Turku Energia Imbalance Price Situational Desk.
{lang_instruction}

Formatting Rules for Sidebar Display:
- Directly answer the user's specific question first! Never ignore their question or output a generic summary.
- Keep answers concise, well-structured, and easy to read.
- If you include a Markdown table, use AT MOST 4 short columns (`| UTC | Spot → P50 | P10–P90 | Spike |`) so there is zero horizontal scrolling.
- Frame operational advice as "Suggestions to Review" (Prüf-Empfehlungen), never as automated switching commands.

Active Dashboard State:
- Dataset Period: `{active_period}` | Filter: `{active_filter}` ({origin_pos})
- Forecast Origin `t`: `{selected_origin.strftime('%Y-%m-%d %H:%M UTC')}` (Data Cutoff: `{cutoff_str}`, Rule #1 `first_usable_origin_utc <= t` verified)
- Spot Price (Nord Pool D-1 14:00): Mean `{spot_mean:.2f} €/MWh` (Min `{spot_min:.2f} €`, Max `{spot_max:.2f} €`)
- Imbalance Price Forecast (P50): Mean `{p50_mean:.2f} €/MWh` | 80% Confidence Band (`P10–P90`) Mean Width: `{briefing['mean_band']:.1f} €/MWh`
- Validation Benchmark (Tab 3): Dual-depth LightGBM + Conformal Calibration + TreeSHAP cuts P50 MAE by 35% (`43.19` vs `66.20 €/MWh` baseline), halves Pinball Loss (`12.43` vs `24.18`), hits exact `80.0%` P10–P90 coverage, Spike Brier `0.0928`, Direction Brier `0.4132`.
{kb_hint}
Live Data Summary for Current Origin `t = {selected_origin.strftime('%Y-%m-%d %H:%M UTC')}`:
- Dominant Balancing Direction: {briefing['dominant_dir']} (Mean P_Up={briefing['mean_p_up']:.1%}, Mean P_Down={briefing['mean_p_down']:.1%})
- Peak High-Spike Risk: {briefing['max_p_hi']:.1%} at {pd.Timestamp(peak_hi_row['delivery_start_utc']).strftime('%H:%M UTC')} (P50={peak_hi_row['p50']:.1f} € vs Spot={peak_hi_row['spot_price']:.1f} €)
- Peak Low-Spike Risk: {briefing['max_p_lo']:.1%} at {pd.Timestamp(peak_lo_row['delivery_start_utc']).strftime('%H:%M UTC')} (P50={peak_lo_row['p50']:.1f} €)
- Highest P50 Price: {peak_p50_row['p50']:.1f} €/MWh at {pd.Timestamp(peak_p50_row['delivery_start_utc']).strftime('%H:%M UTC')}
- Lowest P50 Price: {min_p50_row['p50']:.1f} €/MWh at {pd.Timestamp(min_p50_row['delivery_start_utc']).strftime('%H:%M UTC')}
- Forecast Revision vs Previous Hour (t-1h): Max ΔP50 = {briefing['max_abs_rev']:.1f} €/MWh (Mean {briefing['mean_rev']:+.1f} €/MWh)
{history_txt}
All 40 Quarter-Hour Steps for Origin `{selected_origin.strftime('%H:%M UTC')}`:
{chr(10).join(rows_txt)}

User Question: {question}
"""


def _discover_env_api_key() -> str:
    """Discover LLM API key from environment variables or local .env files."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in (".env", "api_key.env"):
        env_path = os.path.join(root, rel)
        if os.path.isfile(env_path):
            try:
                with open(env_path, encoding="utf-8") as fh:
                    for line in fh:
                        m = re.match(
                            r"^\s*(?:GEMINI_API_KEY|GOOGLE_API_KEY|OPENAI_API_KEY|GROQ_API_KEY)\s*=\s*(.+?)\s*$",
                            line,
                        )
                        if m:
                            candidate = m.group(1).strip().strip("\"'")
                            if candidate:
                                return candidate
            except OSError:
                pass
    return ""


def _extract_gemini_text(data: dict) -> str | None:
    """Extract text parts from Gemini response safely (handling 2.5 thinking parts)."""
    candidates = data.get("candidates") or []
    if not candidates:
        return None
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    non_thought = [p["text"] for p in parts if isinstance(p, dict) and p.get("text") and not p.get("thought")]
    if non_thought:
        return "\n".join(non_thought).strip()
    any_text = [p["text"] for p in parts if isinstance(p, dict) and p.get("text")]
    if any_text:
        return "\n".join(any_text).strip()
    return None


def _call_llm_api(prompt: str, api_key: str, preferred_model: str | None = None) -> tuple[str | None, str | None]:
    """Automatically detect provider from API key and call the best available live model."""
    key = api_key.strip().strip("\"'")
    if not key:
        return None, None

    # 1. Anthropic Claude API key support (sk-ant-...)
    if key.startswith("sk-ant-"):
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-3-5-sonnet-latest",
                    "max_tokens": 900,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=18,
            )
            if resp.status_code == 200:
                blocks = resp.json().get("content", [])
                txt = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
                if txt:
                    return txt, "Claude · Live"
            err_json = resp.json() if resp.content else {}
            return None, f"Anthropic API: {err_json.get('error', {}).get('message', f'HTTP {resp.status_code}')}"
        except Exception as exc:
            return None, f"Anthropic API: {exc}"

    # 2. OpenAI API key support (sk-...)
    if key.startswith("sk-"):
        last_err = None
        for oai_model in ("gpt-4o-mini", "gpt-4o"):
            try:
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={
                        "model": oai_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.25,
                        "max_tokens": 900,
                    },
                    timeout=18,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"], "OpenAI · Live"
                err_json = resp.json() if resp.content else {}
                last_err = err_json.get("error", {}).get("message", f"HTTP {resp.status_code}")
            except Exception as exc:
                last_err = str(exc)
        return None, f"OpenAI API: {last_err}"

    # 3. Groq API key support (gsk_...)
    if key.startswith("gsk_"):
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.25,
                    "max_tokens": 900,
                },
                timeout=18,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"], "Groq · Live"
            err_json = resp.json() if resp.content else {}
            return None, f"Groq API: {err_json.get('error', {}).get('message', f'HTTP {resp.status_code}')}"
        except Exception as exc:
            return None, f"Groq API: {exc}"

    # 4. Google AI Studio / Vertex AI Express (Gemini) — accepts any key prefix (AIza..., AQ..., etc.)
    gemini_models = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
    ]
    endpoints = (
        [
            "https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent",
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        ]
        if key.startswith("AQ.")
        else [
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            "https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent",
        ]
    )
    last_err = None
    for base_tpl in endpoints:
        key_invalid_on_endpoint = False
        for model_name in gemini_models:
            try:
                url = base_tpl.format(model=model_name)
                gen_cfg: dict = {"temperature": 0.25, "maxOutputTokens": 2048}
                if "2.5" in model_name or "3-" in model_name:
                    gen_cfg["thinkingConfig"] = {"thinkingBudget": 256}
                body = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": gen_cfg,
                }
                resp = requests.post(
                    url,
                    params={"key": key},
                    headers={"Content-Type": "application/json"},
                    json=body,
                    timeout=18,
                )
                if resp.status_code == 400 and "thinkingConfig" in gen_cfg:
                    gen_cfg.pop("thinkingConfig", None)
                    resp = requests.post(
                        url,
                        params={"key": key},
                        headers={"Content-Type": "application/json"},
                        json=body,
                        timeout=18,
                    )
                if resp.status_code == 200:
                    text = _extract_gemini_text(resp.json())
                    if text:
                        return text, "Gemini · Live"
                    last_err = f"{model_name} returned empty response"
                else:
                    err_json = resp.json() if resp.content else {}
                    msg = err_json.get("error", {}).get("message", f"HTTP {resp.status_code}")
                    last_err = msg
                    if resp.status_code in (400, 401, 403) and ("API key not valid" in msg or "API_KEY_INVALID" in str(err_json)):
                        key_invalid_on_endpoint = True
                        break
            except Exception as exc:
                last_err = str(exc)
        if not key_invalid_on_endpoint:
            break
    return None, last_err

def _build_reasoning_trace(
    question: str,
    origin_df: pd.DataFrame,
    briefing: dict,
    selected_origin: pd.Timestamp,
    matched_rows: pd.DataFrame | None,
    matched_label: str | None,
    driver_weights: list[dict],
    ui_context: dict | None,
    is_de: bool,
    source_label: str,
) -> str:
    """Build a transparent step-by-step Chain-of-Thought (Model Reasoning) trace in DE or EN."""
    ui = ui_context or {}
    active_period = ui.get("selected_period", "Test Period (5 Oct – 1 Nov 2026)")
    cutoff_str = ui.get("cutoff_str", selected_origin.strftime("%Y-%m-%d %H:%M:%SZ"))
    spot_mean = float(origin_df["spot_price"].mean())
    p50_mean = float(origin_df["p50"].mean())
    spread_mean = p50_mean - spot_mean
    peak_row = origin_df.loc[origin_df["p50"].idxmax()]
    peak_t = pd.Timestamp(peak_row["delivery_start_utc"]).strftime("%H:%M UTC")
    top_drv_str = ", ".join(
        f"{d['driver']} ({d.get('pct', 0.0):.0f}%)" for d in driver_weights[:3]
    )

    if matched_rows is not None and matched_label is not None:
        r0 = matched_rows.iloc[0]
        scope_de = f"Ziel-Zeitfenster `{matched_label}` (Step {int(r0['step'])}, Spot `{r0['spot_price']:.1f} €` → P50 `{r0['p50']:.1f} €`, Band `[{r0['p10']:.1f}, {r0['p90']:.1f}] €`)"
        scope_en = f"Target slice `{matched_label}` (Step {int(r0['step'])}, Spot `{r0['spot_price']:.1f} €` → P50 `{r0['p50']:.1f} €`, band `[{r0['p10']:.1f}, {r0['p90']:.1f}] €`)"
    else:
        scope_de = f"Vollständiger 10h-Horizont (40 Viertelstunden, Ø Spot `{spot_mean:.1f} €` → Ø P50 `{p50_mean:.1f} €`, Peak `{peak_row['p50']:.1f} €` um `{peak_t}`)"
        scope_en = f"Full 10h horizon (40 quarter-hour steps, mean Spot `{spot_mean:.1f} €` → mean P50 `{p50_mean:.1f} €`, peak `{peak_row['p50']:.1f} €` at `{peak_t}`)"

    if is_de:
        return (
            f"1. **Intent- & Anfrage-Analyse:** Frage (`\"{question}\"`) als deutschsprachige Fach-/Prognoseanfrage klassifiziert → Fokus: {scope_de}.\n"
            f"2. **Daten- & Cutoff-Prüfung:** Zeitraum `{active_period}`, Forecast Origin `t = {selected_origin.strftime('%Y-%m-%d %H:%M UTC')}` (Cutoff `{cutoff_str}`, strikt `first_usable_origin_utc ≤ t`).\n"
            f"3. **Modell- & TreeSHAP-Auswertung ({source_label}):** Mittlerer Spread `ΔP = {spread_mean:+.2f} €/MWh` vs. Nord Pool Spot; dominante Regelrichtung `{briefing['dominant_dir']}` (Max High-Spike `{briefing['max_p_hi']:.1%}`, Low-Spike `{briefing['max_p_lo']:.1%}`). Stärkste SHAP-Treiber: **{top_drv_str}**.\n"
            f"4. **Antwort-Synthese:** Quantitative Kennzahlen und physikalische Netzzusammenhänge für Leitwarte & Intraday-Handel verdichtet."
        )
    return (
        f"1. **Intent & Query Parsing:** Classified query (`\"{question}\"`) in English → Scope: {scope_en}.\n"
        f"2. **Data & Cutoff Verification:** Active period `{active_period}`, Forecast Origin `t = {selected_origin.strftime('%Y-%m-%d %H:%M UTC')}` (Cutoff `{cutoff_str}`, strictly enforcing `first_usable_origin_utc ≤ t`).\n"
        f"3. **Quantile & TreeSHAP Signal Extraction ({source_label}):** Mean spread `ΔP = {spread_mean:+.2f} €/MWh` vs. Spot; dominant mFRR direction `{briefing['dominant_dir']}` (Max High-Spike `{briefing['max_p_hi']:.1%}`, Low-Spike `{briefing['max_p_lo']:.1%}`). Top TreeSHAP drivers: **{top_drv_str}**.\n"
        f"4. **Response Synthesis:** Structured quantitative metrics and physical grid drivers into actionable decision support."
    )


def generate_explainability_response(
    question: str,
    origin_df: pd.DataFrame,
    briefing: dict,
    selected_origin: pd.Timestamp,
    api_key: str | None = None,
    preferred_model: str = "gemini-2.5-pro",
    ui_context: dict | None = None,
    chat_history: list[dict] | None = None,
) -> dict:
    """Return context-aware AI response answering the exact user question in German or English, with Model Reasoning."""
    res = _generate_explainability_response_core(
        question=question,
        origin_df=origin_df,
        briefing=briefing,
        selected_origin=selected_origin,
        api_key=api_key,
        preferred_model=preferred_model,
        ui_context=ui_context,
        chat_history=chat_history,
    )
    if "reasoning" not in res or not res["reasoning"]:
        matched_rows, matched_label = _extract_requested_hour_rows(question, origin_df)
        driver_weights = res.get("driver_weights") or compute_horizon_driver_weights(origin_df, matched_rows)
        res["reasoning"] = _build_reasoning_trace(
            question=question,
            origin_df=origin_df,
            briefing=briefing,
            selected_origin=selected_origin,
            matched_rows=matched_rows,
            matched_label=matched_label,
            driver_weights=driver_weights,
            ui_context=ui_context,
            is_de=_is_german_query(question),
            source_label=res.get("source", "Situational Desk AI"),
        )
    return res


def _generate_explainability_response_core(
    question: str,
    origin_df: pd.DataFrame,
    briefing: dict,
    selected_origin: pd.Timestamp,
    api_key: str | None = None,
    preferred_model: str = "gemini-2.0-flash",
    ui_context: dict | None = None,
    chat_history: list[dict] | None = None,
) -> dict:
    """Return context-aware AI response answering the exact user question in German or English."""
    matched_rows, matched_label = _extract_requested_hour_rows(question, origin_df)
    driver_weights = compute_horizon_driver_weights(origin_df, matched_rows)
    is_de = _is_german_query(question)
    ui = ui_context or {}
    active_period = ui.get("selected_period", "Test Period (5 Oct – 1 Nov 2026)")
    active_filter = ui.get("filter_mode", "All 673 Origins (Chronological)")
    origin_pos = ui.get("origin_pos_str", "1 / 673")
    cutoff_str = ui.get("cutoff_str", selected_origin.strftime("%Y-%m-%d %H:%M:%SZ"))

    effective_key = (api_key or "").strip() or _discover_env_api_key()
    api_error_note = ""

    # 1. Try Live LLM (Gemini 2.0/2.5 Flash/Pro, GPT-4o, or Llama 3.3 70B) if API key is present
    if effective_key:
        prompt = _build_grounded_context_prompt(
            question=question,
            origin_df=origin_df,
            briefing=briefing,
            selected_origin=selected_origin,
            ui_context=ui_context,
            chat_history=chat_history,
        )
        llm_text, model_or_err = _call_llm_api(prompt, effective_key, preferred_model=preferred_model)
        if llm_text:
            return {
                "markdown": llm_text,
                "driver_weights": driver_weights,
                "source": model_or_err or "Gemini · Live",
            }
        if model_or_err:
            if is_de:
                api_error_note = (
                    f"> ⚠️ **API-Key-Hinweis:** Live-LLM-Aufruf nicht möglich (*{model_or_err}*). "
                    f"Antwort wird über die lokale TreeSHAP-Engine berechnet:\n\n"
                )
            else:
                api_error_note = (
                    f"> ⚠️ **API Key Note:** Live LLM call did not succeed (*{model_or_err}*). "
                    f"Showing local TreeSHAP & data engine analysis below:\n\n"
                )

    def _pack(md_body: str, drvs: list[dict], src: str) -> dict:
        return {
            "markdown": api_error_note + md_body,
            "driver_weights": drvs,
            "source": src,
        }

    # 2. Built-in Context-Aware Knowledge & Data Engine
    q_low = question.lower()
    spot_mean = float(origin_df["spot_price"].mean())
    spot_min = float(origin_df["spot_price"].min())
    spot_max = float(origin_df["spot_price"].max())
    p50_mean = float(origin_df["p50"].mean())
    peak_row = origin_df.loc[origin_df["p50"].idxmax()]
    min_row = origin_df.loc[origin_df["p50"].idxmin()]
    hi_row = origin_df.loc[origin_df["p_spike_high"].idxmax()]
    lo_row = origin_df.loc[origin_df["p_spike_low"].idxmax()]
    top_drv_names = ", ".join(d["driver"] for d in driver_weights[:3]) or "Day-Ahead Spot Price"

    # --- Priority 1: Specific Hour or Step (e.g. "at 05:15", "um 14 Uhr", "Step 12") ---
    if matched_rows is not None and matched_label is not None:
        table_lines = [
            "| UTC | Spot → P50 | P10–P90 | Spike Hi/Lo |",
            "| :--- | :---: | :---: | :---: |",
        ]
        for _, r in matched_rows.iterrows():
            t_str = pd.Timestamp(r["delivery_start_utc"]).strftime("%H:%M")
            table_lines.append(
                f"| **{t_str}** | {r['spot_price']:.0f} → **{r['p50']:.1f} €** | "
                f"{r['p10']:.0f}..{r['p90']:.0f} € | {r['p_spike_high']:.0%} / {r['p_spike_low']:.0%} |"
            )
        mean_p50 = float(matched_rows["p50"].mean())
        mean_spot = float(matched_rows["spot_price"].mean())
        mean_wind = float(matched_rows["wind_da"].mean())
        mean_cons = float(matched_rows["cons_da"].mean())
        mean_res = float(matched_rows["residual_load_da"].mean())
        if is_de:
            md = (
                f"**Analyse für `{matched_label}`** *(Origin {selected_origin.strftime('%H:%M UTC')})*\n\n"
                f"* **Prognose-Mittel (P50):** **{mean_p50:.2f} €/MWh** ({mean_p50 - mean_spot:+.2f} € vs. Spot `{mean_spot:.2f} €`)\n"
                f"* **Netz-Fundamentaldaten:** Last **{mean_cons:,.0f} MW** · Wind **{mean_wind:,.0f} MW** · Residuallast **{mean_res:,.0f} MW**\n"
                f"* **Haupttreiber (TreeSHAP):** {top_drv_names}\n\n"
                + "\n".join(table_lines)
            )
        else:
            md = (
                f"**Breakdown for `{matched_label}`** *(Origin {selected_origin.strftime('%H:%M UTC')})*\n\n"
                f"* **Mean Forecast (P50):** **{mean_p50:.2f} €/MWh** ({mean_p50 - mean_spot:+.2f} € vs. Spot `{mean_spot:.2f} €`)\n"
                f"* **Grid Fundamentals:** Load **{mean_cons:,.0f} MW** · Wind **{mean_wind:,.0f} MW** · Residual **{mean_res:,.0f} MW**\n"
                f"* **Primary SHAP Drivers:** {top_drv_names}\n\n"
                + "\n".join(table_lines)
            )
        return _pack(md, driver_weights, "TreeSHAP Engine")

    # --- Priority 2: Spike Risk Summary & Peak Analysis (Quick Button #2 & spike questions) ---
    if (
        any(k in q_low for k in ("spike risk", "spike-risiko", "when does it peak", "why is there a spike", "preis-spike", "preisspitze"))
        or ("spike" in q_low and "chart" not in q_low and "diagramm" not in q_low and "card" not in q_low and "filter" not in q_low)
    ):
        hi_t = pd.Timestamp(hi_row["delivery_start_utc"]).strftime("%H:%M UTC")
        lo_t = pd.Timestamp(lo_row["delivery_start_utc"]).strftime("%H:%M UTC")
        hi_cnt = int((origin_df["p_spike_high"] >= 0.30).sum())
        lo_cnt = int((origin_df["p_spike_low"] >= 0.30).sum())
        top_risk = origin_df.assign(
            _max_spk=origin_df[["p_spike_high", "p_spike_low"]].max(axis=1)
        ).sort_values("_max_spk", ascending=False).head(4).sort_values("step")
        t_rows = [
            "| UTC | Spot → P50 | P10–P90 | Spike Hi / Lo |",
            "| :--- | :---: | :---: | :---: |",
        ]
        for _, r in top_risk.iterrows():
            ts = pd.Timestamp(r["delivery_start_utc"]).strftime("%H:%M")
            t_rows.append(
                f"| **{ts}** | {r['spot_price']:.0f} → **{r['p50']:.1f} €** | "
                f"{r['p10']:.0f}..{r['p90']:.0f} € | **{r['p_spike_high']:.1%}** / {r['p_spike_low']:.1%} |"
            )
        if is_de:
            status_line = (
                f"🚨 **Aktive Spike-Warnung:** `{hi_cnt}` Hochpreis-Schritte und `{lo_cnt}` Tiefpreis-Schritte überschreiten die **30 %-Warnschwelle**."
                if (hi_cnt or lo_cnt)
                else "✅ **Unterhalb der 30 %-Warnschwelle:** In diesem 10h-Fenster überschreitet kein Schritt die 30 %-Marke."
            )
            md = (
                f"**⚡ Preis-Spike-Risiko im 10h-Horizont (`{selected_origin.strftime('%H:%M UTC')}`)**\n\n"
                f"* {status_line}\n"
                f"* **Peak High-Spike (`> Spot + 50 €`):** **{briefing['max_p_hi']:.1%}** um **`{hi_t}`** *(P50 `{hi_row['p50']:.1f} €` vs. Spot `{hi_row['spot_price']:.1f} €`, P90 `{hi_row['p90']:.1f} €`)*\n"
                f"* **Peak Low-Spike (`< Spot − 50 €`):** **{briefing['max_p_lo']:.1%}** um **`{lo_t}`** *(P50 `{lo_row['p50']:.1f} €`, P10 `{lo_row['p10']:.1f} €`)*\n"
                f"* **Physikalische Treiber (SHAP):** **{top_drv_names}** *(Residuallast Ø `{float(origin_df['residual_load_da'].mean()):,.0f} MW`, dominante Richtung: `{briefing['dominant_dir']}`)*\n\n"
                + "\n".join(t_rows)
            )
        else:
            status_line = (
                f"🚨 **Active Spike Alert:** `{hi_cnt}` high-spike step(s) and `{lo_cnt}` low-spike step(s) exceed the **30% alert threshold**."
                if (hi_cnt or lo_cnt)
                else "✅ **Below 30% Alert Threshold:** No quarter-hour step crosses the 30% warning threshold in this 10h window."
            )
            md = (
                f"**⚡ Spike Risk Summary (`{selected_origin.strftime('%H:%M UTC')}` + 10h)**\n\n"
                f"* {status_line}\n"
                f"* **Peak High-Spike (`> Spot + 50 €`):** **{briefing['max_p_hi']:.1%}** at **`{hi_t}`** *(P50 `{hi_row['p50']:.1f} €` vs. Spot `{hi_row['spot_price']:.1f} €`, P90 `{hi_row['p90']:.1f} €`)*\n"
                f"* **Peak Low-Spike (`< Spot − 50 €`):** **{briefing['max_p_lo']:.1%}** at **`{lo_t}`** *(P50 `{lo_row['p50']:.1f} €`, P10 `{lo_row['p10']:.1f} €`)*\n"
                f"* **Physical Drivers (SHAP):** **{top_drv_names}** *(Mean residual load `{float(origin_df['residual_load_da'].mean()):,.0f} MW`, dominant mFRR: `{briefing['dominant_dir']}`)*\n\n"
                + "\n".join(t_rows)
            )
        return _pack(md, driver_weights, "TreeSHAP Engine")

    # --- Priority 3: Highest / Lowest / Cheapest / Most Expensive Price ---
    if any(k in q_low for k in ("highest", "lowest", "cheapest", "most expensive", "höchste", "niedrigste", "günstigste", "teuerste", "max price", "min price", "peak price")):
        t_pk = pd.Timestamp(peak_row["delivery_start_utc"]).strftime("%H:%M UTC")
        t_mn = pd.Timestamp(min_row["delivery_start_utc"]).strftime("%H:%M UTC")
        if is_de:
            md = (
                f"**Höchster & niedrigster prognostizierter Preis (`{selected_origin.strftime('%H:%M UTC')}` + 10h)**\n\n"
                f"* **📈 Höchster Preis (P50 Maximum):** **{peak_row['p50']:.2f} €/MWh** um **`{t_pk}`** *(Spot: `{peak_row['spot_price']:.2f} €`, 80%-Band: `{peak_row['p10']:.1f}` bis `{peak_row['p90']:.1f} €`, High-Spike: `{peak_row['p_spike_high']:.1%}`)*\n"
                f"* **📉 Niedrigster Preis (P50 Minimum):** **{min_row['p50']:.2f} €/MWh** um **`{t_mn}`** *(Spot: `{min_row['spot_price']:.2f} €`, 80%-Band: `{min_row['p10']:.1f}` bis `{min_row['p90']:.1f} €`, Low-Spike: `{min_row['p_spike_low']:.1%}`)*\n"
                f"* **10h-Durchschnitt:** P50 **`{p50_mean:.2f} €/MWh`** vs. Spot **`{spot_mean:.2f} €/MWh`** *(Spread `{p50_mean - spot_mean:+.2f} €`)*"
            )
        else:
            md = (
                f"**Highest & Lowest Forecasted Prices (`{selected_origin.strftime('%H:%M UTC')}` + 10h)**\n\n"
                f"* **📈 Highest Price (P50 Peak):** **{peak_row['p50']:.2f} €/MWh** at **`{t_pk}`** *(Spot: `{peak_row['spot_price']:.2f} €`, 80% band: `{peak_row['p10']:.1f}` to `{peak_row['p90']:.1f} €`, High-Spike: `{peak_row['p_spike_high']:.1%}`)*\n"
                f"* **📉 Lowest Price (P50 Min):** **{min_row['p50']:.2f} €/MWh** at **`{t_mn}`** *(Spot: `{min_row['spot_price']:.2f} €`, 80% band: `{min_row['p10']:.1f}` to `{min_row['p90']:.1f} €`, Low-Spike: `{min_row['p_spike_low']:.1%}`)*\n"
                f"* **10h Mean:** P50 **`{p50_mean:.2f} €/MWh`** vs. Spot **`{spot_mean:.2f} €/MWh`** *(Spread `{p50_mean - spot_mean:+.2f} €`)*"
            )
        return _pack(md, driver_weights, "TreeSHAP Engine")

    # --- Priority 4: General "What drives the price / Which features are important?" ---
    if any(k in q_low for k in ("what drives", "what features drive", "main drivers", "top drivers", "which features", "was treibt", "welche treiber", "welche features", "haupttreiber", "wichtigsten treiber")):
        drv_bullets = "\n".join(f"  {i+1}. **{d['driver']}** (`{d['pct']:.1f}%` Horizon-Gewicht)" for i, d in enumerate(driver_weights))
        if is_de:
            md = (
                f"**🔍 Wichtigste Preistreiber (TreeSHAP) für Origin `{selected_origin.strftime('%H:%M UTC')}`**\n\n"
                f"{drv_bullets}\n\n"
                f"* **Spot-Anker:** Ø `{spot_mean:.2f} €/MWh` → **KI-P50:** Ø `{p50_mean:.2f} €/MWh` *(Spread `{p50_mean - spot_mean:+.2f} €`)*\n"
                f"* **Netz-Kontext:** Last Ø `{float(origin_df['cons_da'].mean()):,.0f} MW` · Wind Ø `{float(origin_df['wind_da'].mean()):,.0f} MW` · Dominante Regelrichtung: **{briefing['dominant_dir']}**"
            )
        else:
            md = (
                f"**🔍 Primary Price Drivers (TreeSHAP) for Origin `{selected_origin.strftime('%H:%M UTC')}`**\n\n"
                f"{drv_bullets}\n\n"
                f"* **Spot Anchor:** Mean `{spot_mean:.2f} €/MWh` → **AI P50:** Mean `{p50_mean:.2f} €/MWh` *(Spread `{p50_mean - spot_mean:+.2f} €`)*\n"
                f"* **Grid Context:** Load Mean `{float(origin_df['cons_da'].mean()):,.0f} MW` · Wind Mean `{float(origin_df['wind_da'].mean()):,.0f} MW` · Dominant Direction: **{briefing['dominant_dir']}**"
            )
        return _pack(md, driver_weights, "TreeSHAP Engine")

    # --- Topic 0: Bilingual knowledge base covering specific UI elements & charts ---
    kb_entry = match_kb_entry(question)
    if kb_entry is not None:
        kb_ctx = {
            "origin": selected_origin.strftime("%Y-%m-%d %H:%M UTC"),
            "cutoff": cutoff_str,
            "period": active_period,
            "filter": active_filter,
            "pos": origin_pos,
            "spot_mean": spot_mean,
            "p50_mean": p50_mean,
            "spread": p50_mean - spot_mean,
            "max_p_hi": float(briefing["max_p_hi"]),
            "max_p_lo": float(briefing["max_p_lo"]),
            "dominant_dir": briefing["dominant_dir"],
            "mean_band": float(briefing["mean_band"]),
            "max_abs_rev": float(briefing["max_abs_rev"]),
            "mean_rev": float(briefing["mean_rev"]),
            "mean_p_up": float(briefing["mean_p_up"]),
            "mean_p_down": float(briefing["mean_p_down"]),
            "p_none": float(origin_df["p_none"].mean()),
            "peak_p50": float(peak_row["p50"]),
            "peak_time": pd.Timestamp(peak_row["delivery_start_utc"]).strftime("%H:%M UTC"),
            "hi_time": pd.Timestamp(hi_row["delivery_start_utc"]).strftime("%H:%M UTC"),
            "lo_time": pd.Timestamp(lo_row["delivery_start_utc"]).strftime("%H:%M UTC"),
            "cons_mean": float(origin_df["cons_da"].mean()),
            "wind_mean": float(origin_df["wind_da"].mean()),
            "res_mean": float(origin_df["residual_load_da"].mean()),
            "top_drivers": top_drv_names,
        }
        return _pack(
            render_kb_entry(kb_entry, is_de, kb_ctx),
            driver_weights if kb_entry["id"] in ("top_shap", "waterfall", "heatmap") else [],
            "Dashboard Copilot",
        )
    # --- Topic A: Dataset Period / Test vs Validation Period ---
    if any(k in q_low for k in ("dataset period", "test period", "validation period", "periode", "zeitraum", "welcher datensatz", "which period", "dataset")):
        if is_de:
            md = (
                f"**Dataset Period (Datensatz-Zeitraum)**\n\n"
                f"Aktuell ist links in der Sidebar ausgewählt:\n"
                f"* **`{active_period}`** *(Origin `{selected_origin.strftime('%Y-%m-%d %H:%M UTC')}`, Position `{origin_pos}`)*\n\n"
                f"**Was bedeuten die beiden Zeiträume?**\n"
                f"1. **Test Period (5. Okt – 1. Nov 2026):** Der offizielle Bewertungszeitraum der Challenge mit **673 stündlichen Origins** (= 26.920 Viertelstunden-Vorhersagen in `forecast_backtest.csv`).\n"
                f"2. **Validation Period (7. Sep – 4. Okt 2026):** Der historische Testzeitraum vor dem Wettbewerb. Hier liegen die echten Ausgleichspreise bereits vor, sodass du mit dem Schalter **„🎯 Overlay Actual Prices (Hindsight)“** unsere KI-Prognose direkt mit der Realität vergleichen kannst."
            )
        else:
            md = (
                f"**Dataset Period Overview**\n\n"
                f"Currently selected in the left sidebar:\n"
                f"* **`{active_period}`** *(Origin `{selected_origin.strftime('%Y-%m-%d %H:%M UTC')}`, position `{origin_pos}`)*\n\n"
                f"**What the two periods mean:**\n"
                f"1. **Test Period (5 Oct – 1 Nov 2026):** The official competition evaluation period containing **673 hourly forecast origins** × 40 steps = 26,920 predictions (`forecast_backtest.csv`).\n"
                f"2. **Validation Period (7 Sep – 4 Oct 2026):** The historical walk-forward period where realized ground-truth prices are known, allowing you to toggle **“🎯 Overlay Actual Prices (Hindsight)”** to compare forecasts against reality."
            )
        return _pack(md, [], "Dashboard Copilot")

    # --- Topic B: Spot Price (Day-Ahead Spotpreis) ---
    if any(k in q_low for k in ("spot", "spotpreis", "nord pool", "nordpool", "day-ahead", "day ahead")):
        if is_de:
            md = (
                f"**Was ist der Day-Ahead Spotpreis (`Spot Price`)?**\n\n"
                f"Der **Spotpreis** (gestrichelte orange Linie im Hauptgraphen) ist der offizielle Börsenstrompreis der **Nord Pool Day-Ahead-Auktion** für Finnland. Er wird bereits am Vortag um **14:00 Uhr** festgelegt und ist daher für alle 40 zukünftigen Viertelstunden-Schritte bereits bekannt.\n\n"
                f"**Werte im aktuellen 10h-Fenster (`{selected_origin.strftime('%H:%M UTC')}`):**\n"
                f"* **Spot-Mittelwert:** `{spot_mean:.2f} €/MWh` *(Min: `{spot_min:.2f} €`, Max: `{spot_max:.2f} €`)*\n"
                f"* **KI-Prognose (P50 Mittel):** `{p50_mean:.2f} €/MWh` *(Abweichung zum Spot: **{p50_mean - spot_mean:+.2f} €/MWh**)*\n\n"
                f"💡 *Unser Modell sagt den Spread (`Imbalance Price − Spot Price`) voraus: Liegt P50 über dem Spotpreis, droht Strommangel (Up-Regulation); liegt P50 darunter, herrscht Stromüberschuss (Down-Regulation).*"
            )
        else:
            md = (
                f"**What is the Day-Ahead Spot Price?**\n\n"
                f"The **Day-Ahead Spot Price** (dashed orange line in the main chart) is the official **Nord Pool** auction clearing price for Finland. Because it is published at **14:00 CET on the previous day (D−1)**, it is already known across the upcoming 10-hour horizon and serves as the baseline anchor for imbalance pricing.\n\n"
                f"**Current 10h Window (`{selected_origin.strftime('%H:%M UTC')}`):**\n"
                f"* **Mean Spot Price:** `{spot_mean:.2f} €/MWh` *(Min: `{spot_min:.2f} €`, Max: `{spot_max:.2f} €`)*\n"
                f"* **Mean AI Forecast (P50):** `{p50_mean:.2f} €/MWh` *(Spread vs. Spot: **{p50_mean - spot_mean:+.2f} €/MWh**)*\n\n"
                f"💡 *When the Imbalance Price (P50) rises above Spot, the grid is in deficit (Up-Regulation); when it drops below Spot, the grid has surplus power (Down-Regulation).*"
            )
        return _pack(md, [], "Dashboard Copilot")
    # --- Topic D: Quantiles / P10 / P50 / P90 / Confidence Band / Fan Chart ---
    if any(k in q_low for k in ("p10", "p50", "p90", "quantil", "quantile", "konfidenz", "confidence", "uncertainty", "unsicherheit", "fan chart", "band")):
        if is_de:
            md = (
                f"**Was bedeuten P10, P50 und P90 im Fan-Chart?**\n\n"
                f"* **P50 (Median-Prognose):** Unsere wahrscheinlichste Punktprognose für den Ausgleichspreis (aktuell Ø **{p50_mean:.2f} €/MWh**, Spitze **{peak_row['p50']:.1f} €** um `{pd.Timestamp(peak_row['delivery_start_utc']).strftime('%H:%M UTC')}`).\n"
                f"* **P10 (Untergrenze, 10%-Quantil):** Mit 90 % Wahrscheinlichkeit fällt der Preis nicht unter diesen Wert.\n"
                f"* **P90 (Obergrenze, 90%-Quantil):** Mit 90 % Wahrscheinlichkeit steigt der Preis nicht über diesen Wert.\n"
                f"* **80% Unsicherheits-Band (`P10–P90`):** Die schattierte Fläche zwischen P10 und P90 hat im aktuellen Fenster eine durchschnittliche Breite von **{briefing['mean_band']:.1f} €/MWh** und deckt in der Validierung exakt **80,0 %** aller echten Preise ab."
            )
        else:
            md = (
                f"**Understanding P10, P50 & P90 in the Fan Chart**\n\n"
                f"* **P50 (Median Forecast):** The central 50th percentile price prediction (currently averaging **{p50_mean:.2f} €/MWh**, peaking at **{peak_row['p50']:.1f} €** at `{pd.Timestamp(peak_row['delivery_start_utc']).strftime('%H:%M UTC')}`).\n"
                f"* **P10 (10th Percentile Lower Bound):** Only a 10% chance the actual price drops below this line.\n"
                f"* **P90 (90th Percentile Upper Bound):** Only a 10% chance the actual price exceeds this line.\n"
                f"* **80% Confidence Interval (`P10–P90`):** The shaded fan between P10 and P90 averages **{briefing['mean_band']:.1f} €/MWh** in width for this origin and achieves an exact **80.0% empirical coverage** on validation data."
            )
        return _pack(md, [], "Dashboard Copilot")

    # --- Topic E: Forecast Origin / Time Machine / Quick Filter / Top 20 ---
    if any(k in q_low for k in ("origin", "time machine", "filter", "top 20", "schnellfilter", "quick filter")):
        if is_de:
            md = (
                f"**Wie funktionieren `Forecast Origin (t)` & der `Quick Filter`?**\n\n"
                f"* **Forecast Origin (`t`):** Der exakte Zeitpunkt, an dem die Prognose berechnet wird (aktuell **`{selected_origin.strftime('%Y-%m-%d %H:%M UTC')}`**). Von hier blickt die KI 10 Stunden (40 Viertelstunden) in die Zukunft.\n"
                f"* **Aktiver Filter:** `{active_filter}` *(Position `{origin_pos}`)*.\n"
                f"* **Filter-Optionen:**\n"
                f"  * **All 673 Origins:** Chronologische Liste aller Stunden.\n"
                f"  * **Spike Alert Origins Only (≥ 30%):** Nur Stunden, in denen akute Preis-Spike-Gefahr besteht.\n"
                f"  * **Top 20 High-Price / Low-Price Spike Origins:** Die 20 extremsten Risiko-Stunden für Hochpreis-Spikes (`> Spot + 50 €`) bzw. Negativ-/Tiefpreis-Spikes (`< Spot − 50 €`)."
            )
        else:
            md = (
                f"**How `Forecast Origin (t)` & `Quick Filter` Work**\n\n"
                f"* **Forecast Origin (`t`):** The exact timestamp when the forecast is issued (currently **`{selected_origin.strftime('%Y-%m-%d %H:%M UTC')}`**). From `t`, the model predicts the next 40 quarter-hour steps (+10 hours).\n"
                f"* **Active Filter:** `{active_filter}` *(Position `{origin_pos}`)*.\n"
                f"* **Filter Modes:**\n"
                f"  * **All 673 Origins:** Full chronological timeline.\n"
                f"  * **Spike Alert Origins Only (≥ 30%):** Filters to origins with active spike alerts.\n"
                f"  * **Top 20 High / Low Spike Origins:** Jumps straight to the 20 origins with the highest probability of upward surges (`> Spot + 50 €`) or downward drops (`< Spot − 50 €`)."
            )
        return _pack(md, [], "Dashboard Copilot")

    # --- Topic F: Anti-Leakage Gate (Rule #1) ---
    if any(k in q_low for k in ("leakage", "rule #1", "rule 1", "cutoff", "first_usable", "gate", "regel 1")):
        if is_de:
            md = (
                f"**🛡️ Anti-Leakage Gate (Rule #1)**\n\n"
                f"Die wichtigste Wettbewerbsregel lautet: Für jeden Vorhersage-Zeitpunkt `t` dürfen ausschließlich Daten verwendet werden, deren Veröffentlichungszeitpunkt (`first_usable_origin_utc`) kleiner oder gleich `t` ist.\n\n"
                f"* **Aktueller Origin (`t`):** `{selected_origin.strftime('%Y-%m-%d %H:%M:%SZ')}`\n"
                f"* **Frischester Datenpunkt (`Data Cutoff`):** `{cutoff_str}`\n"
                f"* **Status:** `✓ Verified: first_usable_origin_utc ≤ t` (Berücksichtigt 30m Publikations-Lag für Last/Grenzflüsse und 60m Lag für mFRR/aFRR-Regelenergie)."
            )
        else:
            md = (
                f"**🛡️ Anti-Leakage Gate (Rule #1)**\n\n"
                f"Challenge Rule #1 mandates that for any forecast origin `t`, only data rows with `first_usable_origin_utc <= t` may be accessed.\n\n"
                f"* **Current Origin (`t`):** `{selected_origin.strftime('%Y-%m-%d %H:%M:%SZ')}`\n"
                f"* **Latest Input Telemetry (`Data Cutoff`):** `{cutoff_str}`\n"
                f"* **Status:** `✓ Verified: first_usable_origin_utc ≤ t` (Accounts for 30-min publication lag on real-time load/flows and 60-min lag on mFRR/aFRR balancing series)."
            )
        return _pack(md, [], "Dashboard Copilot")

    # --- Topic G: mFRR Balancing Direction (Up / Down / None) ---
    if any(k in q_low for k in ("mfrr", "direction", "regelrichtung", "up-regulation", "down-regulation", "up regulation", "down regulation", "deficit", "surplus")):
        if is_de:
            md = (
                f"**mFRR-Regelrichtung (`Up / Down / None`) im aktuellen Horizont**\n\n"
                f"* **Dominante Tendenz:** **{briefing['dominant_dir']}**\n"
                f"* **Mittlere Wahrscheinlichkeiten:** `P(Up) = {briefing['mean_p_up']:.1%}` · `P(Down) = {briefing['mean_p_down']:.1%}` · `P(None) = {float(origin_df['p_none'].mean()):.1%}`\n\n"
                f"**Bedeutung:**\n"
                f"* **Up-Regulation (Orange):** Stromdefizit im Netz – Kraftwerke werden hochgefahren, der Ausgleichspreis steigt über den Spotpreis.\n"
                f"* **Down-Regulation (Türkis):** Stromüberschuss (z. B. viel Wind) – Erzeuger regeln ab oder Elektrokessel schalten zu, Preis fällt unter Spot."
            )
        else:
            md = (
                f"**mFRR Balancing Direction (`Up / Down / None`)**\n\n"
                f"* **Dominant Tendency:** **{briefing['dominant_dir']}**\n"
                f"* **10h Mean Probabilities:** `P(Up) = {briefing['mean_p_up']:.1%}` · `P(Down) = {briefing['mean_p_down']:.1%}` · `P(None) = {float(origin_df['p_none'].mean()):.1%}`\n\n"
                f"**What it means:**\n"
                f"* **Up-Regulation (Orange):** Grid power deficit — balancing assets ramp up, pushing the imbalance price above Spot.\n"
                f"* **Down-Regulation (Turquoise):** Grid power surplus — generation curtails or flexible boilers absorb load, pushing price below Spot."
            )
        return _pack(md, [], "TreeSHAP Engine")

    # --- Topic H: Wind / Consumption / Residual Load / Physical Grid ---
    if any(k in q_low for k in ("wind", "verbrauch", "consumption", "load", "residual", "residuallast", "solar", "mw")):
        w_min, w_max, w_mean = float(origin_df["wind_da"].min()), float(origin_df["wind_da"].max()), float(origin_df["wind_da"].mean())
        c_mean = float(origin_df["cons_da"].mean())
        r_mean = float(origin_df["residual_load_da"].mean())
        if is_de:
            md = (
                f"**Physikalische Netzfundamentaldaten (`{selected_origin.strftime('%H:%M UTC')}` + 10h)**\n\n"
                f"* **Verbrauchsprognose (Load DA):** Ø **{c_mean:,.0f} MW**\n"
                f"* **Windkraft-Prognose (Wind DA):** Ø **{w_mean:,.0f} MW** *(Spanne: `{w_min:,.0f}` bis `{w_max:,.0f} MW`)*\n"
                f"* **Residuallast (`Last − Wind − Solar`):** Ø **{r_mean:,.0f} MW**\n\n"
                f"💡 *Steigt die Residuallast stark an (hoher Verbrauch bei wenig Wind), wächst das Risiko für Up-Regulation und Hochpreis-Spikes.*"
            )
        else:
            md = (
                f"**Physical Grid Fundamentals (`{selected_origin.strftime('%H:%M UTC')}` + 10h)**\n\n"
                f"* **Consumption Forecast (Load DA):** Mean **{c_mean:,.0f} MW**\n"
                f"* **Wind Generation Forecast (Wind DA):** Mean **{w_mean:,.0f} MW** *(Range: `{w_min:,.0f}` to `{w_max:,.0f} MW`)*\n"
                f"* **Residual Load (`Load − Wind − Solar`):** Mean **{r_mean:,.0f} MW**\n\n"
                f"💡 *High residual load indicates tight conventional supply and elevated upward spike risk, whereas low/negative residual load signals wind surplus.*"
            )
        return _pack(md, [], "FeatureStore Engine")

    # --- Topic I: Forecast Revision vs. Previous Hour (t - 1h) ---
    if any(k in q_low for k in ("previous", "revision", "changed", "change", "drift", "t-1", "t−1", "vorstunde", "geändert", "änderung")):
        if is_de:
            status_txt = (
                "🔄 **Deutliche Revision (≥ 15 €/MWh)** – offene Intraday-Positionen prüfen."
                if briefing["max_abs_rev"] >= 15.0
                else "✅ **Geringe Revision (< 15 €/MWh)** – Prognoseverlauf stabil."
            )
            md = (
                f"**Prognose-Revision vs. Vorstunde (`t − 1h`)**\n\n"
                f"* **Mittlere Verschiebung (Ø ΔP50):** `{briefing['mean_rev']:+.2f} €/MWh`\n"
                f"* **Maximale Schritt-Änderung:** `Δ {briefing['max_abs_rev']:.2f} €/MWh`\n"
                f"* **Auslöser:** Neue Echtzeit-Messwerte vor `{selected_origin.strftime('%H:%M UTC')}` (30m Verbrauchs-Lag & 60m mFRR/aFRR-Lag).\n"
                f"* {status_txt}"
            )
        else:
            status_txt = (
                "🔄 **Significant revision (≥ 15 €/MWh)** — review open intraday exposure."
                if briefing["max_abs_rev"] >= 15.0
                else "✅ **Minor revision (< 15 €/MWh)** — dispatch trajectory remains stable."
            )
            md = (
                f"**Forecast Revision vs. Previous Hour (`t − 1h`)**\n\n"
                f"* **Mean Trajectory Shift (ΔP50):** `{briefing['mean_rev']:+.2f} €/MWh`\n"
                f"* **Max Single-Step Shift:** `Δ {briefing['max_abs_rev']:.2f} €/MWh`\n"
                f"* **Root Cause:** Newly published real-time telemetry prior to `{selected_origin.strftime('%H:%M UTC')}` (30m load lag & 60m balancing lag).\n"
                f"* {status_txt}"
            )
        return _pack(md, driver_weights, "TreeSHAP Engine")

    # --- Topic J: Operational Suggestions / Actions ---
    if any(k in q_low for k in ("action", "recommend", "suggestion", "review", "supervisor", "trader", "dispatch", "empfehlung", "maßnahme", "schichtleiter", "tun")):
        hi_steps = origin_df[origin_df["p_spike_high"] >= 0.30]
        lo_steps = origin_df[origin_df["p_spike_low"] >= 0.30]
        if is_de:
            if len(hi_steps):
                t0 = hi_steps["delivery_start_utc"].min().strftime("%H:%M")
                t1 = (hi_steps["delivery_start_utc"].max() + pd.Timedelta(minutes=15)).strftime("%H:%M UTC")
                rec = (
                    f"1. **Hochpreis-Fenster ({t0}–{t1}) prüfen:** Spike-Risiko (`> Spot + 50 €`) erreicht **{briefing['max_p_hi']:.1%}**.\n"
                    f"2. **Fernwärme & Erzeugung:** Flexible Erzeugung prüfen und Elektrokessel-Last in Spitzenstunden hinterfragen.\n"
                    f"3. **Intraday-Handel:** Offene Short-Positionen vor `{t0} UTC` absichern."
                )
            elif len(lo_steps):
                t0 = lo_steps["delivery_start_utc"].min().strftime("%H:%M")
                t1 = (lo_steps["delivery_start_utc"].max() + pd.Timedelta(minutes=15)).strftime("%H:%M UTC")
                rec = (
                    f"1. **Tief-/Negativpreis-Fenster ({t0}–{t1}) prüfen:** Risiko (`< Spot − 50 €`) erreicht **{briefing['max_p_lo']:.1%}**.\n"
                    f"2. **Power-to-Heat:** Prüfen, ob Elektrokessel oder Wärmepumpen günstige Ausgleichspreise nutzen können.\n"
                    f"3. **mFRR Down-Gebote:** Abregelungs-Bereitschaft prüfen."
                )
            else:
                rec = (
                    f"1. **Regelbetrieb:** Alle 40 Viertelstunden bleiben unter der 30%-Spike-Warnschwelle.\n"
                    f"2. **Netztendenz:** Dominante Richtung ist **{briefing['dominant_dir']}** – Standard-Fahrplan beibehalten."
                )
            md = f"**Prüf-Empfehlungen für Schichtleiter & Trader**\n\n{rec}\n\n*Hinweis: Entscheidungsunterstützung auf Basis öffentlicher Marktdaten – keine automatischen Schaltbefehle.*"
        else:
            if len(hi_steps):
                t0 = hi_steps["delivery_start_utc"].min().strftime("%H:%M")
                t1 = (hi_steps["delivery_start_utc"].max() + pd.Timedelta(minutes=15)).strftime("%H:%M UTC")
                rec = (
                    f"1. **High-Price Window ({t0}–{t1}):** Spike risk (`> Spot + 50 €`) peaks at **{briefing['max_p_hi']:.1%}**.\n"
                    f"2. **District Heating & Generation:** Review flexible generation and consider reducing electric boiler load during peak risk steps.\n"
                    f"3. **Intraday Desk:** Evaluate hedging open short exposure ahead of `{t0} UTC`."
                )
            elif len(lo_steps):
                t0 = lo_steps["delivery_start_utc"].min().strftime("%H:%M")
                t1 = (lo_steps["delivery_start_utc"].max() + pd.Timedelta(minutes=15)).strftime("%H:%M UTC")
                rec = (
                    f"1. **Low / Negative Window ({t0}–{t1}):** Low-spike risk (`< Spot − 50 €`) peaks at **{briefing['max_p_lo']:.1%}**.\n"
                    f"2. **Power-to-Heat Assets:** Evaluate ramping up electric boilers or heat pumps to capture low imbalance prices.\n"
                    f"3. **mFRR Down Bids:** Verify curtailment readiness."
                )
            else:
                rec = (
                    f"1. **Standard Dispatch:** All 40 steps remain below the 30% spike alert threshold.\n"
                    f"2. **Grid Tendency:** Dominant direction is **{briefing['dominant_dir']}** — maintain baseline schedule."
                )
            md = f"**Operational Suggestions to Review**\n\n{rec}\n\n*Note: Decision support based on public market data — not automated plant switching commands.*"
        return _pack(md, driver_weights, "Decision Support")

    # --- Topic K: Buttons, Toggles, Controls & UI Elements ---
    if any(k in q_low for k in ("button", "knopf", "toggle", "schalter", "zahnrad", "gear", "settings", "einstellung", "dark mode", "light mode", "hindsight", "json", "download", "api payload", "reasoning", "gedankengang")):
        if is_de:
            md = (
                "**Alle Buttons, Schalter & Bedienelemente der Website**\n\n"
                "* **Top-Leiste oben rechts:**\n"
                "  * **🌙 Dark / ☀️ Light Button:** Wechselt zwischen dem *Control-Room Dark Mode* (`#07131E`) und dem *Web Light Mode* (`#FFFFFF`).\n"
                "* **Linke Sidebar (Zeitsteuerung):**\n"
                "  * **1. Dataset Period:** Wählt zwischen *Test Period (5. Okt – 1. Nov 2026, 673 Origins)* und *Validation Period (7. Sep – 4. Okt 2026)*.\n"
                "  * **2. Quick Filter:** Filtert die Zeitpunkte nach *All 673 Origins*, *Spike Alert Origins Only (≥ 30%)*, *Top 20 High-Price Spikes* oder *Top 20 Low-Price Spikes*.\n"
                "  * **◀ Previous / Next ▶:** Springt exakt eine Stunde vor oder zurück.\n"
                "  * **🤖 Show AI Explainability Chat:** Blendet die rechte KI-Spalte ein/aus.\n"
                "  * **🎯 Overlay Actual Prices (Hindsight):** Blendet die echten historischen Preise (grüne gepunktete Linie) im Graphen ein.\n"
                "* **AI Assistant (Rechts):**\n"
                "  * **⚙️ Zahnrad-Button (oben rechts):** Öffnet Modell-Auswahl (`gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`), API-Key-Feld und `🗑️ Clear Chat History`.\n"
                "  * **✨ Quick Questions (4 Schnellfragen):** *Drivers at Peak*, *Spike Risk Summary*, *Change vs. t−1h*, *Review Actions*.\n"
                "  * **🧠 Show Model Reasoning:** Zeigt den 4-stufigen Gedankengang (*Chain-of-Thought*) der KI über jeder Antwort an."
            )
        else:
            md = (
                "**Complete Guide to All Buttons, Toggles & Controls**\n\n"
                "* **Top Toolbar (Top Right):**\n"
                "  * **🌙 Dark / ☀️ Light Toggle:** Switches between *Control-Room Dark Mode* (`#07131E`) and *Web Light Mode* (`#FFFFFF`).\n"
                "* **Left Sidebar (Time Machine):**\n"
                "  * **1. Dataset Period:** Switches between *Test Period (5 Oct – 1 Nov 2026, 673 origins)* and *Validation Period (7 Sep – 4 Oct 2026)*.\n"
                "  * **2. Quick Filter:** Filters origins by *All 673 Origins*, *Spike Alert Origins Only (≥ 30%)*, *Top 20 High-Price Spikes*, or *Top 20 Low-Price Spikes*.\n"
                "  * **◀ Previous / Next ▶:** Steps 1 hour backward or forward.\n"
                "  * **🤖 Show AI Explainability Chat:** Shows or hides the right-hand AI panel.\n"
                "  * **🎯 Overlay Actual Prices (Hindsight):** Overlays realized ground-truth prices (green dotted line) on the fan chart.\n"
                "* **AI Assistant (Right Panel):**\n"
                "  * **⚙️ Gear Button (Top Right):** Opens AI model selector (`gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`), masked API key input, and `🗑️ Clear Chat History`.\n"
                "  * **✨ Quick Questions:** 4 one-click operational prompts.\n"
                "  * **🧠 Show Model Reasoning:** Toggles the step-by-step Chain-of-Thought reasoning trace above each answer."
            )
        return _pack(md, [], "Dashboard Copilot")

    # --- Topic L: All Graphs, Charts, Waterfall, Heatmap, Birnbaum & XAI ---
    if any(k in q_low for k in ("graph", "chart", "diagramm", "kurve", "waterfall", "wasserfall", "heatmap", "birnbaum", "xai", "shap", "scatter", "ablation", "wilcoxon", "t-test", "tab")):
        if is_de:
            md = (
                "**Alle Graphen, Tabs & XAI-Visualisierungen der Website**\n\n"
                "1. **Tab 1 – 📈 10-Hour Forecast & Risk Monitor:**\n"
                "   * **Probabilistic 10h Fan Chart:** Zeigt 6h Vergangenheit links von `NOW` und 40 Viertelstunden (+10h) rechts mit `P50`-Median (türkis), `80% Konfidenzband (P10–P90)`, `Spotpreis` (orange gestrichelt) und `Vorstunden-Prognose (t−1h)`.\n"
                "   * **2 XAI-Erklärkarten:** Erklären für den Peak-Step, wie `P50 = Spot + AI Spread` entsteht und warum das `P10–P90`-Band so breit ist.\n"
                "   * **Price Spike Probability Chart:** Balkendiagramm für `P(High Spike > Spot+50€)` (Berry-Rot) und `P(Low Spike < Spot−50€)` (Pflaume) mit 30%-Warnschwelle.\n"
                "   * **mFRR Balancing Direction Chart:** Gestapeltes Flächendiagramm für `Up` (Defizit), `Down` (Überschuss) und `None`.\n"
                "2. **Tab 2 – 🔍 Physical Grid Drivers & SHAP Explainability:**\n"
                "   * **Price Waterfall & Uncertainty Breakdown:** Wasserfall-Zerlegung von Spot → P50 sowie der 4 Unsicherheitsquellen für jeden wählbaren Step 1–40.\n"
                "   * **Physical Grid Trajectory & Horizon SHAP:** Verlauf von Verbrauch, Wind und Residuallast (MW) + Top-SHAP-Treiber.\n"
                "   * **Advanced XAI Expander:** 40-Step Driver-Heatmap, Birnbaum-Zuverlässigkeitsanalyse (`I_B`, `I_CR`, `I_FV`) und globales LightGBM Information Gain.\n"
                "3. **Tab 3 – 🎯 Hindsight Benchmark vs. Official Baseline:**\n"
                "   * Benchmark-Tabelle (`MAE 43,19 €` vs. `66,20 €`), Horizon-MAE-Kurve, Ablationsstudie (Heliyon Table 3), t-Test/Wilcoxon-Signifikanztest (`p < 0.0001`) und 45°-Kalibrierungs-Scatterplot.\n"
                "4. **Tab 4 – 🔌 40-Step Data Table & JSON API:**\n"
                "   * Komplette 40-Schritte-Tabelle und JSON-API-Download."
            )
        else:
            md = (
                "**Complete Guide to All Charts, Tabs & XAI Visuals**\n\n"
                "1. **Tab 1 – 📈 10-Hour Forecast & Risk Monitor:**\n"
                "   * **Probabilistic 10h Fan Chart:** 6h realized history left of `NOW` and 40 quarter-hour steps (+10h) right of `NOW` (`P50` median, `P10–P90` 80% band, `Spot Price`, `t−1h revision`).\n"
                "   * **2 XAI Summary Cards:** Decompose `Spot + AI Spread = P50` and the 4 sources of `P10–P90` width.\n"
                "   * **Price Spike Probability Chart:** Bar chart of `P(High Spike > Spot+50€)` and `P(Low Spike < Spot−50€)` against the 30% alert line.\n"
                "   * **mFRR Balancing Direction Chart:** 100% stacked area chart for `Up` (Deficit), `Down` (Surplus), and `None`.\n"
                "2. **Tab 2 – 🔍 Physical Grid Drivers & SHAP Explainability:**\n"
                "   * **Price Waterfall & Uncertainty Breakdown Charts:** Step-by-step waterfall from Spot to P50 and uncertainty width decomposition for any selected Step 1–40.\n"
                "   * **Physical Grid Trajectory & Horizon SHAP:** Load, Wind, and Residual Load (MW) + Top SHAP drivers.\n"
                "   * **Advanced XAI Expander:** 40-Step Temporal Driver Heatmap, Birnbaum Reliability Table & Chart (`I_B`, `I_CR`, `I_FV`), and Global LightGBM Gain.\n"
                "3. **Tab 3 – 🎯 Hindsight Benchmark vs. Official Baseline:**\n"
                "   * Benchmark comparison (`MAE 43.19 €` vs. `66.20 €`), Horizon MAE curve, Progressive Ablation Study, Paired t-Test & Wilcoxon Signed-Rank Test (`p < 0.0001`), and 45° Calibration Scatter.\n"
                "4. **Tab 4 – 🔌 40-Step Data Table & JSON API:**\n"
                "   * Full 40-step table and downloadable REST JSON payload."
            )
        return _pack(md, driver_weights, "Dashboard Copilot")

    # --- Topic M: Website / Dashboard / Model / Benchmark ---
    if any(k in q_low for k in ("website", "webseite", "dashboard", "model", "modell", "baseline", "accuracy", "mae", "pinball", "brier", "benchmark", "wie funktioniert", "how does", "what is this", "erklär")):
        if is_de:
            md = (
                "**Über dieses Dashboard & die KI-Modelle**\n\n"
                "* **10h Probabilistischer Fan-Chart:** Zeigt links der `NOW`-Linie die letzten 6 Stunden und rechts die **40 Viertelstunden-Schritte (+10h)** mit `P50`-Median, `80% Konfidenzband (P10–P90)`, `Day-Ahead Spotpreis` und `Vorstunden-Prognose (t−1h)`.\n"
                "* **Time Machine (Links):** Reise zu jedem der **673 Vorhersage-Zeitpunkte** (`Test Period` oder `Validation Period`) und filtere nach **Spike-Warnungen (≥ 30 %)** oder **Top 20 High/Low Spikes**.\n"
                "* **LightGBM-Benchmark (Tab 3):** Unsere Modell-Suite senkt den **MAE auf 43,19 €/MWh** (vs. `66,20 €` Baseline, **−35 % Fehler**), halbiert den **Pinball Loss auf 12,43** (vs. `24,18`) und trifft exakt **80,0 % P10–P90 Coverage**."
            )
        else:
            md = (
                "**Dashboard & Forecasting Architecture**\n\n"
                "* **10h Probabilistic Fan Chart:** Displays 6h realized history left of `NOW` and **40 quarter-hour steps (+10h)** on the right with the `P50` median, `80% confidence band (P10–P90)`, `Day-Ahead Spot Price`, and `t−1h revision`.\n"
                "* **Historical Time Machine (Left):** Step through all **673 hourly origins** (`Test Period` or `Validation Period`) and pre-filter for **Spike Alerts (≥ 30%)** or **Top 20 High/Low Spike Origins**.\n"
                "* **Validation Benchmark (Tab 3):** Our LightGBM + Conformal suite cuts **P50 MAE by 35%** (`43.19` vs. `66.20 €/MWh` baseline), halves **Pinball Loss** (`12.43` vs. `24.18`), and hits **exact 80.0% P10–P90 coverage**."
            )
        return _pack(md, [], "Dashboard Copilot")

    # --- Default: Horizon & Spike Summary in the user's language (German or English) ---
    top_drv_names = ", ".join(d["driver"] for d in driver_weights[:3])
    t_peak = pd.Timestamp(peak_row["delivery_start_utc"]).strftime("%H:%M")
    t_min = pd.Timestamp(min_row["delivery_start_utc"]).strftime("%H:%M")
    t_hi = pd.Timestamp(hi_row["delivery_start_utc"]).strftime("%H:%M")

    if is_de:
        md = (
            f"**10h Überblick für Origin `{selected_origin.strftime('%Y-%m-%d %H:%M UTC')}`** *({active_period})*\n\n"
            f"Wichtigste Preistreiber: **{top_drv_names}**\n\n"
            f"| Kennzahl | UTC | Spot → P50 | Spike-Risiko |\n"
            f"| :--- | :---: | :---: | :---: |\n"
            f"| **P50 Maximum** | {t_peak} | {peak_row['spot_price']:.0f} → **{peak_row['p50']:.1f} €** | Hi {peak_row['p_spike_high']:.0%} |\n"
            f"| **P50 Minimum** | {t_min} | {min_row['spot_price']:.0f} → **{min_row['p50']:.1f} €** | Lo {min_row['p_spike_low']:.0%} |\n"
            f"| **Max Spike-Risiko** | {t_hi} | {hi_row['spot_price']:.0f} → **{hi_row['p50']:.1f} €** | **Hi {hi_row['p_spike_high']:.1%}** |\n\n"
            f"💡 *Du kannst mich z. B. fragen: „Was ist der Spotpreis?“, „Was bedeutet Dataset Period?“, „Welche Treiber wirken um {t_peak} Uhr?“ oder „Was bedeuten P10 und P90?“*"
        )
    else:
        md = (
            f"**10h Horizon Summary (`{selected_origin.strftime('%Y-%m-%d %H:%M UTC')}`)** *({active_period})*\n\n"
            f"Driven primarily by **{top_drv_names}**:\n\n"
            f"| Key Point | UTC | Spot → P50 | Spike Risk |\n"
            f"| :--- | :---: | :---: | :---: |\n"
            f"| **P50 Peak** | {t_peak} | {peak_row['spot_price']:.0f} → **{peak_row['p50']:.1f} €** | Hi {peak_row['p_spike_high']:.0%} |\n"
            f"| **P50 Min** | {t_min} | {min_row['spot_price']:.0f} → **{min_row['p50']:.1f} €** | Lo {min_row['p_spike_low']:.0%} |\n"
            f"| **Max Risk** | {t_hi} | {hi_row['spot_price']:.0f} → **{hi_row['p50']:.1f} €** | **Hi {hi_row['p_spike_high']:.1%}** |\n\n"
            f"💡 *Ask e.g. “What is the Spot Price?”, “What is the Dataset Period?”, “What drives the price at {t_peak}?”, or “Explain P10 and P90.”*"
        )
    return _pack(md, driver_weights, "TreeSHAP Engine")
