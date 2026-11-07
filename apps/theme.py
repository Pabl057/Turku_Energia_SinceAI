"""Turku Energia Brand Design System & UI Components (Graafinen ohjeisto, Syyskuu 2024).

Implements strict Corporate Identity rules from `data/Graafinen ohjeisto.pdf` in both
Control-Room Dark Mode (default) and Clean Web Light Mode:
- Primary Brand Orange: #F59B00 (RGB 245, 155, 0) — NEVER pair with black text on orange!
- Dark Mode Canvas: Deep Nordic Control-Room Navy (#07131E) with elevated #0D1E2E cards
  and #1E3F5E borders derived from Turku Energia's Infographic Navy (#0F4B73, Page 14).
- Light Mode Canvas: #FFFFFF (White) with UI-Kit Bg Light #EAF7FB for box modules (Page 48).
- Secondary Palette (Page 13): Ice Blue #B4DCE6, Turquoise #46BEDC, Lime #AAC800, Berry #E6508C.
- Infographic Palette (Page 14): Deep Navy #0F4B73, Plum #A03C8C, Warm Yellow #FFD719,
  Warm Stone #E6E1DC, Neutral Gray #7D7D7D.
- Typography (Pages 8–11, 48): Fira Sans Extra Condensed (headings, ending with a period '.'),
  Fira Sans (body/UI), Newsreader / Century Schoolbook (serif price highlights & brand slogan).
"""

from __future__ import annotations

import html

# Official Color Tokens from Graafinen ohjeisto.pdf (Pages 12, 13, 14, 48)
TE_COLORS = {
    "orange": "#F59B00",         # Primary Brand Color (Page 12)
    "orange_light": "#FFF4E0",   # -90% tint for subtle warm highlights
    "orange_soft": "#FDE6BE",    # Soft border tint
    "white": "#FFFFFF",          # Main Web Background (Page 48)
    "bg_light": "#EAF7FB",       # UI-Kit Brand Blue 50 "Bg Light" (Page 48)
    "bg_page": "#F8FBFD",        # Ultra-clean subtle page canvas
    "dark_bg": "#07131E",        # Deep Control-Room Dark Canvas
    "dark_card": "#0D1E2E",      # Elevated Dark Card Surface
    "dark_card_alt": "#12283C",  # Secondary Dark Surface / Hover
    "dark_border": "#1E3F5E",    # Crisp Dark Border
    "ice_blue": "#B4DCE6",       # Secondary Palette (Page 13)
    "turquoise": "#46BEDC",      # Secondary Palette (Page 13)
    "lime": "#AAC800",           # Secondary Palette (Page 13)
    "berry": "#E6508C",          # Secondary Palette (Page 13)
    "navy": "#0F4B73",           # Infographic Palette Deep Blue (Page 14)
    "plum": "#A03C8C",           # Infographic Palette Purple (Page 14)
    "yellow": "#FFD719",         # Infographic Palette Yellow (Page 14)
    "stone": "#E6E1DC",          # Infographic Palette Warm Gray (Page 14)
    "gray": "#7D7D7D",           # Infographic Palette Neutral Gray (Page 14)
    "text_dark": "#0B2942",      # High-contrast AA accessible text (Light Mode)
    "text_light": "#F0F6FA",     # High-contrast AAA accessible text (Dark Mode)
}

# Human-friendly English labels for technical SHAP feature names
FRIENDLY_DRIVER_NAMES: dict[str, str] = {
    "day_ahead_spot_price": "Day-Ahead Spot Price (Nord Pool)",
    "spot_at_delivery": "Day-Ahead Spot Price (Nord Pool)",
    "spot_price": "Day-Ahead Spot Price (Nord Pool)",
    "spot_price_eur_mwh": "Day-Ahead Spot Price (Nord Pool)",
    "residual_load_forecast": "Residual Load Forecast (Load − Wind − Solar)",
    "residual_load_da_at_delivery": "Residual Load Forecast (MW)",
    "residual_load_da": "Residual Load Forecast (MW)",
    "net_load_deficit_mw": "Net Load Deficit (MW)",
    "consumption_forecast": "Consumption Forecast (Fingrid DA)",
    "cons_da_at_delivery": "Consumption Forecast (Fingrid DA)",
    "cons_da": "Consumption Forecast (Fingrid DA)",
    "wind_power_forecast": "Wind Generation Forecast (Fingrid DA)",
    "wind_da_at_delivery": "Wind Generation Forecast (Fingrid DA)",
    "wind_da": "Wind Generation Forecast (Fingrid DA)",
    "wind_forecast_mw": "Wind Generation Forecast (MW)",
    "wind_forecast_3h_ramp": "3h Wind Power Ramp (Weather Front)",
    "wind_da_ramp_3h": "3h Wind Power Ramp (MW)",
    "wind_share_da_at_delivery": "Wind Share of Total Load (%)",
    "solar_power_forecast": "Solar Generation Forecast (Fingrid DA)",
    "solar_da_at_delivery": "Solar Generation Forecast (Fingrid DA)",
    "renewable_da_at_delivery": "Renewable Generation (Wind + Solar)",
    "recent_imbalance_spread": "Recent Imbalance Spread (t − 60m)",
    "last_imbalance_spread": "Latest Imbalance Spread (t − 60m)",
    "imbalance_spread_lag4": "Imbalance Spread Lag (t − 60m)",
    "recent_imbalance_price": "Latest Imbalance Price (t − 60m)",
    "last_imbalance_price": "Latest Imbalance Price (t − 60m)",
    "imb_price_min_96q": "24h Rolling Minimum Price",
    "imb_spread_min_96q": "24h Rolling Minimum Spread",
    "imb_spread_max_96q": "24h Rolling Peak Spread",
    "imb_spread_std_96q": "24h Price Volatility (Grid Stress)",
    "recent_mfrr_direction": "Recent mFRR Balancing Direction",
    "last_mfrr_direction": "Latest mFRR Balancing Direction",
    "recent_mfrr_volume": "Activated mFRR Balancing Volume (MW)",
    "recent_afrr_price": "Recent aFRR Capacity Price",
    "consumption_forecast_error": "Real-Time Load Forecast Error (t − 30m)",
    "cons_forecast_error_30m": "Real-Time Load Deviation (t − 30m)",
    "cross_border_flow": "Cross-Border Flows (SE1 / SE3 / NO4 / EE)",
    "neighbor_imbalance": "Neighboring Grid Imbalance (Sweden / Estonia)",
    "diurnal_calendar_pattern": "Diurnal & Weekend Load Profile",
    "forecast_horizon_step": "Forecast Horizon Lead Time (Step 1–40)",
}


def format_driver_name(raw_name: str) -> str:
    """Convert raw feature or driver code into a clear English domain label."""
    clean = raw_name.strip()
    # Strip optional trailing parenthetical SHAP score if present, e.g. "spot_price(+12.4)"
    base_key = clean.split("(")[0].strip()
    suffix = ""
    if "(" in clean and clean.endswith(")"):
        suffix = " (" + clean.split("(", 1)[1]
    if base_key in FRIENDLY_DRIVER_NAMES:
        return FRIENDLY_DRIVER_NAMES[base_key] + suffix
    underscored = base_key.replace(" ", "_").lower()
    if underscored in FRIENDLY_DRIVER_NAMES:
        return FRIENDLY_DRIVER_NAMES[underscored] + suffix
    return base_key.replace("_", " ").title() + suffix


def get_turku_energia_logo_svg(width: int = 136) -> str:
    """Return the official 8:5 ratio Turku Energia orange logo block (Pages 4-6).

    Uses #F59B00 background and pure #FFFFFF custom bold condensed lettering.
    Never uses black text on orange (Page 9 rule).
    """
    height = int(round(width * 5 / 8))
    return f"""
    <svg width="{width}" height="{height}" viewBox="0 0 160 100" fill="none" xmlns="http://www.w3.org/2000/svg"
         style="display:block; flex-shrink:0; border-radius:3px; box-shadow: 0 4px 14px rgba(245, 155, 0, 0.35);">
      <rect width="160" height="100" fill="#F59B00"/>
      <text x="18" y="45" fill="#FFFFFF"
            font-family="'Fira Sans Extra Condensed', 'Franklin Gothic Medium', 'Arial Narrow', sans-serif"
            font-weight="800" font-size="34" letter-spacing="0.8">TURKU</text>
      <text x="18" y="81" fill="#FFFFFF"
            font-family="'Fira Sans Extra Condensed', 'Franklin Gothic Medium', 'Arial Narrow', sans-serif"
            font-weight="800" font-size="34" letter-spacing="0.5">ENERGIA</text>
    </svg>
    """.strip()


def inject_brand_css(dark_mode: bool = True) -> str:
    """Return full CSS stylesheet implementing Graafinen ohjeisto.pdf in Dark or Light mode."""
    if dark_mode:
        theme_vars = """
        --te-orange: #F59B00;
        --te-orange-hover: #FFA81A;
        --te-orange-light: rgba(245, 155, 0, 0.16);
        --te-orange-soft: rgba(245, 155, 0, 0.38);
        --te-bg-app: #07131E;
        --te-bg-surface: #0D1E2E;
        --te-bg-elevated: #12283C;
        --te-bg-sidebar: linear-gradient(180deg, #0A1927 0%, #07131E 100%);
        --te-bg-hero: linear-gradient(135deg, #0D2235 0%, #0A1928 58%, #1F1D18 100%);
        --te-border: #1E3F5E;
        --te-border-subtle: #152F47;
        --te-ice: #B4DCE6;
        --te-turquoise: #46BEDC;
        --te-lime: #AAC800;
        --te-berry: #E6508C;
        --te-navy: #0F4B73;
        --te-plum: #C85AB2;
        --te-yellow: #FFD719;
        --te-stone: #233B52;
        --te-gray: #92AFC4;
        --te-heading-color: #FFFFFF;
        --te-text: #F0F6FA;
        --te-text-muted: #9CB8CE;
        --te-kpi-value-color: #FFFFFF;
        --te-delta-bg: rgba(70, 190, 220, 0.14);
        --te-delta-text: #CFEAF2;
        --te-delta-ok-bg: rgba(170, 200, 0, 0.18);
        --te-delta-ok-text: #D4F04D;
        --te-alert-high-bg: linear-gradient(90deg, rgba(230, 80, 140, 0.18) 0%, rgba(13, 30, 46, 0.95) 100%);
        --te-alert-high-border: rgba(230, 80, 140, 0.45);
        --te-alert-low-bg: linear-gradient(90deg, rgba(160, 60, 140, 0.22) 0%, rgba(13, 30, 46, 0.95) 100%);
        --te-alert-low-border: rgba(200, 90, 178, 0.45);
        --te-alert-rev-bg: linear-gradient(90deg, rgba(245, 155, 0, 0.18) 0%, rgba(13, 30, 46, 0.95) 100%);
        --te-alert-rev-border: rgba(245, 155, 0, 0.45);
        --te-alert-norm-bg: linear-gradient(90deg, rgba(170, 200, 0, 0.15) 0%, rgba(13, 30, 46, 0.95) 100%);
        --te-alert-norm-border: rgba(170, 200, 0, 0.42);
        --te-msg-user-bg: linear-gradient(135deg, #0F4B73 0%, #145D8D 100%);
        --te-msg-ai-bg: #102437;
        --te-bar-track: #17324A;
        --te-shadow-sm: 0 4px 16px rgba(0, 0, 0, 0.35);
        --te-shadow-md: 0 10px 28px rgba(0, 0, 0, 0.5);
        """
    else:
        theme_vars = """
        --te-orange: #F59B00;
        --te-orange-hover: #DF8C00;
        --te-orange-light: #FFF4E0;
        --te-orange-soft: #FDE6BE;
        --te-bg-app: #FFFFFF;
        --te-bg-surface: #EAF7FB;
        --te-bg-elevated: #FFFFFF;
        --te-bg-sidebar: linear-gradient(180deg, #EAF7FB 0%, #FFFFFF 100%);
        --te-bg-hero: linear-gradient(135deg, #EAF7FB 0%, #FFFFFF 62%, #FFF8EB 100%);
        --te-border: #B4DCE6;
        --te-border-subtle: #D5ECF2;
        --te-ice: #B4DCE6;
        --te-turquoise: #46BEDC;
        --te-lime: #AAC800;
        --te-berry: #E6508C;
        --te-navy: #0F4B73;
        --te-plum: #A03C8C;
        --te-yellow: #FFD719;
        --te-stone: #E6E1DC;
        --te-gray: #5A7184;
        --te-heading-color: #0F4B73;
        --te-text: #0F2942;
        --te-text-muted: #3B5975;
        --te-kpi-value-color: #0F4B73;
        --te-delta-bg: rgba(255, 255, 255, 0.85);
        --te-delta-text: #2B4C66;
        --te-delta-ok-bg: #EBF5D6;
        --te-delta-ok-text: #3D5200;
        --te-alert-high-bg: #FFF5F8;
        --te-alert-high-border: #F5C2D7;
        --te-alert-low-bg: #FAF5F9;
        --te-alert-low-border: #E2C2DA;
        --te-alert-rev-bg: #FFF9EB;
        --te-alert-rev-border: #FDE6BE;
        --te-alert-norm-bg: #F4FAE8;
        --te-alert-norm-border: #D7E8B5;
        --te-msg-user-bg: #0F4B73;
        --te-msg-ai-bg: #EAF7FB;
        --te-bar-track: #D5ECF2;
        --te-shadow-sm: 0 2px 10px rgba(15, 75, 115, 0.06);
        --te-shadow-md: 0 8px 24px rgba(15, 75, 115, 0.11);
        """

    return f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fira+Sans+Extra+Condensed:wght@600;700;800&family=Fira+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

      :root {{
        {theme_vars}
        --te-radius-card: 12px 26px 12px 12px;
      }}

      /* Global page & typography */
      .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background-color: var(--te-bg-app) !important;
        color: var(--te-text) !important;
        font-family: 'Fira Sans', 'Franklin Gothic Book', -apple-system, BlinkMacSystemFont, sans-serif;
      }}

      .block-container {{
        padding-top: 1.1rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
      }}

      h1, h2, h3, h4, .te-heading {{
        font-family: 'Fira Sans Extra Condensed', 'Trade Gothic Bold Condensed', sans-serif !important;
        color: var(--te-heading-color) !important;
        letter-spacing: 0.015em;
      }}

      p, li, label, .stMarkdown, [data-testid="stWidgetLabel"] p {{
        color: var(--te-text) !important;
      }}

      code {{
        background: var(--te-bg-elevated) !important;
        color: var(--te-turquoise) !important;
        border: 1px solid var(--te-border-subtle);
        padding: 0.1rem 0.38rem !important;
        border-radius: 5px !important;
      }}

      /* Sidebar styling (Time Machine) */
      section[data-testid="stSidebar"] {{
        background: var(--te-bg-sidebar) !important;
        border-right: 1px solid var(--te-border) !important;
      }}

      section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.1rem !important;
        padding-left: 1.15rem !important;
        padding-right: 1.15rem !important;
      }}

      /* Native Selectbox / Input / Expander Dark Polish */
      div[data-baseweb="select"] > div,
      div[data-baseweb="input"] > div,
      div[data-baseweb="base-input"] {{
        background-color: var(--te-bg-surface) !important;
        border-color: var(--te-border) !important;
        color: var(--te-text) !important;
        border-radius: 8px !important;
      }}

      div[data-testid="stExpander"] {{
        background-color: var(--te-bg-surface) !important;
        border: 1px solid var(--te-border) !important;
        border-radius: 10px !important;
      }}

      div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: var(--te-bg-surface) !important;
        border-color: var(--te-border) !important;
        border-radius: 12px !important;
      }}

      /* Hero Banner with Organic Wave Shape (Pages 3, 20, 47, 48) */
      .te-hero-banner {{
        position: relative;
        overflow: hidden;
        background: var(--te-bg-hero);
        border: 1px solid var(--te-border);
        border-left: 5px solid var(--te-orange);
        border-radius: 14px 36px 14px 14px;
        padding: 1.3rem 1.75rem;
        margin-bottom: 1.2rem;
        box-shadow: var(--te-shadow-sm);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
      }}

      .te-hero-banner::before {{
        content: "";
        position: absolute;
        right: -50px;
        top: -90px;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(245, 155, 0, 0.16) 0%, rgba(70, 190, 220, 0.06) 50%, transparent 72%);
        border-radius: 50%;
        pointer-events: none;
      }}

      .te-hero-left {{
        display: flex;
        align-items: center;
        gap: 1.35rem;
        z-index: 1;
      }}

      .te-hero-title {{
        font-family: 'Fira Sans Extra Condensed', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        color: var(--te-heading-color) !important;
        line-height: 1.08;
        margin: 0 0 0.28rem 0;
        text-transform: uppercase;
      }}

      .te-hero-title span.te-dot {{
        color: var(--te-orange);
      }}

      .te-hero-subtitle {{
        font-size: 0.93rem;
        color: var(--te-text-muted) !important;
        margin: 0;
        font-weight: 400;
      }}

      .te-slogan-badge {{
        font-family: 'Newsreader', 'Century Schoolbook', Georgia, serif;
        font-size: 0.95rem;
        font-style: italic;
        color: var(--te-heading-color);
        background: var(--te-bg-surface);
        border: 1px solid var(--te-border);
        border-left: 4px solid var(--te-orange);
        padding: 0.55rem 1.05rem;
        border-radius: 8px 18px 8px 8px;
        white-space: nowrap;
        box-shadow: var(--te-shadow-sm);
        z-index: 1;
        text-align: right;
      }}

      .te-slogan-sub {{
        display: block;
        font-family: 'Fira Sans', sans-serif;
        font-style: normal;
        font-size: 0.72rem;
        color: var(--te-text-muted);
        margin-top: 0.12rem;
      }}

      /* KPI Cards — Page 24 & 48 style + Asymmetric Corner */
      .te-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.95rem;
        margin-bottom: 1.15rem;
      }}

      @media (max-width: 1200px) {{
        .te-kpi-grid {{
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
      }}

      .te-kpi-card {{
        background: var(--te-bg-surface);
        border: 1px solid var(--te-border);
        border-top: 4px solid var(--te-turquoise);
        border-radius: var(--te-radius-card);
        padding: 1rem 1.15rem;
        box-shadow: var(--te-shadow-sm);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
      }}

      .te-kpi-card:hover {{
        transform: translateY(-3px);
        box-shadow: var(--te-shadow-md);
        border-color: var(--te-orange);
      }}

      .te-kpi-card.accent-orange {{ border-top-color: var(--te-orange); }}
      .te-kpi-card.accent-berry  {{ border-top-color: var(--te-berry); }}
      .te-kpi-card.accent-plum   {{ border-top-color: var(--te-plum); }}
      .te-kpi-card.accent-turquoise {{ border-top-color: var(--te-turquoise); }}
      .te-kpi-card.accent-lime   {{ border-top-color: var(--te-lime); }}

      .te-kpi-label {{
        font-family: 'Fira Sans Extra Condensed', sans-serif;
        font-size: 0.86rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--te-text-muted);
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}

      .te-kpi-value {{
        font-family: 'Newsreader', 'Century Schoolbook', Georgia, serif;
        font-size: 1.78rem;
        font-weight: 600;
        color: var(--te-kpi-value-color);
        line-height: 1.15;
        margin-bottom: 0.38rem;
      }}

      .te-kpi-delta {{
        font-size: 0.79rem;
        font-weight: 500;
        color: var(--te-delta-text);
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.16rem 0.55rem;
        border-radius: 99px;
        background: var(--te-delta-bg);
      }}

      .te-kpi-delta.warn {{
        background: var(--te-berry);
        color: #FFFFFF !important;
        font-weight: 600;
      }}

      .te-kpi-delta.ok {{
        background: var(--te-delta-ok-bg);
        color: var(--te-delta-ok-text);
        font-weight: 600;
      }}

      /* Situational Alert Banner */
      .te-alert-card {{
        border-radius: 12px 24px 12px 12px;
        padding: 1rem 1.35rem;
        margin-bottom: 0.9rem;
        border: 1px solid var(--te-border);
        border-left-width: 6px;
        box-shadow: var(--te-shadow-sm);
        transition: all 0.2s ease;
      }}

      .te-alert-card.level-HIGH {{
        background: var(--te-alert-high-bg);
        border-color: var(--te-alert-high-border);
        border-left-color: var(--te-berry);
      }}

      .te-alert-card.level-LOW_SPIKE {{
        background: var(--te-alert-low-bg);
        border-color: var(--te-alert-low-border);
        border-left-color: var(--te-plum);
      }}

      .te-alert-card.level-REVISION {{
        background: var(--te-alert-rev-bg);
        border-color: var(--te-alert-rev-border);
        border-left-color: var(--te-orange);
      }}

      .te-alert-card.level-NORMAL {{
        background: var(--te-alert-norm-bg);
        border-color: var(--te-alert-norm-border);
        border-left-color: var(--te-lime);
      }}

      .te-alert-header {{
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin-bottom: 0.35rem;
      }}

      .te-alert-badge {{
        font-family: 'Fira Sans Extra Condensed', sans-serif;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.2rem 0.62rem;
        border-radius: 4px;
        color: #FFFFFF !important;
      }}

      .badge-HIGH {{ background: var(--te-berry); }}
      .badge-LOW_SPIKE {{ background: var(--te-plum); }}
      .badge-REVISION {{ background: var(--te-orange); color: #FFFFFF !important; }}
      .badge-NORMAL {{ background: var(--te-navy); border: 1px solid var(--te-turquoise); }}

      .te-alert-title {{
        font-family: 'Fira Sans Extra Condensed', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--te-heading-color) !important;
        margin: 0;
      }}

      .te-alert-body {{
        font-size: 0.9rem;
        line-height: 1.52;
        color: var(--te-text) !important;
        margin: 0;
      }}

      /* Chart Section Header */
      .te-section-header {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-bottom: 2px solid var(--te-border-subtle);
        padding-bottom: 0.45rem;
        margin-top: 0.35rem;
        margin-bottom: 0.65rem;
      }}

      .te-section-title {{
        font-family: 'Fira Sans Extra Condensed', sans-serif;
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--te-heading-color) !important;
        text-transform: uppercase;
        margin: 0;
      }}

      .te-section-title span {{
        color: var(--te-orange);
      }}

      .te-section-subtitle {{
        font-size: 0.82rem;
        color: var(--te-text-muted);
      }}

      /* AI Explainability Chat Sidebar Panel */
      .te-chat-panel-header {{
        background: linear-gradient(135deg, #0F4B73 0%, #092C46 100%);
        color: #FFFFFF;
        padding: 0.95rem 1.15rem;
        border-radius: 12px 24px 10px 10px;
        margin-bottom: 0.85rem;
        box-shadow: var(--te-shadow-sm);
        border: 1px solid var(--te-border);
        border-bottom: 3px solid var(--te-orange);
      }}

      .te-chat-panel-title {{
        font-family: 'Fira Sans Extra Condensed', sans-serif;
        font-size: 1.22rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        color: #FFFFFF !important;
        margin: 0 0 0.2rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}

      .te-chat-status-pill {{
        font-family: 'Fira Sans', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        background: var(--te-orange);
        color: #FFFFFF !important;
        padding: 0.16rem 0.58rem;
        border-radius: 99px;
        text-transform: uppercase;
      }}

      .te-chat-panel-sub {{
        font-size: 0.8rem;
        color: #D2EEF7 !important;
        margin: 0;
      }}

      /* Differentiated Chat Bubbles */
      .te-msg-user {{
        background: var(--te-msg-user-bg);
        color: #FFFFFF !important;
        padding: 0.75rem 1rem;
        border-radius: 14px 14px 2px 14px;
        margin: 0.5rem 0 0.5rem 1.5rem;
        font-size: 0.88rem;
        line-height: 1.45;
        border: 1px solid rgba(70, 190, 220, 0.35);
        box-shadow: var(--te-shadow-sm);
      }}

      .te-msg-user-label {{
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--te-ice);
        font-weight: 700;
        margin-bottom: 0.2rem;
      }}

      .te-msg-ai {{
        background: var(--te-msg-ai-bg);
        color: var(--te-text);
        border: 1px solid var(--te-border);
        border-left: 4px solid var(--te-orange);
        padding: 0.65rem 0.9rem;
        border-radius: 14px 14px 14px 2px;
        margin: 0.5rem 0.3rem 0.35rem 0;
        font-size: 0.87rem;
        line-height: 1.5;
        box-shadow: var(--te-shadow-sm);
      }}

      .te-msg-ai-label {{
        font-family: 'Fira Sans Extra Condensed', sans-serif;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--te-turquoise);
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 0.4rem;
      }}

      /* Feature Weighting Bars inside Chat / Explainability */
      .te-driver-box {{
        margin-top: 0.45rem;
        padding: 0.65rem 0.85rem;
        background: var(--te-bg-elevated);
        border-radius: 8px;
        border: 1px solid var(--te-border);
      }}

      .te-driver-box-title {{
        font-family: 'Fira Sans Extra Condensed', sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--te-turquoise);
        text-transform: uppercase;
        margin-bottom: 0.35rem;
      }}

      .te-driver-bar-row {{
        margin: 0.4rem 0;
      }}

      .te-driver-bar-top {{
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--te-text);
        margin-bottom: 0.15rem;
      }}

      .te-driver-bar-track {{
        width: 100%;
        height: 7px;
        background: var(--te-bar-track);
        border-radius: 99px;
        overflow: hidden;
      }}

      .te-driver-bar-fill {{
        height: 100%;
        background: linear-gradient(90deg, var(--te-turquoise) 0%, var(--te-orange) 100%);
        border-radius: 99px;
      }}

      /* Buttons & Tabs refinement */
      .stButton > button {{
        background-color: var(--te-bg-surface) !important;
        color: var(--te-text) !important;
        border-radius: 8px !important;
        font-family: 'Fira Sans', sans-serif !important;
        font-weight: 600 !important;
        border: 1px solid var(--te-border) !important;
        transition: all 0.18s ease !important;
      }}

      .stButton > button:hover {{
        border-color: var(--te-orange) !important;
        background-color: var(--te-orange-light) !important;
        color: var(--te-heading-color) !important;
        transform: translateY(-1px);
      }}

      .stButton > button[kind="primary"],
      div[data-testid="stFormSubmitButton"] > button {{
        background-color: var(--te-orange) !important;
        color: #FFFFFF !important; /* Strict Rule: White on Orange, never black! */
        border-color: var(--te-orange) !important;
        font-weight: 700 !important;
      }}

      .stButton > button[kind="primary"]:hover,
      div[data-testid="stFormSubmitButton"] > button:hover {{
        background-color: var(--te-orange-hover) !important;
        color: #FFFFFF !important;
      }}

      button[data-baseweb="tab"] {{
        font-family: 'Fira Sans Extra Condensed', sans-serif !important;
        font-size: 1.04rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        color: var(--te-text-muted) !important;
      }}

      button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--te-orange) !important;
      }}

      /* Anti-Leakage Gate Box */
      .te-gate-box {{
        background: var(--te-bg-surface);
        border: 1px solid var(--te-border);
        border-left: 4px solid var(--te-lime);
        border-radius: 10px;
        padding: 0.8rem 0.95rem;
        font-size: 0.8rem;
        color: var(--te-text);
        box-shadow: var(--te-shadow-sm);
      }}
    </style>
    """


def render_hero_header(origin_str: str, period_label: str) -> str:
    """Render the Turku Energia brand header with official 8:5 logo and slogan."""
    logo_svg = get_turku_energia_logo_svg(width=120)
    return f"""
    <div class="te-hero-banner">
      <div class="te-hero-left">
        {logo_svg}
        <div>
          <h1 class="te-hero-title">Imbalance Price Situational Desk<span class="te-dot">.</span></h1>
          <p class="te-hero-subtitle">
            Decision Support for Shift Supervisors &amp; Power Traders ·
            <strong>{html.escape(period_label)}</strong> · Origin <code>{html.escape(origin_str)}</code>
          </p>
        </div>
      </div>
      <div class="te-slogan-badge">
        “Olemme lähellä, näemme kauas.”
        <span class="te-slogan-sub">Close to you, seeing far ahead.</span>
      </div>
    </div>
    """


def render_kpi_strip(
    p50_mean: float,
    spot_mean: float,
    max_p_hi: float,
    n_hi_steps: int,
    max_p_lo: float,
    n_lo_steps: int,
    dominant_dir_short: str,
    mean_p_up: float,
    mean_p_down: float,
    mean_band: float,
    max_abs_rev: float,
) -> str:
    """Render the 5 KPI cards in Turku Energia Corporate Identity style."""
    spread_mean = p50_mean - spot_mean
    spread_sign = f"{spread_mean:+.2f} €/MWh vs. Spot"

    hi_accent = "accent-berry" if n_hi_steps > 0 else "accent-turquoise"
    hi_delta_cls = "warn" if n_hi_steps > 0 else "ok"
    hi_delta_txt = f"🚨 {n_hi_steps} QH steps ≥ 30%" if n_hi_steps > 0 else "✓ Below 30% alert threshold"

    lo_accent = "accent-plum" if n_lo_steps > 0 else "accent-turquoise"
    lo_delta_cls = "warn" if n_lo_steps > 0 else "ok"
    lo_delta_txt = f"⚠️ {n_lo_steps} QH steps ≥ 30%" if n_lo_steps > 0 else "✓ Below 30% alert threshold"

    dir_label = "↑ UP (Deficit)" if dominant_dir_short == "UP" else "↓ DOWN (Surplus)"

    return f"""
    <div class="te-kpi-grid">
      <div class="te-kpi-card accent-orange">
        <div class="te-kpi-label"><span>10h Forecast Mean (P50)</span><span>€/MWh</span></div>
        <div class="te-kpi-value">{p50_mean:.2f} €</div>
        <div class="te-kpi-delta">{html.escape(spread_sign)} (Spot Avg {spot_mean:.1f} €)</div>
      </div>

      <div class="te-kpi-card {hi_accent}">
        <div class="te-kpi-label"><span>High-Price Spike (&gt;Spot+50€)</span><span>Max P</span></div>
        <div class="te-kpi-value">{max_p_hi:.1%}</div>
        <div class="te-kpi-delta {hi_delta_cls}">{html.escape(hi_delta_txt)}</div>
      </div>

      <div class="te-kpi-card {lo_accent}">
        <div class="te-kpi-label"><span>Low-Price Spike (&lt;Spot−50€)</span><span>Max P</span></div>
        <div class="te-kpi-value">{max_p_lo:.1%}</div>
        <div class="te-kpi-delta {lo_delta_cls}">{html.escape(lo_delta_txt)}</div>
      </div>

      <div class="te-kpi-card accent-turquoise">
        <div class="te-kpi-label"><span>Dominant Balancing Dir.</span><span>mFRR</span></div>
        <div class="te-kpi-value" style="font-size:1.42rem;">{html.escape(dir_label)}</div>
        <div class="te-kpi-delta">P(Up) {mean_p_up:.0%} · P(Down) {mean_p_down:.0%}</div>
      </div>

      <div class="te-kpi-card accent-lime">
        <div class="te-kpi-label"><span>80% Uncertainty Band</span><span>P10–P90</span></div>
        <div class="te-kpi-value">{mean_band:.1f} €</div>
        <div class="te-kpi-delta">Max Revision vs. t−1h: Δ {max_abs_rev:.1f} €</div>
      </div>
    </div>
    """


def render_alert_cards(alerts: list[dict[str, str]]) -> str:
    """Render structured situational awareness alerts following Tone of Voice rules."""
    badge_labels = {
        "HIGH": "High-Price Spike Alert (>Spot+50€)",
        "LOW_SPIKE": "Low / Negative Price Alert (<Spot−50€)",
        "REVISION": "Forecast Revision Alert (vs. t−1h)",
        "NORMAL": "Balanced Grid Outlook",
    }
    cards_html = []
    for alert in alerts:
        lvl = alert.get("level", "NORMAL")
        badge_txt = badge_labels.get(lvl, lvl)
        title = html.escape(alert.get("title", ""))
        body = html.escape(alert.get("body", ""))
        cards_html.append(
            f"""
            <div class="te-alert-card level-{lvl}">
              <div class="te-alert-header">
                <span class="te-alert-badge badge-{lvl}">{badge_txt}</span>
                <h4 class="te-alert-title">{title}.</h4>
              </div>
              <p class="te-alert-body">{body}</p>
            </div>
            """
        )
    return "\n".join(cards_html)
