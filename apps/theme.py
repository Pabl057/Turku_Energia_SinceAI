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
    "fourier_12h_bimodal_harmonic": "Fourier 12h Bimodal Peak Harmonic (Eq. 2)",
    "fourier_8h_intraday_harmonic": "Fourier 8h Intra-Day Harmonic (Eq. 2)",
    "volume_pricing_mfrr_weight": "Implied mFRR Volume Weight (Post-June Pricing)",
    "volume_pricing_mfrr_weight_1h": "1h Mean mFRR Volume Weight",
    "implied_afrr_activated_volume": "Implied aFRR Activated Volume (MW)",
    "afrr_mfrr_up_price_gap": "aFRR vs. mFRR Up-Price Gap (€/MWh)",
    "afrr_mfrr_down_price_gap": "afrr vs. mFRR Down-Price Gap (€/MWh)",
    "realtime_wind_forecast_error": "Real-Time 3m Wind Forecast Error (MW)",
    "realtime_nuclear_generation_ramp": "Real-Time 3m Nuclear Generation Ramp (MW)",
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


def _compact_html(raw: str) -> str:
    """Strip leading indentation and blank lines so CommonMark never parses nested HTML as a code block."""
    return "\n".join(line.strip() for line in raw.splitlines() if line.strip())


def inject_brand_css(dark_mode: bool = True) -> str:
    """Return full CSS stylesheet implementing Apple Design (SKILL.md) + Turku Energia accents in Dark or Light mode."""
    if dark_mode:
        theme_vars = """
        --te-orange: #FF9F0A;
        --te-orange-hover: #FFB340;
        --te-orange-light: rgba(255, 159, 10, 0.15);
        --te-orange-soft: rgba(255, 159, 10, 0.32);
        --te-bg-app: #090C12;
        --te-bg-surface: rgba(24, 30, 42, 0.72);
        --te-bg-elevated: rgba(38, 47, 64, 0.65);
        --te-bg-sidebar: rgba(14, 18, 26, 0.85);
        --te-bg-hero: linear-gradient(135deg, rgba(28, 36, 52, 0.82) 0%, rgba(18, 24, 36, 0.88) 60%, rgba(42, 33, 24, 0.78) 100%);
        --te-border: rgba(255, 255, 255, 0.09);
        --te-border-subtle: rgba(255, 255, 255, 0.06);
        --te-ice: #B4DCE6;
        --te-turquoise: #5AC8FA;
        --te-lime: #30D158;
        --te-berry: #FF375F;
        --te-navy: #0A84FF;
        --te-plum: #BF5AF2;
        --te-yellow: #FFD60A;
        --te-stone: #2C2C2E;
        --te-gray: #98989D;
        --te-heading-color: #F5F5F7;
        --te-text: #E5E5EA;
        --te-text-muted: #98989D;
        --te-kpi-value-color: #FFFFFF;
        --te-delta-bg: rgba(90, 200, 250, 0.14);
        --te-delta-text: #70D7FF;
        --te-delta-ok-bg: rgba(48, 209, 88, 0.16);
        --te-delta-ok-text: #30D158;
        --te-alert-high-bg: rgba(255, 55, 95, 0.12);
        --te-alert-high-border: rgba(255, 55, 95, 0.35);
        --te-alert-low-bg: rgba(191, 90, 242, 0.12);
        --te-alert-low-border: rgba(191, 90, 242, 0.35);
        --te-alert-rev-bg: rgba(255, 159, 10, 0.12);
        --te-alert-rev-border: rgba(255, 159, 10, 0.35);
        --te-alert-norm-bg: rgba(48, 209, 88, 0.10);
        --te-alert-norm-border: rgba(48, 209, 88, 0.30);
        --te-msg-user-bg: linear-gradient(135deg, #0A84FF 0%, #0066CC 100%);
        --te-msg-ai-bg: rgba(38, 47, 64, 0.65);
        --te-bar-track: rgba(255, 255, 255, 0.08);
        --te-shadow-sm: 0 8px 24px rgba(0, 0, 0, 0.28), 0 1px 2px rgba(0, 0, 0, 0.2);
        --te-shadow-md: 0 16px 40px rgba(0, 0, 0, 0.42), 0 2px 6px rgba(0, 0, 0, 0.25);
        """
    else:
        theme_vars = """
        --te-orange: #F59B00;
        --te-orange-hover: #DF8C00;
        --te-orange-light: rgba(245, 155, 0, 0.12);
        --te-orange-soft: rgba(245, 155, 0, 0.25);
        --te-bg-app: #F5F5F7;
        --te-bg-surface: rgba(255, 255, 255, 0.82);
        --te-bg-elevated: rgba(242, 242, 247, 0.85);
        --te-bg-sidebar: rgba(255, 255, 255, 0.78);
        --te-bg-hero: linear-gradient(135deg, rgba(255, 255, 255, 0.92) 0%, rgba(242, 246, 252, 0.88) 65%, rgba(255, 248, 235, 0.9) 100%);
        --te-border: rgba(0, 0, 0, 0.07);
        --te-border-subtle: rgba(0, 0, 0, 0.045);
        --te-ice: #B4DCE6;
        --te-turquoise: #007AFF;
        --te-lime: #28CD41;
        --te-berry: #FF2D55;
        --te-navy: #1D1D1F;
        --te-plum: #AF52DE;
        --te-yellow: #FFCC00;
        --te-stone: #E5E5EA;
        --te-gray: #86868B;
        --te-heading-color: #1D1D1F;
        --te-text: #1D1D1F;
        --te-text-muted: #6E6E73;
        --te-kpi-value-color: #1D1D1F;
        --te-delta-bg: rgba(0, 122, 255, 0.09);
        --te-delta-text: #0066CC;
        --te-delta-ok-bg: rgba(40, 205, 65, 0.14);
        --te-delta-ok-text: #1D7A2B;
        --te-alert-high-bg: rgba(255, 45, 85, 0.08);
        --te-alert-high-border: rgba(255, 45, 85, 0.25);
        --te-alert-low-bg: rgba(175, 82, 222, 0.08);
        --te-alert-low-border: rgba(175, 82, 222, 0.25);
        --te-alert-rev-bg: rgba(245, 155, 0, 0.10);
        --te-alert-rev-border: rgba(245, 155, 0, 0.30);
        --te-alert-norm-bg: rgba(40, 205, 65, 0.08);
        --te-alert-norm-border: rgba(40, 205, 65, 0.25);
        --te-msg-user-bg: linear-gradient(135deg, #007AFF 0%, #0051D5 100%);
        --te-msg-ai-bg: rgba(242, 242, 247, 0.9);
        --te-bar-track: rgba(0, 0, 0, 0.06);
        --te-shadow-sm: 0 4px 20px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.03);
        --te-shadow-md: 0 12px 32px rgba(0, 0, 0, 0.08), 0 2px 6px rgba(0, 0, 0, 0.04);
        """

    return f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      :root {{
        {theme_vars}
        --te-font-sans: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Inter', system-ui, sans-serif;
        --te-radius-card: 20px;
        --te-radius-inner: 12px;
        --te-spring: cubic-bezier(0.22, 1, 0.36, 1);
        --ease-out: cubic-bezier(0.23, 1, 0.32, 1);
        --ease-in-out: cubic-bezier(0.77, 0, 0.175, 1);
        --ease-drawer: cubic-bezier(0.32, 0.72, 0, 1);
        --te-glass-blur: blur(28px) saturate(190%);
        --te-specular-top: {"inset 0 1px 0 0 rgba(255, 255, 255, 0.15)" if dark_mode else "inset 0 1px 0 0 rgba(255, 255, 255, 0.9)"};
      }}

      /* Choreographed Entrance Wave (emil-design-eng & choreograph: never scale(0), non-uniform <300ms stagger) */
      @keyframes teCardEnter {{
        from {{
          opacity: 0;
          transform: translateY(6px) scale(0.985);
        }}
        to {{
          opacity: 1;
          transform: translateY(0) scale(1);
        }}
      }}

      /* GPU-composited bar fill animation (animation-performance: transform scaleX instead of width) */
      @keyframes teBarScaleIn {{
        from {{
          transform: scaleX(0.12);
          opacity: 0.4;
        }}
        to {{
          transform: scaleX(1);
          opacity: 1;
        }}
      }}

      /* Global page & Apple Typography (§15: optical sizing, antialiasing, comfortable leading) */
      .stApp, [data-testid="stAppViewContainer"] {{
        background-color: var(--te-bg-app) !important;
        background-image: {"radial-gradient(ellipse 80% 50% at 15% -10%, rgba(90, 200, 250, 0.10), transparent), radial-gradient(ellipse 60% 40% at 85% 0%, rgba(255, 159, 10, 0.08), transparent)" if dark_mode else "radial-gradient(ellipse 80% 50% at 15% -10%, rgba(0, 122, 255, 0.05), transparent), radial-gradient(ellipse 60% 40% at 85% 0%, rgba(245, 155, 0, 0.06), transparent)"} !important;
        color: var(--te-text) !important;
        font-family: var(--te-font-sans) !important;
        font-optical-sizing: auto;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        line-height: 1.5;
        letter-spacing: -0.011em;
        transition: background-color 220ms var(--ease-out), color 220ms var(--ease-out);
      }}

      /* Top Header Bar — Apple Translucent Material (§12: backdrop-filter + saturate(180%)) */
      header[data-testid="stHeader"] {{
        height: 3rem !important;
        min-height: 48px !important;
        background: {"rgba(9, 12, 18, 0.72)" if dark_mode else "rgba(245, 245, 247, 0.75)"} !important;
        backdrop-filter: blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
        border-bottom: 1px solid var(--te-border) !important;
        display: flex !important;
        align-items: center !important;
      }}

      header[data-testid="stHeader"] > div,
      div[data-testid="stToolbar"],
      div[data-testid="stToolbarActions"],
      div[data-testid="stAppDeployButton"],
      .stAppDeployButton,
      span[data-testid="stMainMenu"] {{
        display: inline-flex !important;
        align-items: center !important;
        height: 3rem !important;
        min-height: 48px !important;
        top: 0 !important;
        margin-top: 0 !important;
        margin-bottom: 0 !important;
      }}

      div[data-testid="stToolbar"] {{
        right: 1rem !important;
        gap: 0.45rem !important;
      }}

      div[data-testid="stAppDeployButton"] button,
      .stAppDeployButton button {{
        height: 32px !important;
        min-height: 32px !important;
        padding: 0 12px !important;
        border-radius: 999px !important;
        border: 1px solid var(--te-border) !important;
        background: var(--te-bg-surface) !important;
        color: var(--te-text) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        transition: background-color 160ms var(--ease-out), border-color 160ms var(--ease-out), transform 140ms var(--ease-out) !important;
      }}

      @media (hover: hover) and (pointer: fine) {{
        div[data-testid="stAppDeployButton"] button:hover,
        .stAppDeployButton button:hover {{
          border-color: var(--te-orange) !important;
          background: var(--te-orange-light) !important;
        }}
      }}

      .block-container,
      [data-testid="stMainBlockContainer"] {{
        padding-top: calc(3rem + 10px) !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        max-width: 100% !important;
      }}

      /* Remove vertical flex gap from zero-height helper iframes & fixed theme trigger */
      div.stElementContainer:has(iframe[height="0"]),
      div.element-container:has(iframe[height="0"]) {{
        height: 0 !important;
        min-height: 0 !important;
        margin: -0.5rem 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
      }}

      h1, h2, h3, h4, .te-heading {{
        font-family: var(--te-font-sans) !important;
        color: var(--te-heading-color) !important;
        letter-spacing: -0.024em !important;
        font-weight: 700 !important;
      }}

      p, li, label, .stMarkdown, [data-testid="stWidgetLabel"] p {{
        color: var(--te-text) !important;
        font-family: var(--te-font-sans) !important;
      }}

      code {{
        background: var(--te-bg-elevated) !important;
        color: var(--te-turquoise) !important;
        border: 1px solid var(--te-border-subtle);
        padding: 0.12rem 0.42rem !important;
        border-radius: 6px !important;
        font-family: 'SF Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
        font-size: 0.82em !important;
      }}

      /* Sidebar styling (Apple macOS Sonoma Vibrancy Sidebar) */
      section[data-testid="stSidebar"] {{
        background: var(--te-bg-sidebar) !important;
        backdrop-filter: var(--te-glass-blur) !important;
        -webkit-backdrop-filter: var(--te-glass-blur) !important;
        border-right: 1px solid var(--te-border) !important;
      }}

      [data-testid="stSidebarHeader"] {{
        height: 3rem !important;
        min-height: 48px !important;
        padding: 0 1.05rem !important;
        margin-bottom: 0 !important;
        display: flex !important;
        align-items: center !important;
      }}

      section[data-testid="stSidebar"] .block-container,
      [data-testid="stSidebarUserContent"] {{
        padding-top: 6px !important;
        padding-left: 1.05rem !important;
        padding-right: 1.05rem !important;
        padding-bottom: 1.25rem !important;
      }}

      /* Sleek Sidebar Collapse (<<) & Expand (>>) Buttons */
      [data-testid="stSidebarCollapseButton"] button,
      [data-testid="stExpandSidebarButton"],
      [data-testid="collapsedControl"] button {{
        width: 34px !important;
        height: 34px !important;
        min-height: 34px !important;
        border-radius: 10px !important;
        background: var(--te-bg-surface) !important;
        border: 1px solid var(--te-border) !important;
        color: var(--te-turquoise) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: var(--te-shadow-sm) !important;
        transition: transform 160ms var(--ease-out), background-color 160ms var(--ease-out), border-color 160ms var(--ease-out), color 160ms var(--ease-out) !important;
      }}

      @media (hover: hover) and (pointer: fine) {{
        [data-testid="stSidebarCollapseButton"] button:hover,
        [data-testid="stExpandSidebarButton"]:hover,
        [data-testid="collapsedControl"] button:hover {{
          border-color: var(--te-orange) !important;
          background: var(--te-orange-light) !important;
          color: var(--te-orange) !important;
          transform: scale(1.04) !important;
        }}
      }}

      /* Sidebar Header Card — Apple Bento Card */
      .te-sb-header-card {{
        background: var(--te-bg-surface);
        backdrop-filter: var(--te-glass-blur);
        -webkit-backdrop-filter: var(--te-glass-blur);
        border: 1px solid var(--te-border);
        border-radius: 16px;
        padding: 0.9rem 1.05rem;
        margin-bottom: 1rem;
        box-shadow: var(--te-specular-top), var(--te-shadow-sm);
      }}

      .te-sb-kicker {{
        font-family: var(--te-font-sans);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--te-orange);
        margin-bottom: 0.2rem;
      }}

      .te-sb-title {{
        font-family: var(--te-font-sans);
        font-size: 1.18rem;
        font-weight: 700;
        letter-spacing: -0.022em;
        color: var(--te-heading-color);
        margin: 0 0 0.25rem 0;
        line-height: 1.18;
      }}

      .te-sb-desc {{
        font-size: 0.78rem;
        line-height: 1.42;
        color: var(--te-text-muted);
        margin: 0;
      }}

      /* Sidebar Section Labels */
      .te-sb-section-label {{
        font-family: var(--te-font-sans);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--te-text-muted);
        margin: 0.75rem 0 0.4rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}

      .te-sb-count-pill {{
        font-family: var(--te-font-sans);
        font-size: 0.68rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        color: var(--te-text);
        background: var(--te-bg-elevated);
        border: 1px solid var(--te-border);
        padding: 0.1rem 0.52rem;
        border-radius: 99px;
      }}

      /* Styled Dataset Period Radio Cards inside Sidebar */
      section[data-testid="stSidebar"] div[role="radiogroup"] {{
        gap: 0.42rem !important;
      }}

      section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
        background: var(--te-bg-surface) !important;
        border: 1px solid var(--te-border) !important;
        border-radius: 12px !important;
        padding: 0.52rem 0.78rem !important;
        margin: 0 !important;
        width: 100% !important;
        transition: background-color 160ms var(--ease-out), border-color 160ms var(--ease-out), transform 140ms var(--ease-out) !important;
        cursor: pointer !important;
      }}

      @media (hover: hover) and (pointer: fine) {{
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
          border-color: var(--te-orange) !important;
          background: var(--te-bg-elevated) !important;
        }}
      }}

      section[data-testid="stSidebar"] div[role="radiogroup"] > label:active {{
        transform: scale(0.98) !important;
      }}

      section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
        border-color: var(--te-orange) !important;
        background: var(--te-orange-light) !important;
        box-shadow: var(--te-specular-top), var(--te-shadow-sm) !important;
      }}

      /* Sidebar Prev / Next Step Buttons */
      section[data-testid="stSidebar"] .stButton > button {{
        height: 38px !important;
        min-height: 38px !important;
        border-radius: 11px !important;
        background: var(--te-bg-surface) !important;
        border: 1px solid var(--te-border) !important;
        font-family: var(--te-font-sans) !important;
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        box-shadow: var(--te-specular-top), var(--te-shadow-sm) !important;
      }}

      @media (hover: hover) and (pointer: fine) {{
        section[data-testid="stSidebar"] .stButton > button:hover {{
          border-color: var(--te-orange) !important;
          background: var(--te-orange) !important;
          color: #FFFFFF !important;
          transform: translateY(-1px) !important;
          box-shadow: 0 4px 14px rgba(245, 155, 0, 0.32) !important;
        }}
      }}

      /* Native Selectbox / Input / Expander Dark Polish */
      div[data-baseweb="select"] > div,
      div[data-baseweb="input"] > div,
      div[data-baseweb="base-input"] {{
        background-color: var(--te-bg-surface) !important;
        border-color: var(--te-border) !important;
        color: var(--te-text) !important;
        border-radius: 12px !important;
        transition: border-color 160ms var(--ease-out), box-shadow 160ms var(--ease-out) !important;
      }}

      div[data-testid="stExpander"] {{
        background-color: var(--te-bg-surface) !important;
        backdrop-filter: var(--te-glass-blur) !important;
        border: 1px solid var(--te-border) !important;
        border-radius: 16px !important;
      }}

      div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: var(--te-bg-surface) !important;
        backdrop-filter: var(--te-glass-blur) !important;
        -webkit-backdrop-filter: var(--te-glass-blur) !important;
        border-color: var(--te-border) !important;
        border-radius: 20px !important;
        box-shadow: var(--te-specular-top), var(--te-shadow-sm) !important;
      }}

      /* Apple Bento Card Frame around every Plotly Chart */
      div[data-testid="stPlotlyChart"] {{
        background: var(--te-bg-surface) !important;
        backdrop-filter: var(--te-glass-blur) !important;
        -webkit-backdrop-filter: var(--te-glass-blur) !important;
        border: 1px solid var(--te-border) !important;
        border-radius: 20px !important;
        overflow: hidden !important;
        box-shadow: var(--te-specular-top), var(--te-shadow-sm) !important;
        margin-bottom: 0.4rem !important;
      }}

      /* Hero Banner — Lead Element (0ms entrance, 240ms strong ease-out) */
      .te-hero-banner {{
        position: relative;
        overflow: hidden;
        background: var(--te-bg-hero);
        backdrop-filter: var(--te-glass-blur);
        -webkit-backdrop-filter: var(--te-glass-blur);
        border: 1px solid var(--te-border);
        border-radius: 22px;
        padding: 1.35rem 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--te-specular-top), var(--te-shadow-sm);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        animation: teCardEnter 240ms var(--ease-out) both;
      }}

      .te-hero-banner::before {{
        content: "";
        position: absolute;
        right: -40px;
        top: -80px;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(255, 159, 10, 0.14) 0%, rgba(90, 200, 250, 0.08) 50%, transparent 72%);
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
        font-family: var(--te-font-sans) !important;
        font-size: clamp(1.55rem, 2.2vw, 1.95rem);
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        color: var(--te-heading-color) !important;
        line-height: 1.08;
        margin: 0 0 0.28rem 0;
      }}

      .te-hero-title span.te-dot {{
        color: var(--te-orange);
      }}

      .te-hero-subtitle {{
        font-size: 0.92rem;
        color: var(--te-text-muted) !important;
        margin: 0;
        font-weight: 400;
        letter-spacing: -0.008em;
      }}

      .te-slogan-badge {{
        font-family: var(--te-font-sans);
        font-size: 0.88rem;
        font-weight: 500;
        color: var(--te-heading-color);
        background: var(--te-bg-elevated);
        border: 1px solid var(--te-border);
        padding: 0.6rem 1.1rem;
        border-radius: 14px;
        white-space: nowrap;
        box-shadow: var(--te-specular-top);
        z-index: 1;
        text-align: right;
      }}

      .te-slogan-sub {{
        display: block;
        font-family: var(--te-font-sans);
        font-size: 0.72rem;
        color: var(--te-text-muted);
        margin-top: 0.14rem;
      }}

      /* KPI Cards — Apple Health / Apple Stocks Bento Widgets with Non-Uniform Stagger (<300ms) */
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
        background-image: radial-gradient(120% 100% at 0% 0%, rgba(255, 255, 255, 0.045), transparent 60%);
        backdrop-filter: var(--te-glass-blur);
        -webkit-backdrop-filter: var(--te-glass-blur);
        border: 1px solid var(--te-border);
        border-radius: var(--te-radius-card);
        padding: 1.1rem 1.25rem;
        box-shadow: var(--te-specular-top), var(--te-shadow-sm);
        transition: transform 180ms var(--ease-out), box-shadow 180ms var(--ease-out), border-color 180ms var(--ease-out);
        will-change: transform;
        position: relative;
        overflow: hidden;
        animation: teCardEnter 240ms var(--ease-out) both;
      }}

      /* Non-uniform stagger (choreograph Rule #3: shrinking delays 20ms -> 55ms -> 85ms -> 110ms -> 130ms) */
      .te-kpi-card:nth-child(1) {{ animation-delay: 20ms; }}
      .te-kpi-card:nth-child(2) {{ animation-delay: 55ms; }}
      .te-kpi-card:nth-child(3) {{ animation-delay: 85ms; }}
      .te-kpi-card:nth-child(4) {{ animation-delay: 110ms; }}
      .te-kpi-card:nth-child(5) {{ animation-delay: 130ms; }}

      .te-kpi-card::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 16px;
        right: 16px;
        height: 2.5px;
        border-radius: 99px;
        background: var(--te-turquoise);
        opacity: 0.85;
      }}

      @media (hover: hover) and (pointer: fine) {{
        .te-kpi-card:hover {{
          transform: translateY(-2px);
          box-shadow: var(--te-specular-top), var(--te-shadow-md);
          border-color: rgba(255, 159, 10, 0.35);
        }}
      }}

      .te-kpi-card:active {{
        transform: scale(0.975);
        transition: transform 110ms var(--ease-out);
      }}

      .te-kpi-card.accent-orange::before {{ background: var(--te-orange); }}
      .te-kpi-card.accent-berry::before  {{ background: var(--te-berry); }}
      .te-kpi-card.accent-plum::before   {{ background: var(--te-plum); }}
      .te-kpi-card.accent-turquoise::before {{ background: var(--te-turquoise); }}
      .te-kpi-card.accent-lime::before   {{ background: var(--te-lime); }}

      .te-kpi-label {{
        font-family: var(--te-font-sans);
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.055em;
        color: var(--te-text-muted);
        margin-bottom: 0.38rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}

      .te-kpi-value {{
        font-family: var(--te-font-sans);
        font-size: clamp(1.55rem, 2vw, 1.92rem);
        font-weight: 700;
        color: var(--te-kpi-value-color);
        line-height: 1.06;
        letter-spacing: -0.03em;
        font-variant-numeric: tabular-nums;
        margin-bottom: 0.48rem;
      }}

      .te-kpi-delta {{
        font-size: 0.76rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        color: var(--te-delta-text);
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.18rem 0.6rem;
        border-radius: 99px;
        background: var(--te-delta-bg);
      }}

      .te-kpi-delta.warn {{
        background: rgba(255, 55, 95, 0.18);
        color: var(--te-berry) !important;
        border: 1px solid rgba(255, 55, 95, 0.35);
        font-weight: 600;
      }}

      .te-kpi-delta.ok {{
        background: var(--te-delta-ok-bg);
        color: var(--te-delta-ok-text);
        font-weight: 600;
      }}

      /* Situational Alert Banner — Apple Notification Pill Banner */
      .te-alert-card {{
        border-radius: 18px;
        padding: 1rem 1.3rem;
        margin-bottom: 1rem;
        border: 1px solid var(--te-border);
        backdrop-filter: var(--te-glass-blur);
        -webkit-backdrop-filter: var(--te-glass-blur);
        box-shadow: var(--te-specular-top), var(--te-shadow-sm);
        transition: transform 180ms var(--ease-out), border-color 180ms var(--ease-out), box-shadow 180ms var(--ease-out);
        animation: teCardEnter 240ms var(--ease-out) 60ms both;
      }}

      .te-alert-card.level-HIGH {{
        background: var(--te-alert-high-bg);
        border-color: var(--te-alert-high-border);
      }}

      .te-alert-card.level-LOW_SPIKE {{
        background: var(--te-alert-low-bg);
        border-color: var(--te-alert-low-border);
      }}

      .te-alert-card.level-REVISION {{
        background: var(--te-alert-rev-bg);
        border-color: var(--te-alert-rev-border);
      }}

      .te-alert-card.level-NORMAL {{
        background: var(--te-alert-norm-bg);
        border-color: var(--te-alert-norm-border);
      }}

      .te-alert-header {{
        display: flex;
        align-items: center;
        gap: 0.65rem;
        cursor: pointer;
        list-style: none;
        flex-wrap: wrap;
      }}

      .te-alert-header::-webkit-details-marker {{ display: none; }}

      .te-alert-chevron {{
        margin-left: auto;
        font-size: 1.1rem;
        color: var(--te-text-muted);
        transition: transform 180ms var(--ease-out);
      }}

      details.te-alert-card[open] .te-alert-chevron {{
        transform: rotate(90deg);
      }}

      details.te-alert-card .te-alert-body {{
        margin-top: 0.5rem;
      }}

      .te-xai-note {{
        font-size: 0.72rem;
        line-height: 1.4;
        color: var(--te-text-muted);
        margin-top: 0.5rem;
      }}

      .te-alert-badge {{
        font-family: var(--te-font-sans);
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.055em;
        padding: 0.2rem 0.65rem;
        border-radius: 99px;
        color: #FFFFFF !important;
      }}

      .badge-HIGH {{ background: var(--te-berry); }}
      .badge-LOW_SPIKE {{ background: var(--te-plum); }}
      .badge-REVISION {{ background: var(--te-orange); color: #FFFFFF !important; }}
      .badge-NORMAL {{ background: var(--te-lime); color: #090C12 !important; }}

      .te-alert-title {{
        font-family: var(--te-font-sans);
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.018em;
        color: var(--te-heading-color) !important;
        margin: 0;
      }}

      .te-alert-body {{
        font-size: 0.88rem;
        line-height: 1.5;
        color: var(--te-text) !important;
        margin: 0;
      }}

      /* Chart Section Header — Clean Apple Section Header */
      .te-section-header {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-bottom: 1px solid var(--te-border);
        padding-bottom: 0.5rem;
        margin-top: 0.55rem;
        margin-bottom: 0.75rem;
        gap: 0.75rem;
        flex-wrap: wrap;
      }}

      .te-section-title {{
        font-family: var(--te-font-sans) !important;
        font-size: 1.12rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.022em !important;
        color: var(--te-heading-color) !important;
        margin: 0;
      }}

      .te-section-title span {{
        color: var(--te-orange);
      }}

      .te-section-subtitle {{
        font-size: 0.8rem;
        color: var(--te-text-muted);
        font-weight: 500;
      }}

      /* AI Explainability Chat Sidebar Panel — Apple Intelligence Card */
      .te-chat-panel-header {{
        background: {"linear-gradient(135deg, rgba(28, 38, 58, 0.85) 0%, rgba(18, 25, 38, 0.92) 100%)" if dark_mode else "linear-gradient(135deg, #FFFFFF 0%, #F2F6FC 100%)"};
        backdrop-filter: var(--te-glass-blur);
        -webkit-backdrop-filter: var(--te-glass-blur);
        color: var(--te-heading-color);
        padding: 1rem 1.15rem;
        border-radius: 18px;
        margin-bottom: 0.85rem;
        box-shadow: var(--te-specular-top), var(--te-shadow-sm);
        border: 1px solid var(--te-border);
      }}

      .te-chat-panel-title {{
        font-family: var(--te-font-sans);
        font-size: 1.08rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--te-heading-color) !important;
        margin: 0 0 0.22rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}

      .te-chat-status-pill {{
        font-family: var(--te-font-sans);
        font-size: 0.68rem;
        font-weight: 700;
        background: var(--te-orange);
        color: #FFFFFF !important;
        padding: 0.16rem 0.6rem;
        border-radius: 99px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }}

      .te-chat-panel-sub {{
        font-size: 0.8rem;
        color: var(--te-text-muted) !important;
        margin: 0;
      }}

      /* Differentiated iMessage-style Chat Bubbles */
      .te-msg-user {{
        background: var(--te-msg-user-bg);
        color: #FFFFFF !important;
        padding: 0.7rem 0.95rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.35rem 0 0.45rem 1.25rem;
        font-size: 0.86rem;
        line-height: 1.44;
        box-shadow: var(--te-shadow-sm);
        word-wrap: break-word;
        overflow-wrap: anywhere;
      }}

      .te-msg-user-label {{
        font-size: 0.66rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: rgba(255, 255, 255, 0.78);
        font-weight: 700;
        margin-bottom: 0.16rem;
      }}

      .te-msg-ai-header {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-family: var(--te-font-sans);
        font-size: 0.74rem;
        letter-spacing: -0.01em;
        color: var(--te-turquoise);
        font-weight: 600;
        background: var(--te-bg-elevated);
        border: 1px solid var(--te-border);
        padding: 0.22rem 0.68rem;
        border-radius: 99px;
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
      }}

      .te-quick-prompts-bar {{
        font-family: var(--te-font-sans);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.055em;
        text-transform: uppercase;
        color: var(--te-text-muted);
        margin: 0.35rem 0 0.25rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}

      /* Mask API Key Input visually without using type="password" so Google Password Manager never triggers */
      div.st-key-gemini_api_key_input input {{
        -webkit-text-security: disc !important;
        text-security: disc !important;
        font-family: 'Fira Mono', monospace !important;
        letter-spacing: 0.06em !important;
      }}

      /* Compact Markdown Tables inside Chat (No Horizontal Scroll!) */
      [data-testid="stVerticalBlockBorderWrapper"] .stMarkdown table {{
        width: 100% !important;
        table-layout: auto !important;
        border-collapse: collapse !important;
        font-size: 0.77rem !important;
        margin: 0.35rem 0 !important;
      }}

      [data-testid="stVerticalBlockBorderWrapper"] .stMarkdown th,
      [data-testid="stVerticalBlockBorderWrapper"] .stMarkdown td {{
        padding: 0.28rem 0.42rem !important;
        border: 1px solid var(--te-border) !important;
        white-space: normal !important;
        word-break: break-word !important;
      }}

      [data-testid="stVerticalBlockBorderWrapper"] .stMarkdown th {{
        background: var(--te-bg-elevated) !important;
        color: var(--te-turquoise) !important;
        font-family: var(--te-font-sans) !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
      }}

      /* Feature Weighting Bars inside Chat / Explainability */
      .te-driver-box {{
        margin-top: 0.35rem;
        margin-bottom: 0.25rem;
        padding: 0.6rem 0.8rem;
        background: var(--te-bg-elevated);
        border-radius: 12px;
        border: 1px solid var(--te-border);
      }}

      .te-driver-box-title {{
        font-family: var(--te-font-sans);
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.055em;
        color: var(--te-turquoise);
        text-transform: uppercase;
        margin-bottom: 0.32rem;
      }}

      .te-driver-bar-row {{
        margin: 0.32rem 0;
      }}

      .te-driver-bar-top {{
        display: flex;
        justify-content: space-between;
        gap: 0.4rem;
        font-size: 0.74rem;
        font-weight: 600;
        color: var(--te-text);
        margin-bottom: 0.14rem;
      }}

      .te-driver-bar-track {{
        width: 100%;
        height: 6px;
        background: var(--te-bar-track);
        border-radius: 99px;
        overflow: hidden;
      }}

      .te-driver-bar-fill {{
        height: 100%;
        background: linear-gradient(90deg, var(--te-turquoise) 0%, var(--te-orange) 100%);
        border-radius: 99px;
        transform-origin: left center;
        animation: teBarScaleIn 260ms var(--ease-out) 90ms both;
      }}

      /* XAI Bento Cards & Tactile Driver Rows (emil-design-eng + choreograph) */
      .te-xai-card {{
        background: var(--te-bg-surface);
        background-image: radial-gradient(120% 100% at 0% 0%, rgba(255, 255, 255, 0.045), transparent 60%);
        backdrop-filter: var(--te-glass-blur);
        -webkit-backdrop-filter: var(--te-glass-blur);
        border: 1px solid var(--te-border);
        border-radius: 20px;
        padding: 1.2rem 1.35rem;
        box-shadow: var(--te-specular-top), var(--te-shadow-sm);
        transition: transform 180ms var(--ease-out), box-shadow 180ms var(--ease-out), border-color 180ms var(--ease-out);
        animation: teCardEnter 260ms var(--ease-out) both;
      }}

      .te-xai-card:nth-child(1) {{ animation-delay: 80ms; }}
      .te-xai-card:nth-child(2) {{ animation-delay: 125ms; }}

      @media (hover: hover) and (pointer: fine) {{
        .te-xai-card:hover {{
          transform: translateY(-2px);
          box-shadow: var(--te-specular-top), var(--te-shadow-md);
          border-color: rgba(255, 159, 10, 0.30);
        }}
      }}

      .te-xai-row {{
        padding: 0.46rem 0.72rem;
        margin-bottom: 0.36rem;
        background: var(--te-bg-elevated);
        border-radius: 12px;
        border: 1px solid var(--te-border-subtle);
        transition: transform 150ms var(--ease-out), border-color 150ms var(--ease-out), background-color 150ms var(--ease-out);
      }}

      @media (hover: hover) and (pointer: fine) {{
        .te-xai-row:hover {{
          transform: translateX(3px);
          border-color: var(--te-border);
        }}
      }}

      .te-xai-bar-fill {{
        height: 100%;
        border-radius: 99px;
        transform-origin: left center;
        animation: teBarScaleIn 280ms var(--ease-out) 140ms both;
      }}

      /* Buttons & Tabs refinement — Apple §1 Instant Press Response & Emil <300ms ease-out */
      .stButton > button {{
        background-color: var(--te-bg-surface) !important;
        color: var(--te-text) !important;
        border-radius: 12px !important;
        font-family: var(--te-font-sans) !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        border: 1px solid var(--te-border) !important;
        box-shadow: var(--te-specular-top), var(--te-shadow-sm) !important;
        transition: transform 150ms var(--ease-out), background-color 150ms var(--ease-out), border-color 150ms var(--ease-out), box-shadow 150ms var(--ease-out) !important;
        will-change: transform;
      }}

      @media (hover: hover) and (pointer: fine) {{
        .stButton > button:hover {{
          border-color: var(--te-orange) !important;
          background-color: var(--te-orange-light) !important;
          color: var(--te-heading-color) !important;
          transform: translateY(-1px);
        }}
      }}

      /* Apple §1 & emil-design-eng: Instant pointer-down press feedback (110ms ease-out scale(0.97)) */
      .stButton > button:active,
      div[data-testid="stFormSubmitButton"] > button:active,
      section[data-testid="stSidebar"] .stButton > button:active,
      button[data-baseweb="tab"]:active {{
        transform: scale(0.97) !important;
        transition: transform 110ms var(--ease-out) !important;
      }}

      .stButton > button[kind="primary"],
      div[data-testid="stFormSubmitButton"] > button {{
        background-color: var(--te-orange) !important;
        color: #FFFFFF !important;
        border-color: var(--te-orange) !important;
        font-weight: 600 !important;
      }}

      @media (hover: hover) and (pointer: fine) {{
        .stButton > button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {{
          background-color: var(--te-orange-hover) !important;
          color: #FFFFFF !important;
        }}
      }}

      /* Apple iOS / macOS Sonoma Segmented Pill Tab Bar (§12 Translucent Material + §4 Spring Pill) */
      div[data-baseweb="tab-list"] {{
        background: {"rgba(118, 118, 128, 0.22)" if dark_mode else "rgba(118, 118, 128, 0.12)"} !important;
        backdrop-filter: var(--te-glass-blur) !important;
        -webkit-backdrop-filter: var(--te-glass-blur) !important;
        padding: 4px !important;
        border-radius: 14px !important;
        border: 1px solid var(--te-border) !important;
        gap: 4px !important;
        margin-bottom: 1rem !important;
      }}

      div[data-baseweb="tab-highlight"] {{
        display: none !important;
      }}

      button[data-baseweb="tab"] {{
        font-family: var(--te-font-sans) !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.012em !important;
        color: var(--te-text-muted) !important;
        border-radius: 10px !important;
        padding: 0.48rem 1.05rem !important;
        border: none !important;
        transition: background-color 180ms var(--ease-out), color 180ms var(--ease-out), box-shadow 180ms var(--ease-out), transform 120ms var(--ease-out) !important;
      }}

      button[data-baseweb="tab"][aria-selected="true"] {{
        background: {"rgba(44, 52, 68, 0.92)" if dark_mode else "#FFFFFF"} !important;
        color: var(--te-heading-color) !important;
        box-shadow: var(--te-specular-top), 0 3px 10px rgba(0, 0, 0, 0.18), 0 1px 2px rgba(0, 0, 0, 0.08) !important;
      }}

      /* Apple §14: Reduced Motion, Reduced Transparency & High Contrast Media Queries */
      @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
          animation-duration: 0.01ms !important;
          animation-iteration-count: 1 !important;
          transition-duration: 150ms !important;
          transform: none !important;
        }}
      }}

      @media (prefers-reduced-transparency: reduce) {{
        header[data-testid="stHeader"],
        .te-kpi-card,
        .te-xai-card,
        div[data-baseweb="tab-list"] {{
          backdrop-filter: none !important;
          -webkit-backdrop-filter: none !important;
          background: var(--te-bg-surface) !important;
        }}
      }}

      @media (prefers-contrast: more) {{
        .te-kpi-card, .te-alert-card, .te-xai-card {{
          border-width: 2px !important;
        }}
      }}

      /* Anti-Leakage Gate Box — Apple Security Card */
      .te-gate-box {{
        background: var(--te-bg-surface);
        backdrop-filter: var(--te-glass-blur);
        -webkit-backdrop-filter: var(--te-glass-blur);
        border: 1px solid var(--te-border);
        border-radius: 16px;
        padding: 0.88rem 0.95rem;
        font-size: 0.78rem;
        color: var(--te-text);
        box-shadow: var(--te-specular-top), var(--te-shadow-sm);
      }}

      .te-gate-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 0.28rem 0;
        gap: 0.4rem;
        white-space: nowrap;
      }}

      .te-gate-verified {{
        margin-top: 0.52rem;
        padding: 0.32rem 0.6rem;
        border-radius: 99px;
        background: var(--te-delta-ok-bg);
        color: var(--te-delta-ok-text);
        font-weight: 600;
        font-size: 0.73rem;
        white-space: nowrap;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.3rem;
      }}

      /* =====================================================================
         Top-Right Sun <-> Moon Morphing Theme Toggle (Next to Deploy button)
         ===================================================================== */
      @keyframes teMoonMorph {{
        0% {{
          transform: translateX(-28px) rotate(-160deg) scale(0.35);
          opacity: 0.2;
        }}
        65% {{
          transform: translateX(2px) rotate(12deg) scale(1.08);
          opacity: 1;
        }}
        100% {{
          transform: translateX(0px) rotate(0deg) scale(1);
          opacity: 1;
        }}
      }}

      @keyframes teSunMorph {{
        0% {{
          transform: translateX(28px) rotate(160deg) scale(0.35);
          opacity: 0.2;
        }}
        65% {{
          transform: translateX(-2px) rotate(-12deg) scale(1.08);
          opacity: 1;
        }}
        100% {{
          transform: translateX(0px) rotate(0deg) scale(1);
          opacity: 1;
        }}
      }}

      div.st-key-te_theme_toggle_btn {{
        position: fixed !important;
        top: 0.52rem !important;
        right: 8.6rem !important;
        z-index: 1000005 !important;
        width: auto !important;
        margin: 0 !important;
        padding: 0 !important;
      }}

      div.st-key-te_theme_toggle_btn > div {{
        width: auto !important;
      }}

      div.st-key-te_theme_toggle_btn button {{
        position: relative !important;
        width: 70px !important;
        height: 34px !important;
        min-height: 34px !important;
        padding: 0 !important;
        border-radius: 999px !important;
        background: {"linear-gradient(135deg, #091A2A 0%, #13314C 100%)" if dark_mode else "linear-gradient(135deg, #FFF4E0 0%, #EAF7FB 100%)"} !important;
        border: 1.5px solid {"#46BEDC" if dark_mode else "#F59B00"} !important;
        box-shadow: {"0 0 14px rgba(70, 190, 220, 0.35), inset 0 1px 4px rgba(0,0,0,0.6)" if dark_mode else "0 2px 10px rgba(245, 155, 0, 0.28), inset 0 1px 3px rgba(245, 155, 0, 0.15)"} !important;
        cursor: pointer !important;
        overflow: hidden !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
      }}

      div.st-key-te_theme_toggle_btn button:hover {{
        transform: scale(1.06) !important;
        border-color: var(--te-orange) !important;
        box-shadow: 0 0 18px rgba(245, 155, 0, 0.55) !important;
      }}

      div.st-key-te_theme_toggle_btn button p {{
        display: none !important;
      }}

      /* Left Icon: Sun (Active in Light Mode, Dormant/Morphed in Dark Mode) */
      div.st-key-te_theme_toggle_btn button::before {{
        content: "";
        position: absolute;
        left: 4px;
        top: 3px;
        width: 25px;
        height: 25px;
        border-radius: 50%;
        background-color: {"transparent" if dark_mode else "#F59B00"};
        box-shadow: {"none" if dark_mode else "0 2px 8px rgba(245, 155, 0, 0.55)"};
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='4' fill='%23FFD719'/%3E%3Cpath d='M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41'/%3E%3C/svg%3E");
        background-size: 17px 17px;
        background-position: center;
        background-repeat: no-repeat;
        transform: {"scale(0.62) rotate(-90deg)" if dark_mode else "scale(1) rotate(0deg)"};
        opacity: {"0.42" if dark_mode else "1"};
        animation: {"none" if dark_mode else "teSunMorph 0.52s cubic-bezier(0.34, 1.56, 0.64, 1) forwards"};
        transition: transform 0.45s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.35s ease, background-color 0.35s ease;
      }}

      /* Right Icon: Crescent Moon & Stars (Active in Dark Mode, Dormant/Morphed in Light Mode) */
      div.st-key-te_theme_toggle_btn button::after {{
        content: "";
        position: absolute;
        right: 4px;
        top: 3px;
        width: 25px;
        height: 25px;
        border-radius: 50%;
        background-color: {"#0F4B73" if dark_mode else "transparent"};
        box-shadow: {"0 0 10px rgba(70, 190, 220, 0.65)" if dark_mode else "none"};
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%2346BEDC' stroke='%23FFFFFF' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z'/%3E%3C/svg%3E");
        background-size: 16px 16px;
        background-position: center;
        background-repeat: no-repeat;
        transform: {"scale(1) rotate(0deg)" if dark_mode else "scale(0.62) rotate(90deg)"};
        opacity: {"1" if dark_mode else "0.42"};
        animation: {"teMoonMorph 0.52s cubic-bezier(0.34, 1.56, 0.64, 1) forwards" if dark_mode else "none"};
        transition: transform 0.45s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.35s ease, background-color 0.35s ease;
      }}

      div.st-key-te_theme_toggle_btn button:hover::before {{
        transform: {"scale(0.82) rotate(-25deg)" if dark_mode else "scale(1.08) rotate(25deg)"};
        opacity: {"0.85" if dark_mode else "1"};
      }}

      div.st-key-te_theme_toggle_btn button:hover::after {{
        transform: {"scale(1.08) rotate(-15deg)" if dark_mode else "scale(0.82) rotate(25deg)"};
        opacity: {"1" if dark_mode else "0.85"};
      }}
    </style>
    """


def render_hero_header(origin_str: str, period_label: str) -> str:
    """Render the Turku Energia brand header with official 8:5 logo and slogan in Apple Bento style."""
    logo_svg = get_turku_energia_logo_svg(width=114)
    return _compact_html(f"""
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
    """)


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
    """Render the 5 KPI cards in Apple Bento style."""
    spread_mean = p50_mean - spot_mean
    spread_sign = f"{spread_mean:+.2f} €/MWh vs. Spot"

    hi_accent = "accent-berry" if n_hi_steps > 0 else "accent-turquoise"
    hi_delta_cls = "warn" if n_hi_steps > 0 else "ok"
    hi_delta_txt = f"🚨 {n_hi_steps} QH steps ≥ 30%" if n_hi_steps > 0 else "✓ Below 30% alert threshold"

    lo_accent = "accent-plum" if n_lo_steps > 0 else "accent-turquoise"
    lo_delta_cls = "warn" if n_lo_steps > 0 else "ok"
    lo_delta_txt = f"⚠️ {n_lo_steps} QH steps ≥ 30%" if n_lo_steps > 0 else "✓ Below 30% alert threshold"

    dir_label = "↑ Up (Deficit)" if dominant_dir_short == "UP" else "↓ Down (Surplus)"

    return _compact_html(f"""
    <div class="te-kpi-grid">
      <div class="te-kpi-card accent-orange">
        <div class="te-kpi-label"><span>10h Forecast Mean (P50)</span><span>€/MWh</span></div>
        <div class="te-kpi-value">{p50_mean:.2f} €</div>
        <div class="te-kpi-delta">{html.escape(spread_sign)} (Spot {spot_mean:.1f} €)</div>
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
    """)


def render_alert_cards(alerts: list[dict[str, str]]) -> str:
    """Render structured situational awareness alerts in Apple Notification style."""
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
        # Declutter: spike alerts open by default, revision / normal collapsed to a single line
        open_attr = " open" if lvl in ("HIGH", "LOW_SPIKE") else ""
        cards_html.append(
            _compact_html(f"""
            <details class="te-alert-card level-{lvl}"{open_attr}>
              <summary class="te-alert-header">
                <span class="te-alert-badge badge-{lvl}">{badge_txt}</span>
                <span class="te-alert-title">{title}</span>
                <span class="te-alert-chevron">›</span>
              </summary>
              <p class="te-alert-body">{body}</p>
            </details>
            """)
        )
    return "\n".join(cards_html)


def get_toolbar_theme_toggle_html(dark_mode: bool = True, chat_msg_count: int = 1) -> str:
    """Return JS/HTML snippet that mounts a morphing Sun <-> Moon button directly left of 'Deploy' and handles chat auto-scroll."""
    state_cls = "is-dark" if dark_mode else "is-light"
    label_txt = "Dark" if dark_mode else "Light"
    return f"""
    <script>
    (function() {{
      const doc = window.parent.document;
      if (!doc) return;

      // 1. Inject CSS for the toolbar Sun <-> Moon morphing button once
      let styleEl = doc.getElementById('te-toolbar-toggle-style');
      if (!styleEl) {{
        styleEl = doc.createElement('style');
        styleEl.id = 'te-toolbar-toggle-style';
        styleEl.textContent = `
          #te-toolbar-theme-btn {{
            display: inline-flex;
            align-items: center;
            align-self: center;
            gap: 7px;
            height: 34px;
            min-height: 34px;
            padding: 0 12px 0 8px;
            margin: 0 6px 0 0;
            border-radius: 999px;
            cursor: pointer;
            user-select: none;
            font-family: 'Fira Sans Extra Condensed', sans-serif;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
            position: relative;
            z-index: 1000010;
          }}
          #te-toolbar-theme-btn.is-dark {{
            background: linear-gradient(135deg, #0B1E30 0%, #143552 100%);
            color: #E6F6FA;
            border: 1.5px solid #46BEDC;
            box-shadow: 0 0 12px rgba(70, 190, 220, 0.35);
          }}
          #te-toolbar-theme-btn.is-light {{
            background: linear-gradient(135deg, #FFF5E2 0%, #EAF7FB 100%);
            color: #0F4B73;
            border: 1.5px solid #F59B00;
            box-shadow: 0 2px 10px rgba(245, 155, 0, 0.3);
          }}
          #te-toolbar-theme-btn:hover {{
            transform: translateY(-1px) scale(1.04);
            border-color: #F59B00;
            box-shadow: 0 0 16px rgba(245, 155, 0, 0.55);
          }}
          #te-toolbar-theme-btn .te-morph-icon {{
            width: 22px;
            height: 22px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
          }}
          #te-toolbar-theme-btn.is-dark .te-morph-icon {{
            background: #0F4B73;
            color: #46BEDC;
            transform: rotate(360deg);
          }}
          #te-toolbar-theme-btn.is-light .te-morph-icon {{
            background: #F59B00;
            color: #FFFFFF;
            transform: rotate(0deg);
          }}
          /* SVG Mask Morphing: Sun <-> Crescent Moon */
          #te-toolbar-theme-btn svg.te-sun-moon-svg {{
            width: 15px;
            height: 15px;
            overflow: visible;
            transition: transform 0.55s cubic-bezier(0.34, 1.56, 0.64, 1);
          }}
          #te-toolbar-theme-btn.is-dark svg.te-sun-moon-svg {{
            transform: rotate(40deg);
          }}
          #te-toolbar-theme-btn.is-light svg.te-sun-moon-svg {{
            transform: rotate(90deg);
          }}
          #te-toolbar-theme-btn .te-core {{
            transition: r 0.45s cubic-bezier(0.34, 1.56, 0.64, 1), fill 0.35s ease;
          }}
          #te-toolbar-theme-btn.is-dark .te-core {{
            fill: #46BEDC;
          }}
          #te-toolbar-theme-btn.is-light .te-core {{
            fill: #FFFFFF;
          }}
          #te-toolbar-theme-btn .te-cutout {{
            transition: cx 0.5s cubic-bezier(0.34, 1.56, 0.64, 1), cy 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
          }}
          #te-toolbar-theme-btn .te-rays {{
            transform-origin: center center;
            transition: opacity 0.35s ease, transform 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
          }}
          #te-toolbar-theme-btn.is-dark .te-rays {{
            opacity: 0;
            transform: scale(0.2) rotate(-60deg);
          }}
          #te-toolbar-theme-btn.is-light .te-rays {{
            opacity: 1;
            transform: scale(1) rotate(0deg);
          }}
          /* Hide the underlying native trigger button once toolbar button is mounted */
          body.te-toolbar-mounted div.st-key-te_theme_toggle_btn {{
            opacity: 0 !important;
            pointer-events: none !important;
            width: 1px !important;
            height: 1px !important;
            overflow: hidden !important;
          }}
        `;
        doc.head.appendChild(styleEl);
      }}

      function mountButton() {{
        const deployBtn =
          doc.querySelector('[data-testid="stAppDeployButton"]') ||
          doc.querySelector('.stAppDeployButton') ||
          doc.querySelector('[data-testid="stToolbarActions"]');
        const toolbar =
          doc.querySelector('[data-testid="stToolbar"]') ||
          doc.querySelector('header[data-testid="stHeader"]');

        if (!toolbar && !deployBtn) return false;

        let btn = doc.getElementById('te-toolbar-theme-btn');
        if (!btn) {{
          btn = doc.createElement('button');
          btn.id = 'te-toolbar-theme-btn';
          btn.type = 'button';
          btn.title = 'Switch between Dark Mode (Moon) and Light Mode (Sun)';
          btn.innerHTML = `
            <span class="te-morph-icon">
              <svg class="te-sun-moon-svg" viewBox="0 0 24 24">
                <mask id="te-moon-cutout-mask">
                  <rect x="0" y="0" width="24" height="24" fill="white"/>
                  <circle class="te-cutout" cx="{16 if dark_mode else 26}" cy="{7 if dark_mode else 0}" r="6.5" fill="black"/>
                </mask>
                <circle class="te-core" cx="12" cy="12" r="{8.2 if dark_mode else 5}" mask="url(#te-moon-cutout-mask)"/>
                <g class="te-rays" stroke="#FFFFFF" stroke-width="2.2" stroke-linecap="round">
                  <line x1="12" y1="1.5" x2="12" y2="3.8"/>
                  <line x1="12" y1="20.2" x2="12" y2="22.5"/>
                  <line x1="4.5" y1="4.5" x2="6.2" y2="6.2"/>
                  <line x1="17.8" y1="17.8" x2="19.5" y2="19.5"/>
                  <line x1="1.5" y1="12" x2="3.8" y2="12"/>
                  <line x1="20.2" y1="12" x2="22.5" y2="12"/>
                  <line x1="4.5" y1="19.5" x2="6.2" y2="17.8"/>
                  <line x1="17.8" y1="6.2" x2="19.5" y2="4.5"/>
                </g>
              </svg>
            </span>
            <span class="te-mode-text">{label_txt}</span>
          `;

          if (deployBtn && deployBtn.parentNode) {{
            deployBtn.parentNode.insertBefore(btn, deployBtn);
          }} else if (toolbar) {{
            toolbar.prepend(btn);
          }}
        }}

        // Always attach the click handler on window.parent so it never dies when iframes re-render
        window.parent.__teToggleTheme = function() {{
          const currentBtn = doc.getElementById('te-toolbar-theme-btn');
          if (!currentBtn) return;
          const isCurrentlyDark = currentBtn.classList.contains('is-dark');
          const nextDark = !isCurrentlyDark;
          currentBtn.className = nextDark ? 'is-dark' : 'is-light';
          const cutout = currentBtn.querySelector('.te-cutout');
          const core = currentBtn.querySelector('.te-core');
          const txt = currentBtn.querySelector('.te-mode-text');
          if (cutout) {{
            cutout.setAttribute('cx', nextDark ? '16' : '26');
            cutout.setAttribute('cy', nextDark ? '7' : '0');
          }}
          if (core) {{
            core.setAttribute('r', nextDark ? '8.2' : '5');
          }}
          if (txt) {{
            txt.textContent = nextDark ? 'Dark' : 'Light';
          }}
          const trigger = doc.querySelector('div.st-key-te_theme_toggle_btn button');
          if (trigger) {{
            trigger.click();
          }}
        }};
        btn.onclick = window.parent.__teToggleTheme;

        // Sync visual state with Python session_state
        btn.className = '{state_cls}';
        const cutout = btn.querySelector('.te-cutout');
        const core = btn.querySelector('.te-core');
        const txt = btn.querySelector('.te-mode-text');
        if (cutout) {{
          cutout.setAttribute('cx', '{16 if dark_mode else 26}');
          cutout.setAttribute('cy', '{7 if dark_mode else 0}');
        }}
        if (core) {{
          core.setAttribute('r', '{8.2 if dark_mode else 5}');
        }}
        if (txt) {{
          txt.textContent = '{label_txt}';
        }}
        doc.body.classList.add('te-toolbar-mounted');
        return true;
      }}

      if (!mountButton()) {{
        setTimeout(mountButton, 120);
        setTimeout(mountButton, 400);
      }}

      // 2. Prevent Google Password Manager / Autofill on the API Key field
      function hardenApiKeyInput() {{
        const apiInput = doc.querySelector('div.st-key-gemini_api_key_input input');
        if (apiInput) {{
          apiInput.setAttribute('autocomplete', 'off');
          apiInput.setAttribute('autocorrect', 'off');
          apiInput.setAttribute('autocapitalize', 'off');
          apiInput.setAttribute('spellcheck', 'false');
          apiInput.setAttribute('data-lpignore', 'true');
          apiInput.setAttribute('data-1p-ignore', 'true');
          apiInput.setAttribute('data-form-type', 'other');
        }}
      }}
      hardenApiKeyInput();
      setTimeout(hardenApiKeyInput, 250);

      // 3. Auto-scroll AI Assistant Chat container to bottom when new messages arrive
      function scrollChatToBottom() {{
        const msgCount = {int(chat_msg_count)};
        const anchor = doc.getElementById('te-chat-bottom-anchor');
        if (!anchor) return;
        if (msgCount > 1 && window.parent.__teLastMsgCount !== msgCount) {{
          window.parent.__teLastMsgCount = msgCount;
          // Find the scrollable container around the chat messages
          let el = anchor.parentElement;
          while (el && el !== doc.body) {{
            const style = window.parent.getComputedStyle(el);
            if ((style.overflowY === 'auto' || style.overflowY === 'scroll') && el.scrollHeight > el.clientHeight) {{
              el.scrollTo({{ top: el.scrollHeight, behavior: 'smooth' }});
              break;
            }}
            el = el.parentElement;
          }}
          anchor.scrollIntoView({{ behavior: 'smooth', block: 'end' }});
        }} else if (msgCount <= 1) {{
          window.parent.__teLastMsgCount = msgCount;
        }}
      }}
      setTimeout(scrollChatToBottom, 60);
      setTimeout(scrollChatToBottom, 220);
    }})();
    </script>
    """


def decompose_step_prediction_and_uncertainty(
    step_row: dict,
    origin_df,
) -> dict:
    """Decompose a single step's P50 prediction and P10-P90 uncertainty width into concrete €/MWh causes."""
    step = int(step_row["step"])
    spot = float(step_row["spot_price"])
    p50 = float(step_row["p50"])
    p10 = float(step_row["p10"])
    p90 = float(step_row["p90"])
    spread = p50 - spot
    band_width = max(0.1, p90 - p10)
    up_width = max(0.0, p90 - p50)
    dn_width = max(0.0, p50 - p10)

    # 1. Decompose spread (P50 - Spot) across the top 3 TreeSHAP drivers for this step
    raw_drivers = [d.strip() for d in str(step_row.get("top_drivers", "")).split(";") if d.strip()][:3]
    while len(raw_drivers) < 3:
        raw_drivers.append("residual_load_forecast")
    friendly_drivers = [format_driver_name(d) for d in raw_drivers]

    # Rank weights (Rank 1 = 50%, Rank 2 = 30%, Rank 3 = 20% of the predicted spread ΔP)
    w_drv = [0.50, 0.30, 0.20]
    drv_eur = [round(spread * w, 2) for w in w_drv]
    # Ensure exact sum == spread
    drv_eur[2] = round(spread - drv_eur[0] - drv_eur[1], 2)

    # 2. Decompose Uncertainty Band Width (P90 - P10) into 4 intuitive €/MWh sources
    #    a) Base Grid Noise (minimum uncertainty at Step 1)
    min_band = float((origin_df["p90"] - origin_df["p10"]).min())
    base_noise_eur = round(min(band_width * 0.42, min_band * 0.65), 1)

    #    b) Horizon Lead Time (+15m * step): further into the future = higher weather/load uncertainty
    lead_frac = 0.08 + 0.28 * ((step - 1) / 39.0)
    lead_eur = round(band_width * lead_frac, 1)

    #    c) Weather & Load Ramp Volatility (wind / residual load swings around this step)
    wind_val = float(step_row.get("wind_da", 1500.0))
    cons_val = float(step_row.get("cons_da", 9000.0))
    wind_std = float(origin_df["wind_da"].std()) if "wind_da" in origin_df else 200.0
    ramp_frac = min(0.28, 0.12 + 0.16 * min(1.0, wind_std / 400.0))
    ramp_eur = round(band_width * ramp_frac, 1)

    #    d) Balancing Reserve Tail Asymmetry (remaining width from mFRR Up/Down spike risk)
    asym_eur = round(max(0.5, band_width - base_noise_eur - lead_eur - ramp_eur), 1)

    # Physical explanation sentences
    if spread >= 5.0:
        price_reason = (
            f"The model predicts an <strong>upward spread of {spread:+.1f} €/MWh above Spot</strong> "
            f"(Spot {spot:.1f} € → P50 {p50:.1f} €) because grid demand/stress pushes Finland toward "
            f"<strong>up-regulation ({float(step_row['p_up']):.0%} probability)</strong>, where expensive mFRR reserve bids set the price."
        )
    elif spread <= -5.0:
        price_reason = (
            f"The model predicts a <strong>downward spread of {spread:+.1f} €/MWh below Spot</strong> "
            f"(Spot {spot:.1f} € → P50 {p50:.1f} €) because power surplus (e.g. wind generation {wind_val:,.0f} MW vs. load {cons_val:,.0f} MW) "
            f"pushes the grid toward <strong>down-regulation ({float(step_row['p_down']):.0%} probability)</strong>."
        )
    else:
        price_reason = (
            f"The model keeps <strong>P50 ({p50:.1f} €/MWh) close to the Day-Ahead Spot Price ({spot:.1f} €/MWh)</strong> "
            f"(Spread {spread:+.1f} €), as upward ({float(step_row['p_up']):.0%}) and downward ({float(step_row['p_down']):.0%}) pressures are nearly balanced."
        )

    lead_hours = step * 0.25
    if up_width > dn_width * 1.25:
        skew_text = (
            f"<strong>Upward-skewed risk (+{up_width:.1f} € up vs. −{dn_width:.1f} € down):</strong> "
            f"Upward mFRR price spikes are steeper than downward price drops."
        )
    elif dn_width > up_width * 1.25:
        skew_text = (
            f"<strong>Downward-skewed risk (−{dn_width:.1f} € down vs. +{up_width:.1f} € up):</strong> "
            f"High surplus / negative price risk pulls P10 downward."
        )
    else:
        skew_text = (
            f"<strong>Symmetric uncertainty (+{up_width:.1f} € up / −{dn_width:.1f} € down)</strong> around P50."
        )

    return {
        "step": step,
        "lead_hours": lead_hours,
        "deliv_str": step_row["delivery_start_utc"].strftime("%H:%M UTC"),
        "spot": spot,
        "p50": p50,
        "p10": p10,
        "p90": p90,
        "spread": spread,
        "band_width": band_width,
        "up_width": up_width,
        "dn_width": dn_width,
        "drivers": list(zip(friendly_drivers, drv_eur)),
        "unc_components": [
            ("1. Base Balancing Volatility (Grid Noise)", base_noise_eur),
            (f"2. Horizon Lead Time (+{lead_hours:.2f}h / Step {step})", lead_eur),
            ("3. Wind & Residual Load Ramp Uncertainty", ramp_eur),
            ("4. Asymmetric mFRR Spike Tail Risk", asym_eur),
        ],
        "price_reason": price_reason,
        "skew_text": skew_text,
    }


def render_xai_explanation_cards(info: dict) -> str:
    """Render two Apple-style translucent material cards explaining WHY P50 and WHY P10-P90 uncertainty look the way they do."""
    max_abs_drv = max([abs(v) for _, v in info["drivers"]] + [1.0])
    drv_rows_html = ""
    for rank, (d_name, d_val) in enumerate(info["drivers"], start=1):
        badge_col = "var(--te-orange)" if d_val >= 0 else "var(--te-turquoise)"
        pct_w = int(round(min(100.0, max(12.0, (abs(d_val) / max_abs_drv) * 100.0))))
        drv_rows_html += (
            f'<div class="te-xai-row">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.24rem;">'
            f'<span style="font-size:0.84rem; font-weight:600; color:var(--te-text); letter-spacing:-0.01em;">#{rank} · {html.escape(d_name)}</span>'
            f'<span style="font-variant-numeric:tabular-nums; font-weight:700; font-size:0.9rem; color:{badge_col};">{d_val:+.1f} €/MWh</span>'
            f'</div>'
            f'<div style="width:100%; height:6px; background:var(--te-bar-track); border-radius:99px; overflow:hidden;">'
            f'<div class="te-xai-bar-fill" style="width:{pct_w}%; background:{badge_col};"></div>'
            f'</div>'
            f'</div>'
        )

    max_unc = max([v for _, v in info["unc_components"]] + [1.0])
    unc_rows_html = ""
    for u_name, u_val in info["unc_components"]:
        pct_u = int(round(min(100.0, max(12.0, (u_val / max_unc) * 100.0))))
        unc_rows_html += (
            f'<div class="te-xai-row">'
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.22rem;">'
            f'<span style="font-size:0.82rem; font-weight:500; color:var(--te-text); letter-spacing:-0.008em;">{html.escape(u_name)}</span>'
            f'<span style="font-variant-numeric:tabular-nums; font-weight:700; font-size:0.88rem; color:var(--te-turquoise);">±{u_val:.1f} €</span>'
            f'</div>'
            f'<div style="width:100%; height:5px; background:var(--te-bar-track); border-radius:99px; overflow:hidden;">'
            f'<div class="te-xai-bar-fill" style="width:{pct_u}%; background:linear-gradient(90deg, var(--te-turquoise) 0%, #B4DCE6 100%);"></div>'
            f'</div>'
            f'</div>'
        )

    p50_pos_pct = int(round(min(92.0, max(8.0, (info["dn_width"] / max(info["band_width"], 0.1)) * 100.0))))
    spread_col = "var(--te-orange)" if info["spread"] >= 0 else "var(--te-turquoise)"

    return _compact_html(f"""
    <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap:1.05rem; margin:0.65rem 0 1.15rem 0;">
      <div class="te-xai-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.65rem; gap:0.5rem;">
          <span style="font-family:var(--te-font-sans); font-weight:700; font-size:1.06rem; letter-spacing:-0.022em; color:var(--te-heading-color);">
            💡 1. Why is P50 = {info['p50']:.1f} €/MWh at {info['deliv_str']} (Step {info['step']})?
          </span>
          <span style="font-variant-numeric:tabular-nums; background:var(--te-orange); color:#FFFFFF; padding:0.2rem 0.68rem; border-radius:99px; font-size:0.75rem; font-weight:700; white-space:nowrap;">
            Spot {info['spot']:.1f} € {info['spread']:+.1f} €
          </span>
        </div>
        <div style="display:grid; grid-template-columns: 1fr auto 1fr auto 1fr; align-items:center; gap:0.35rem; background:var(--te-bg-elevated); border:1px solid var(--te-border-subtle); border-radius:12px; padding:0.55rem 0.75rem; margin-bottom:0.7rem; text-align:center;">
          <div>
            <div style="font-size:0.66rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--te-text-muted); font-weight:700;">Spot Anchor</div>
            <div style="font-variant-numeric:tabular-nums; font-size:1.1rem; font-weight:700; letter-spacing:-0.025em; color:var(--te-heading-color);">{info['spot']:.1f} €</div>
          </div>
          <div style="font-weight:800; color:var(--te-orange); font-size:1.1rem;">+</div>
          <div>
            <div style="font-size:0.66rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--te-text-muted); font-weight:700;">AI Spread (ΔP)</div>
            <div style="font-variant-numeric:tabular-nums; font-size:1.1rem; font-weight:700; letter-spacing:-0.025em; color:{spread_col};">{info['spread']:+.1f} €</div>
          </div>
          <div style="font-weight:800; color:var(--te-turquoise); font-size:1.1rem;">=</div>
          <div>
            <div style="font-size:0.66rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--te-text-muted); font-weight:700;">P50 Forecast</div>
            <div style="font-variant-numeric:tabular-nums; font-size:1.16rem; font-weight:800; letter-spacing:-0.028em; color:var(--te-lime);">{info['p50']:.1f} €</div>
          </div>
        </div>
        <p style="font-size:0.85rem; line-height:1.5; color:var(--te-text); margin:0 0 0.7rem 0;">
          {info['price_reason']}
        </p>
        <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--te-text-muted); font-weight:700; margin-bottom:0.38rem;">
          Top 3 Drivers Pushing the Price ({info['spread']:+.1f} €/MWh):
        </div>
        {drv_rows_html}
        <div class="te-xai-note">ⓘ Driver <strong>ranking</strong> comes from the model (top_drivers). The € split (50/30/20 %) is an <strong>estimate</strong>, not exact SHAP values.</div>
      </div>
      <div class="te-xai-card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.65rem; gap:0.5rem;">
          <span style="font-family:var(--te-font-sans); font-weight:700; font-size:1.06rem; letter-spacing:-0.022em; color:var(--te-heading-color);">
            📏 2. Why is the Uncertainty (P10–P90) {info['band_width']:.1f} €/MWh Wide?
          </span>
          <span style="font-variant-numeric:tabular-nums; background:var(--te-delta-bg); color:var(--te-turquoise); border:1px solid var(--te-border); padding:0.18rem 0.65rem; border-radius:99px; font-size:0.75rem; font-weight:700; white-space:nowrap;">
            80% Confidence Band
          </span>
        </div>
        <div style="background:var(--te-bg-elevated); border:1px solid var(--te-border-subtle); border-radius:12px; padding:0.6rem 0.85rem; margin-bottom:0.7rem;">
          <div style="display:flex; justify-content:space-between; font-variant-numeric:tabular-nums; font-size:0.78rem; font-weight:700; margin-bottom:0.32rem;">
            <span style="color:var(--te-turquoise);">P10: {info['p10']:.1f} € (−{info['dn_width']:.1f} €)</span>
            <span style="color:var(--te-heading-color);">P50: {info['p50']:.1f} €</span>
            <span style="color:var(--te-berry);">P90: {info['p90']:.1f} € (+{info['up_width']:.1f} €)</span>
          </div>
          <div style="position:relative; width:100%; height:8px; background:linear-gradient(90deg, rgba(90,200,250,0.45) 0%, rgba(255,159,10,0.65) 50%, rgba(255,55,95,0.75) 100%); border-radius:99px;">
            <div style="position:absolute; left:{p50_pos_pct}%; top:-3px; width:14px; height:14px; border-radius:50%; background:#FFFFFF; border:3px solid var(--te-orange); transform:translateX(-50%); box-shadow:0 2px 6px rgba(0,0,0,0.35);"></div>
          </div>
        </div>
        <p style="font-size:0.85rem; line-height:1.5; color:var(--te-text); margin:0 0 0.7rem 0;">
          {info['skew_text']}
        </p>
        <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--te-text-muted); font-weight:700; margin-bottom:0.38rem;">
          4 Sources Adding Up to {info['band_width']:.1f} €/MWh Uncertainty Width:
        </div>
        {unc_rows_html}
        <div class="te-xai-note">ⓘ The <strong>total</strong> width (P90−P10) is the model output. Its split into sources is an <strong>estimated</strong> allocation.</div>
      </div>
    </div>
    """)


