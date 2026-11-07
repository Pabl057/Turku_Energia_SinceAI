"""Bilingual (English / German) knowledge base for the Turku Energia AI Assistant.

Every UI element of the Situational Desk (header, sidebar, KPI cards, alert banners,
all 4 tabs and their charts/tables, AI assistant controls) has one entry with
English + German keywords and English + German answer templates. The assistant
answers in the language of the question (EN -> EN, DE -> DE).

Templates use ``str.format_map`` placeholders filled with live values of the
currently selected forecast origin (no future data is used - Rule #1 safe).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Language detection (score-based, robust against shared EN/DE cognates)
# ---------------------------------------------------------------------------
_DE_MARKERS = {
    "ist", "sind", "das", "die", "der", "den", "dem", "des", "ein", "eine", "einen", "einer",
    "wie", "wo", "wann", "warum", "wieso", "weshalb", "welche", "welcher", "welches", "wofür",
    "zeig", "zeige", "erkläre", "erklär", "erklären", "bedeutet", "bedeuten", "funktioniert",
    "gibt", "mir", "mich", "ich", "du", "kannst", "bitte", "nicht", "und", "oder", "für",
    "über", "auf", "mit", "im", "beim", "zum", "zur", "wird", "werden", "hat", "haben",
    "stunde", "schritt", "preis", "spotpreis", "treiber", "karte", "karten", "warnung",
    "warnungen", "zeitraum", "kurve", "schalter", "knopf", "zahnrad", "einstellungen",
    "tabelle", "wasserfall", "unsicherheit", "kalibrierung", "signifikanz", "ausgleichspreis",
    "aktuell", "genau", "heißt", "soll", "sehe", "siehst", "man", "uns", "unser", "unsere",
    "noch", "auch", "dieser", "diese", "dieses", "vorstunde", "regelrichtung", "seite",
    "webseite", "datensatz", "zeigt", "macht", "passiert", "drücke", "klicke", "klick",
}
_EN_MARKERS = {
    "what", "how", "why", "which", "when", "where", "who", "is", "are", "the", "does", "do",
    "explain", "show", "shows", "mean", "means", "meaning", "can", "you", "of", "for", "and",
    "this", "that", "these", "price", "tell", "me", "about", "work", "works", "happens",
    "button", "card", "cards", "table", "chart", "graph", "my", "i", "it", "there", "please",
    "should", "would", "if", "click", "press", "current", "currently", "website", "page",
}


def is_german_query(question: str) -> bool:
    """Return True if the question is (predominantly) written in German."""
    q = question.lower()
    tokens = re.findall(r"[a-zäöüß]+", q)
    de_score = sum(1 for t in tokens if t in _DE_MARKERS)
    en_score = sum(1 for t in tokens if t in _EN_MARKERS)
    if any(ch in q for ch in ("ä", "ö", "ü", "ß")):
        de_score += 2
    # "was" is German ("Was ist …") but also English ("What was …")
    if tokens and tokens[0] == "was":
        de_score += 1
    return de_score > en_score


# ---------------------------------------------------------------------------
# Knowledge base entries
# ---------------------------------------------------------------------------
# Each entry: keywords (lowercase substrings, EN + DE), title/body in EN and DE.
KB_ENTRIES: list[dict] = [
    # ======================= 1. HEADER & TOP TOOLBAR =======================
    {
        "id": "dark_mode",
        "keywords": ["dark mode", "light mode", "dark / light", "darkmode", "lightmode", "dunkelmodus",
                     "hellmodus", "dunkler modus", "heller modus", "theme"],
        "en": ("🌙 Dark / ☀️ Light Mode Button",
               "The toggle in the **top-right toolbar** switches the whole dashboard between:\n"
               "* **Control-Room Dark Mode** (`#07131E` navy background) – optimised for 24/7 control rooms and low glare.\n"
               "* **Web Light Mode** (`#FFFFFF`) – classic office/web look with sharp borders and soft shadows.\n\n"
               "All charts, KPI cards, alert banners and the AI Assistant window re-colour automatically. The forecast data does not change."),
        "de": ("🌙 Dark / ☀️ Light Mode Button",
               "Der Schalter **oben rechts in der Toolbar** wechselt das gesamte Dashboard zwischen:\n"
               "* **Control-Room Dark Mode** (`#07131E` Navy-Hintergrund) – optimiert für Leitwarten im 24/7-Betrieb, blendfrei.\n"
               "* **Web Light Mode** (`#FFFFFF`) – klassischer Büro-/Web-Look mit klaren Rahmen und dezenten Schatten.\n\n"
               "Alle Graphen, KPI-Karten, Warnbanner und das KI-Fenster passen ihre Farben automatisch an. Die Prognosedaten ändern sich nicht."),
    },
    {
        "id": "sidebar_toggle",
        "keywords": ["sidebar collapse", "collapse the sidebar", "expand the sidebar", "sidebar ein", "sidebar aus",
                     "sidebar einklappen", "sidebar ausklappen", "<<", ">>", "hide the sidebar", "seitenleiste"],
        "en": ("Sidebar Collapse / Expand Button (`<<` / `>>`)",
               "The small arrow button at the top of the left sidebar **collapses** it (`<<`) to give the charts the full screen width, "
               "and **expands** it again (`>>`). In Light Mode the icon is dark grey (`#2C2C2E`) with a thin border so it is always visible."),
        "de": ("Sidebar Ein-/Ausklapp-Button (`<<` / `>>`)",
               "Der kleine Pfeil-Button oben in der linken Sidebar **klappt sie ein** (`<<`), damit die Graphen die volle Bildschirmbreite nutzen, "
               "und **klappt sie wieder aus** (`>>`). Im Light Mode ist das Icon dunkelgrau (`#2C2C2E`) mit dünnem Rahmen und dadurch immer sichtbar."),
    },
    {
        "id": "three_dots",
        "keywords": ["three dots", "3 dots", "drei punkte", "deploy", "main menu", "hauptmenü", "⋮"],
        "en": ("Top-Right Menu (⋮) & Deploy",
               "The **three dots (⋮)** next to *Deploy* open Streamlit's main menu: *Rerun*, *Settings*, *Print*, *Record a screencast* and *About*. "
               "*Deploy* publishes the app to Streamlit Community Cloud. Neither affects the forecast itself."),
        "de": ("Menü oben rechts (⋮) & Deploy",
               "Die **drei Punkte (⋮)** neben *Deploy* öffnen das Streamlit-Hauptmenü: *Rerun*, *Settings*, *Print*, *Record a screencast* und *About*. "
               "*Deploy* veröffentlicht die App in der Streamlit Community Cloud. Beides beeinflusst die Prognose nicht."),
    },
    {
        "id": "hero",
        "keywords": ["hero", "header banner", "top banner", "kopfbereich", "kopfzeile", "überschrift", "title banner", "headline"],
        "en": ("Hero Banner (Page Header)",
               "The banner at the top shows the **Turku Energia** branding, the title of the *Imbalance Price Situational Desk*, the slogan "
               "*„Olemme lähellä, näemme kauas.“* and the currently selected **forecast origin `{origin}`** with its data cutoff `{cutoff}`."),
        "de": ("Hero-Banner (Seitenkopf)",
               "Das Banner ganz oben zeigt das **Turku Energia** Branding, den Titel des *Imbalance Price Situational Desk*, den Slogan "
               "*„Olemme lähellä, näemme kauas.“* sowie den aktuell gewählten **Forecast Origin `{origin}`** mit Daten-Cutoff `{cutoff}`."),
    },
    {
        "id": "slogan",
        "keywords": ["slogan", "olemme", "lähellä", "naemme", "näemme", "kauas", "motto"],
        "en": ("Slogan „Olemme lähellä, näemme kauas.“",
               "Finnish for **“We are close, we see far.”** – Turku Energia's brand promise. It fits the dashboard: close to the grid "
               "(15-minute resolution, live telemetry) and seeing far ahead (10-hour probabilistic forecast)."),
        "de": ("Slogan „Olemme lähellä, näemme kauas.“",
               "Finnisch für **„Wir sind nah, wir sehen weit.“** – das Markenversprechen von Turku Energia. Passt zum Dashboard: nah am Netz "
               "(15-Minuten-Auflösung, Echtzeit-Telemetrie) und weit vorausschauend (10-Stunden-Wahrscheinlichkeitsprognose)."),
    },
    # ======================= 2. LEFT SIDEBAR =======================
    {
        "id": "prev_next",
        "keywords": ["previous button", "next button", "previous / next", "previous/next", "◀", "▶", "zurück-button",
                     "weiter-button", "vor und zurück", "vor/zurück", "nächste stunde", "vorherige stunde", "next hour", "previous hour button"],
        "en": ("◀ Previous / Next ▶ Buttons",
               "They move the **forecast origin `t` exactly one step** within the active Quick Filter list (current position `{pos}`). "
               "With *All 673 Origins* this equals ±1 hour; with *Spike Alerts* or *Top 20* it jumps to the previous/next origin in that filtered list. "
               "All charts, KPI cards, alerts and the AI context update instantly."),
        "de": ("◀ Previous / Next ▶ Buttons",
               "Sie verschieben den **Forecast Origin `t` um genau einen Eintrag** innerhalb der aktiven Quick-Filter-Liste (aktuelle Position `{pos}`). "
               "Bei *All 673 Origins* entspricht das ±1 Stunde; bei *Spike Alerts* oder *Top 20* springt man zum vorherigen/nächsten Origin dieser gefilterten Liste. "
               "Alle Graphen, KPI-Karten, Warnungen und der KI-Kontext aktualisieren sich sofort."),
    },
    {
        "id": "show_ai_toggle",
        "keywords": ["show ai explainability", "ai explainability chat", "hide the ai", "hide the chat", "show the chat",
                     "chat ausblenden", "chat einblenden", "ki ausblenden", "ki einblenden", "ai panel"],
        "en": ("🤖 Show AI Explainability Chat (Toggle)",
               "Shows or hides this **AI Assistant column** on the right. When off, the charts use the full page width – useful for presentations."),
        "de": ("🤖 Show AI Explainability Chat (Schalter)",
               "Blendet diese **KI-Assistenten-Spalte** rechts ein oder aus. Ist er aus, nutzen die Graphen die volle Seitenbreite – praktisch für Präsentationen."),
    },
    {
        "id": "hindsight",
        "keywords": ["hindsight", "overlay actual", "actual prices", "realized price", "realised price", "echte preise",
                     "tatsächliche preise", "ist-preise", "grüne linie", "green dotted", "ground truth"],
        "en": ("🎯 Overlay Actual Prices (Hindsight)",
               "Adds the **realized imbalance price** as a green dotted line to the fan chart so you can compare forecast vs. reality. "
               "Ground truth exists only for the **Validation Period (7 Sep – 4 Oct 2026)**; in the Test Period there are no actuals yet. "
               "This is a *display-only* overlay – it is never fed into the model (Rule #1)."),
        "de": ("🎯 Overlay Actual Prices (Hindsight)",
               "Blendet den **tatsächlich eingetretenen Ausgleichspreis** als grüne gepunktete Linie im Fan-Chart ein, um Prognose und Realität zu vergleichen. "
               "Echte Werte gibt es nur für die **Validation Period (7. Sep – 4. Okt 2026)**; in der Test Period liegen noch keine Ist-Werte vor. "
               "Die Überlagerung dient nur der Anzeige – sie fließt nie ins Modell ein (Regel #1)."),
    },
    # ======================= 3. KPI CARDS =======================
    {
        "id": "kpi_overview",
        "keywords": ["kpi", "metric card", "kennzahlen", "kennzahl", "the 5 cards", "five cards", "5 karten", "fünf karten", "badge", "tag under"],
        "en": ("The 5 KPI Cards",
               "1. **10h Forecast Mean (P50):** `{p50_mean:.2f} €/MWh` average median imbalance price over the next 40 steps.\n"
               "2. **High-Price Spike (>Spot+50€):** max probability `{max_p_hi:.1%}`.\n"
               "3. **Low-Price Spike (<Spot−50€):** max probability `{max_p_lo:.1%}`.\n"
               "4. **Dominant Balancing Dir. (mFRR):** **{dominant_dir}**.\n"
               "5. **80% Uncertainty Band (P10–P90):** average width `{mean_band:.1f} €/MWh`.\n\n"
               "The **sharp rectangular tags** under each number give context (e.g. spread vs. Spot, alert status): orange = warning (≥ 30 %), green = OK."),
        "de": ("Die 5 KPI-Karten",
               "1. **10h Forecast Mean (P50):** Ø `{p50_mean:.2f} €/MWh` Median-Ausgleichspreis über die nächsten 40 Schritte.\n"
               "2. **High-Price Spike (>Spot+50€):** maximale Wahrscheinlichkeit `{max_p_hi:.1%}`.\n"
               "3. **Low-Price Spike (<Spot−50€):** maximale Wahrscheinlichkeit `{max_p_lo:.1%}`.\n"
               "4. **Dominant Balancing Dir. (mFRR):** **{dominant_dir}**.\n"
               "5. **80% Uncertainty Band (P10–P90):** durchschnittliche Breite `{mean_band:.1f} €/MWh`.\n\n"
               "Die **eckigen Tags** unter jeder Zahl geben Kontext (z. B. Abstand zum Spot, Warnstatus): orange = Warnung (≥ 30 %), grün = OK."),
    },
    {
        "id": "kpi_mean",
        "keywords": ["forecast mean", "10h mean", "mean card", "mittelwert-karte", "prognose-mittel", "durchschnittspreis"],
        "en": ("KPI: 10h Forecast Mean (P50)",
               "Average of the 40 P50 median forecasts for origin `{origin}`: **{p50_mean:.2f} €/MWh** vs. mean Spot `{spot_mean:.2f} €/MWh` "
               "(spread **{spread:+.2f} €/MWh**). A positive spread hints at grid deficit, a negative one at surplus."),
        "de": ("KPI: 10h Forecast Mean (P50)",
               "Durchschnitt der 40 P50-Medianprognosen für Origin `{origin}`: **{p50_mean:.2f} €/MWh** vs. Spot-Mittel `{spot_mean:.2f} €/MWh` "
               "(Spread **{spread:+.2f} €/MWh**). Positiver Spread deutet auf Stromdefizit, negativer auf Überschuss hin."),
    },
    {
        "id": "kpi_high_spike",
        "keywords": ["high-price spike card", "high spike card", "high-price spike kpi", "high spike kpi", "hochpreis-karte",
                     "hochpreis-spike-karte", "high-price spike (>spot"],
        "en": ("KPI: High-Price Spike (>Spot+50€)",
               "Highest probability across the 40 steps that the imbalance price exceeds **Spot + 50 €/MWh**: **{max_p_hi:.1%}** (at `{hi_time}`). "
               "From **30 %** the tag turns into a warning and the *High-Price Spike Alert* banner appears."),
        "de": ("KPI: High-Price Spike (>Spot+50€)",
               "Höchste Wahrscheinlichkeit über die 40 Schritte, dass der Ausgleichspreis **Spot + 50 €/MWh** übersteigt: **{max_p_hi:.1%}** (um `{hi_time}`). "
               "Ab **30 %** wird der Tag zur Warnung und das Banner *High-Price Spike Alert* erscheint."),
    },
    {
        "id": "kpi_low_spike",
        "keywords": ["low-price spike card", "low spike card", "low-price spike kpi", "low spike kpi", "tiefpreis-karte",
                     "tiefpreis-spike-karte", "low-price spike (<spot"],
        "en": ("KPI: Low-Price Spike (<Spot−50€)",
               "Highest probability that the imbalance price falls below **Spot − 50 €/MWh** (often negative prices on windy nights): **{max_p_lo:.1%}** (at `{lo_time}`). "
               "From **30 %** the *Low / Negative Price Alert* banner appears."),
        "de": ("KPI: Low-Price Spike (<Spot−50€)",
               "Höchste Wahrscheinlichkeit, dass der Ausgleichspreis unter **Spot − 50 €/MWh** fällt (oft negative Preise in windreichen Nächten): **{max_p_lo:.1%}** (um `{lo_time}`). "
               "Ab **30 %** erscheint das Banner *Low / Negative Price Alert*."),
    },
    {
        "id": "kpi_direction",
        "keywords": ["dominant balancing", "dominant direction", "balancing dir", "direction card", "richtungs-karte", "dominante richtung"],
        "en": ("KPI: Dominant Balancing Dir. (mFRR)",
               "The mFRR class with the highest mean probability over the horizon: **{dominant_dir}** "
               "(`P(Up) {mean_p_up:.1%}` · `P(Down) {mean_p_down:.1%}` · `P(None) {p_none:.1%}`). Up = deficit, Down = surplus, None = balanced."),
        "de": ("KPI: Dominant Balancing Dir. (mFRR)",
               "Die mFRR-Klasse mit der höchsten mittleren Wahrscheinlichkeit im Horizont: **{dominant_dir}** "
               "(`P(Up) {mean_p_up:.1%}` · `P(Down) {mean_p_down:.1%}` · `P(None) {p_none:.1%}`). Up = Defizit, Down = Überschuss, None = ausgeglichen."),
    },
    {
        "id": "kpi_band",
        "keywords": ["uncertainty band card", "band card", "80% uncertainty", "80 % uncertainty", "unsicherheitsband-karte", "bandbreite"],
        "en": ("KPI: 80% Uncertainty Band (P10–P90)",
               "Average distance between P90 and P10: **{mean_band:.1f} €/MWh**. A wide band = high uncertainty (volatile wind, tight grid); "
               "the band is conformally calibrated and covers exactly **80.0 %** of realized prices in validation."),
        "de": ("KPI: 80% Uncertainty Band (P10–P90)",
               "Durchschnittlicher Abstand zwischen P90 und P10: **{mean_band:.1f} €/MWh**. Breites Band = hohe Unsicherheit (volatiler Wind, knappes Netz); "
               "das Band ist konform kalibriert und deckt in der Validierung exakt **80,0 %** der echten Preise ab."),
    },
    # ======================= 4. ALERT BANNERS =======================
    {
        "id": "alerts_overview",
        "keywords": ["alert banner", "alerts", "alert", "warnbanner", "warnung", "warnungen", "banner", "situational alert", "hinweisbox"],
        "exclude": ["filter", "origins", "quick"],
        "en": ("The 4 Situational Alert Banners",
               "Shown below the KPI cards, prioritised top-down:\n"
               "1. 🔴 **High-Price Spike Alert (>Spot+50€)** – any step with `P(high spike) ≥ 30 %` (now max `{max_p_hi:.1%}`).\n"
               "2. 🟣 **Low / Negative Price Alert (<Spot−50€)** – any step with `P(low spike) ≥ 30 %` (now max `{max_p_lo:.1%}`).\n"
               "3. 🔄 **Forecast Revision Alert (vs. t−1h)** – P50 moved ≥ 15 €/MWh vs. last hour's forecast (now max `Δ {max_abs_rev:.1f} €`).\n"
               "4. ✅ **Balanced Grid Outlook** – none of the above; standard dispatch.\n\n"
               "Each banner names the affected time window and a *suggestion to review* – never an automatic switching command."),
        "de": ("Die 4 Situations-Warnbanner",
               "Unter den KPI-Karten, priorisiert von oben nach unten:\n"
               "1. 🔴 **High-Price Spike Alert (>Spot+50€)** – ein Schritt mit `P(High Spike) ≥ 30 %` (aktuell max `{max_p_hi:.1%}`).\n"
               "2. 🟣 **Low / Negative Price Alert (<Spot−50€)** – ein Schritt mit `P(Low Spike) ≥ 30 %` (aktuell max `{max_p_lo:.1%}`).\n"
               "3. 🔄 **Forecast Revision Alert (vs. t−1h)** – P50 hat sich um ≥ 15 €/MWh gegenüber der Vorstunden-Prognose verändert (aktuell max `Δ {max_abs_rev:.1f} €`).\n"
               "4. ✅ **Balanced Grid Outlook** – nichts davon trifft zu; Standard-Fahrplan.\n\n"
               "Jedes Banner nennt das betroffene Zeitfenster und eine *Prüf-Empfehlung* – nie einen automatischen Schaltbefehl."),
    },
    {
        "id": "alert_high",
        "keywords": ["high-price spike alert", "high price alert", "red alert", "roter alarm", "hochpreis-warnung", "hochpreis-alarm"],
        "en": ("🔴 High-Price Spike Alert",
               "Triggered when at least one of the 40 steps has `P(price > Spot + 50 €) ≥ 30 %`. Current max: **{max_p_hi:.1%}** at `{hi_time}`. "
               "Suggested review: flexible generation, electric boiler load and open intraday short positions before the window."),
        "de": ("🔴 High-Price Spike Alert",
               "Wird ausgelöst, wenn mindestens einer der 40 Schritte `P(Preis > Spot + 50 €) ≥ 30 %` hat. Aktuelles Maximum: **{max_p_hi:.1%}** um `{hi_time}`. "
               "Prüf-Empfehlung: flexible Erzeugung, Elektrokessel-Last und offene Intraday-Short-Positionen vor dem Fenster."),
    },
    {
        "id": "alert_low",
        "keywords": ["low / negative price alert", "negative price alert", "low price alert", "negativpreis", "tiefpreis-warnung", "negative preise"],
        "en": ("🟣 Low / Negative Price Alert",
               "Triggered when any step has `P(price < Spot − 50 €) ≥ 30 %`, typically with strong wind and low demand. Current max: **{max_p_lo:.1%}** at `{lo_time}`. "
               "Suggested review: power-to-heat (electric boilers, heat pumps) and mFRR down-regulation bids."),
        "de": ("🟣 Low / Negative Price Alert",
               "Wird ausgelöst, wenn ein Schritt `P(Preis < Spot − 50 €) ≥ 30 %` hat, typisch bei viel Wind und geringer Last. Aktuelles Maximum: **{max_p_lo:.1%}** um `{lo_time}`. "
               "Prüf-Empfehlung: Power-to-Heat (Elektrokessel, Wärmepumpen) und mFRR-Down-Gebote."),
    },
    {
        "id": "alert_revision",
        "keywords": ["revision alert", "forecast revision alert", "revisions-warnung", "revisionswarnung"],
        "en": ("🔄 Forecast Revision Alert (vs. t−1h)",
               "Compares this origin's P50 with the forecast issued one hour earlier for the same delivery quarter-hours. "
               "If any step moved by **≥ 15 €/MWh**, the banner appears. Now: max `Δ {max_abs_rev:.1f} €/MWh`, mean `{mean_rev:+.1f} €/MWh`."),
        "de": ("🔄 Forecast Revision Alert (vs. t−1h)",
               "Vergleicht den P50 dieses Origins mit der eine Stunde früher erstellten Prognose für dieselben Lieferviertelstunden. "
               "Hat sich ein Schritt um **≥ 15 €/MWh** verändert, erscheint das Banner. Aktuell: max `Δ {max_abs_rev:.1f} €/MWh`, Mittel `{mean_rev:+.1f} €/MWh`."),
    },
    {
        "id": "alert_normal",
        "keywords": ["balanced grid outlook", "balanced grid", "green banner", "grünes banner", "ausgeglichenes netz", "normalbetrieb"],
        "en": ("✅ Balanced Grid Outlook",
               "Shown when no spike probability reaches 30 % and no revision ≥ 15 €/MWh occurred. Dominant tendency: **{dominant_dir}** – keep the standard schedule."),
        "de": ("✅ Balanced Grid Outlook",
               "Erscheint, wenn keine Spike-Wahrscheinlichkeit 30 % erreicht und keine Revision ≥ 15 €/MWh auftrat. Dominante Tendenz: **{dominant_dir}** – Standard-Fahrplan beibehalten."),
    },
    # ======================= 5. TAB 1 =======================
    {
        "id": "fan_chart",
        "keywords": ["fan chart", "fan-chart", "main chart", "hauptgraph", "hauptdiagramm", "probabilistic 10h", "now line", "now-linie", "fächer"],
        "en": ("📈 Probabilistic 10h Fan Chart (Tab 1)",
               "* **Left of `NOW`:** last 6 hours of realized history.\n"
               "* **Right of `NOW`:** 40 quarter-hour steps (+10 h) with the **P50 median** (turquoise), the shaded **P10–P90 80 % band**, "
               "the **Day-Ahead Spot Price** (dashed orange) and the **t−1h forecast** (thin line) to see revisions.\n"
               "* Current origin `{origin}`: P50 peaks at **{peak_p50:.1f} €/MWh** at `{peak_time}`."),
        "de": ("📈 Probabilistic 10h Fan Chart (Tab 1)",
               "* **Links von `NOW`:** die letzten 6 Stunden realisierte Historie.\n"
               "* **Rechts von `NOW`:** 40 Viertelstunden-Schritte (+10 h) mit **P50-Median** (türkis), schattiertem **P10–P90-Band (80 %)**, "
               "**Day-Ahead-Spotpreis** (orange gestrichelt) und der **t−1h-Prognose** (dünne Linie) zur Erkennung von Revisionen.\n"
               "* Aktueller Origin `{origin}`: P50-Spitze **{peak_p50:.1f} €/MWh** um `{peak_time}`."),
    },
    {
        "id": "xai_cards",
        "keywords": ["xai summary", "summary card", "why p50", "why uncertainty", "why is the band", "erklärkarte", "xai-karte", "warum p50", "warum ist das band"],
        "en": ("2 XAI Summary Cards (Tab 1)",
               "Below the fan chart, for the peak-risk step:\n"
               "* **Why P50?** – `P50 = Spot + AI spread`, with the top SHAP drivers pushing the price up or down.\n"
               "* **Why this uncertainty width?** – splits the P10–P90 width into 4 €/MWh sources (horizon distance, wind volatility, grid tightness, base model uncertainty)."),
        "de": ("2 XAI-Erklärkarten (Tab 1)",
               "Unter dem Fan-Chart, für den Schritt mit dem höchsten Risiko:\n"
               "* **Warum P50?** – `P50 = Spot + KI-Spread`, mit den wichtigsten SHAP-Treibern, die den Preis nach oben oder unten drücken.\n"
               "* **Warum diese Unsicherheitsbreite?** – zerlegt die P10–P90-Breite in 4 Quellen in €/MWh (Horizontabstand, Windvolatilität, Netzknappheit, Basis-Modellunsicherheit)."),
    },
    {
        "id": "spike_chart",
        "keywords": ["spike probability chart", "spike chart", "spike-diagramm", "spike diagramm", "spike-wahrscheinlichkeits", "spike probability graph", "balkendiagramm"],
        "en": ("Price Spike Probability Chart (Tab 1)",
               "Bars per quarter-hour: **berry red** = `P(price > Spot + 50 €)`, **plum** = `P(price < Spot − 50 €)`. "
               "The dashed horizontal line marks the **30 % alert threshold**. Now: high max `{max_p_hi:.1%}` at `{hi_time}`, low max `{max_p_lo:.1%}` at `{lo_time}`."),
        "de": ("Price Spike Probability Chart (Tab 1)",
               "Balken pro Viertelstunde: **Beeren-Rot** = `P(Preis > Spot + 50 €)`, **Pflaume** = `P(Preis < Spot − 50 €)`. "
               "Die gestrichelte Linie markiert die **30 %-Warnschwelle**. Aktuell: High max `{max_p_hi:.1%}` um `{hi_time}`, Low max `{max_p_lo:.1%}` um `{lo_time}`."),
    },
    {
        "id": "mfrr_chart",
        "keywords": ["mfrr chart", "direction chart", "balancing direction chart", "stacked area", "regelrichtungs-diagramm", "flächendiagramm", "mfrr-diagramm", "mfrr graph"],
        "en": ("mFRR Balancing Direction Chart (Tab 1)",
               "100 % stacked area per step: **orange = Up** (deficit), **turquoise = Down** (surplus), **grey = None**. "
               "The three probabilities always sum to 1. Horizon means: Up `{mean_p_up:.1%}`, Down `{mean_p_down:.1%}`, None `{p_none:.1%}`."),
        "de": ("mFRR Balancing Direction Chart (Tab 1)",
               "100 %-gestapelte Fläche pro Schritt: **Orange = Up** (Defizit), **Türkis = Down** (Überschuss), **Grau = None**. "
               "Die drei Wahrscheinlichkeiten ergeben immer 1. Horizont-Mittel: Up `{mean_p_up:.1%}`, Down `{mean_p_down:.1%}`, None `{p_none:.1%}`."),
    },
    # ======================= 6. TAB 2 =======================
    {
        "id": "step_selector",
        "keywords": ["step selector", "select a step", "select step", "schritt-auswahl", "schrittauswahl", "slider", "schieberegler"],
        "en": ("Step Selector (Tab 2)",
               "Choose any of the **40 quarter-hour steps** (Step 1 = first 15 min after `t`, Step 40 = +10 h). "
               "The Price Waterfall and Uncertainty Breakdown below then explain exactly that delivery quarter-hour."),
        "de": ("Step Selector (Tab 2)",
               "Wähle einen der **40 Viertelstunden-Schritte** (Step 1 = erste 15 Min nach `t`, Step 40 = +10 h). "
               "Der Preis-Wasserfall und die Unsicherheits-Zerlegung darunter erklären dann genau diese Lieferviertelstunde."),
    },
    {
        "id": "waterfall",
        "keywords": ["waterfall", "wasserfall", "how p50 is built", "wie p50 entsteht"],
        "en": ("Price Waterfall Chart – How P50 Is Built From Spot (Tab 2)",
               "Starts at the **Day-Ahead Spot Price**, then adds one bar per SHAP driver group (green = pushes price up, red = pushes down) "
               "and ends at the **P50 forecast**. Formula: `P50 = Spot + Σ SHAP contributions`. Current horizon mean: Spot `{spot_mean:.2f} €` → P50 `{p50_mean:.2f} €`."),
        "de": ("Preis-Wasserfall – Wie P50 aus dem Spotpreis entsteht (Tab 2)",
               "Startet beim **Day-Ahead-Spotpreis**, addiert dann je einen Balken pro SHAP-Treibergruppe (grün = treibt Preis hoch, rot = senkt ihn) "
               "und endet beim **P50-Prognosewert**. Formel: `P50 = Spot + Σ SHAP-Beiträge`. Aktuelles Horizont-Mittel: Spot `{spot_mean:.2f} €` → P50 `{p50_mean:.2f} €`."),
    },
    {
        "id": "uncertainty_breakdown",
        "keywords": ["uncertainty breakdown", "breakdown chart", "unsicherheits-zerlegung", "unsicherheitszerlegung", "sources of uncertainty", "unsicherheitsquellen"],
        "en": ("Uncertainty Breakdown Chart (Tab 2)",
               "Decomposes the selected step's band width `P90 − P10` into 4 €/MWh sources: **forecast horizon distance**, **wind volatility**, "
               "**grid tightness / residual load** and **base model uncertainty**. Horizon mean width now: `{mean_band:.1f} €/MWh`."),
        "de": ("Uncertainty Breakdown Chart (Tab 2)",
               "Zerlegt die Bandbreite `P90 − P10` des gewählten Schritts in 4 Quellen (€/MWh): **Prognose-Horizontabstand**, **Windvolatilität**, "
               "**Netzknappheit / Residuallast** und **Basis-Modellunsicherheit**. Aktuelle mittlere Breite: `{mean_band:.1f} €/MWh`."),
    },
    {
        "id": "grid_trajectory",
        "keywords": ["grid trajectory", "physical grid", "netz-verlauf", "netzverlauf", "physikalisch"],
        "en": ("Physical Grid Trajectory (Tab 2)",
               "Lines for **consumption forecast**, **wind forecast** and **residual load** (`load − wind − solar`) in MW over the 10 h. "
               "Now: load Ø `{cons_mean:,.0f} MW`, wind Ø `{wind_mean:,.0f} MW`, residual Ø `{res_mean:,.0f} MW`. Rising residual load → higher up-regulation and spike risk."),
        "de": ("Physical Grid Trajectory (Tab 2)",
               "Linien für **Verbrauchsprognose**, **Windprognose** und **Residuallast** (`Last − Wind − Solar`) in MW über 10 h. "
               "Aktuell: Last Ø `{cons_mean:,.0f} MW`, Wind Ø `{wind_mean:,.0f} MW`, Residuallast Ø `{res_mean:,.0f} MW`. Steigende Residuallast → höheres Up-Regulation- und Spike-Risiko."),
    },
    {
        "id": "top_shap",
        "keywords": ["top shap", "shap drivers", "shap-treiber", "overall drivers", "wichtigste treiber", "main drivers", "haupttreiber", "what is shap", "was ist shap"],
        "en": ("Overall Top SHAP Drivers (Tab 2)",
               "**SHAP** (SHapley Additive exPlanations, TreeSHAP for LightGBM) assigns each feature its exact €/MWh contribution to the forecast. "
               "The chart ranks the drivers by their share over all 40 steps. Top drivers now: **{top_drivers}**."),
        "de": ("Overall Top SHAP Drivers (Tab 2)",
               "**SHAP** (SHapley Additive exPlanations, TreeSHAP für LightGBM) weist jedem Merkmal seinen exakten Beitrag in €/MWh zur Prognose zu. "
               "Das Diagramm sortiert die Treiber nach ihrem Anteil über alle 40 Schritte. Aktuelle Top-Treiber: **{top_drivers}**."),
    },
    {
        "id": "advanced_xai",
        "keywords": ["advanced xai", "xai details", "erweiterte xai", "expander"],
        "en": ("🔬 Advanced XAI Details (Expander in Tab 2)",
               "Contains three deep-dive visuals: the **40-Step Temporal Horizon × Driver Heatmap**, the **Birnbaum Reliability & Criticality Importance** "
               "table/chart (`I_B`, `I_CR`, `I_FV`) and the **Global LightGBM Feature Importance** (information gain)."),
        "de": ("🔬 Advanced XAI Details (Expander in Tab 2)",
               "Enthält drei Detail-Visualisierungen: die **40-Step Temporal Horizon × Driver Heatmap**, die **Birnbaum Reliability & Criticality Importance** "
               "Tabelle/Grafik (`I_B`, `I_CR`, `I_FV`) und die **Global LightGBM Feature Importance** (Information Gain)."),
    },
    {
        "id": "heatmap",
        "keywords": ["heatmap", "heat map", "wärmekarte", "attention"],
        "en": ("40-Step Temporal Horizon × Driver Heatmap",
               "Rows = driver groups, columns = steps 1–40, colour = share of SHAP attribution. It shows **how driver importance shifts with the horizon**: "
               "near-term steps rely on recent imbalance/mFRR telemetry, distant steps on Spot, wind and load forecasts (cf. Huang et al., Heliyon 2024, Fig. 11)."),
        "de": ("40-Step Temporal Horizon × Driver Heatmap",
               "Zeilen = Treibergruppen, Spalten = Schritte 1–40, Farbe = Anteil der SHAP-Attribution. Sie zeigt, **wie sich die Treiber-Wichtigkeit mit dem Horizont verschiebt**: "
               "nahe Schritte stützen sich auf aktuelle Imbalance-/mFRR-Telemetrie, ferne Schritte auf Spot-, Wind- und Lastprognosen (vgl. Huang et al., Heliyon 2024, Fig. 11)."),
    },
    {
        "id": "birnbaum",
        "keywords": ["birnbaum", "i_b", "i_cr", "i_fv", "criticality", "kritikalität", "reliability importance", "zuverlässigkeit", "fussell"],
        "en": ("Birnbaum Reliability & Criticality Importance",
               "Reliability-engineering metrics applied to the forecast drivers:\n"
               "* **`I_B` (Birnbaum):** how sensitive the spike risk is to a driver changing.\n"
               "* **`I_CR` (Criticality):** `I_B` weighted by how likely the driver is to be in a critical state.\n"
               "* **`I_FV` (Fussell–Vesely):** share of total risk attributable to that driver.\n\n"
               "High values = drivers that operators should monitor most closely."),
        "de": ("Birnbaum Reliability & Criticality Importance",
               "Kennzahlen aus der Zuverlässigkeitstechnik, angewendet auf die Prognosetreiber:\n"
               "* **`I_B` (Birnbaum):** wie empfindlich das Spike-Risiko auf eine Änderung des Treibers reagiert.\n"
               "* **`I_CR` (Criticality):** `I_B` gewichtet mit der Wahrscheinlichkeit, dass der Treiber in einem kritischen Zustand ist.\n"
               "* **`I_FV` (Fussell–Vesely):** Anteil des Gesamtrisikos, der diesem Treiber zuzuordnen ist.\n\n"
               "Hohe Werte = Treiber, die der Schichtbetrieb besonders genau beobachten sollte."),
    },
    {
        "id": "global_importance",
        "keywords": ["global lightgbm", "feature importance", "global importance", "information gain", "globale wichtigkeit", "merkmalswichtigkeit"],
        "en": ("Global LightGBM Feature Importance",
               "Total **information gain** each feature contributed across all trees of the trained LightGBM models – a model-wide view, "
               "in contrast to the origin-specific SHAP values. Spot price, calendar features and wind ramps typically rank highest."),
        "de": ("Global LightGBM Feature Importance",
               "Gesamter **Information Gain**, den jedes Merkmal über alle Bäume der trainierten LightGBM-Modelle beigetragen hat – eine modellweite Sicht, "
               "im Gegensatz zu den origin-spezifischen SHAP-Werten. Spotpreis, Kalendermerkmale und Wind-Rampen liegen typischerweise vorne."),
    },
    # ======================= 7. TAB 3 =======================
    {
        "id": "benchmark_table",
        "keywords": ["benchmark table", "validation benchmark", "benchmark-tabelle", "vergleichstabelle", "official baseline", "offizielle baseline", "vs baseline", "vs. baseline", "beat the baseline"],
        "en": ("Validation Benchmark Table (Tab 3)",
               "| Metric | Baseline | Ours |\n| :--- | :---: | :---: |\n"
               "| P50 MAE | 66.20 | **43.19** (−35 %) |\n| Pinball P10/P90 | 24.18 | **12.43** (−49 %) |\n"
               "| P10–P90 coverage | – | **80.0 %** |\n| Spike Brier | – | **0.0928** |\n| Direction Brier | – | **0.4222** |\n\n"
               "Scored on the Validation Period (7 Sep – 4 Oct 2026) with the official `score.py`."),
        "de": ("Validation Benchmark Table (Tab 3)",
               "| Kennzahl | Baseline | Unser Modell |\n| :--- | :---: | :---: |\n"
               "| P50 MAE | 66,20 | **43,19** (−35 %) |\n| Pinball P10/P90 | 24,18 | **12,43** (−49 %) |\n"
               "| P10–P90-Abdeckung | – | **80,0 %** |\n| Spike Brier | – | **0,0928** |\n| Richtungs-Brier | – | **0,4222** |\n\n"
               "Bewertet auf der Validation Period (7. Sep – 4. Okt 2026) mit dem offiziellen `score.py`."),
    },
    {
        "id": "horizon_mae",
        "keywords": ["mae by horizon", "horizon step chart", "horizon mae", "mae per step", "mae pro schritt", "fehler pro schritt", "horizont-mae", "error by horizon"],
        "en": ("Validation MAE by Horizon Step (Tab 3)",
               "Plots the P50 error (MAE) for each of the 40 steps, ours vs. baseline. Error grows with distance from `t`, "
               "but our model stays below the baseline at every step – the biggest gain is in the first 1–3 hours thanks to fresh telemetry."),
        "de": ("Validation MAE by Horizon Step (Tab 3)",
               "Zeigt den P50-Fehler (MAE) für jeden der 40 Schritte, unser Modell vs. Baseline. Der Fehler wächst mit dem Abstand zu `t`, "
               "aber unser Modell liegt in jedem Schritt unter der Baseline – der größte Vorsprung in den ersten 1–3 Stunden dank frischer Telemetrie."),
    },
    {
        "id": "ablation",
        "keywords": ["ablation", "ablationsstudie", "architecture stages", "progressive"],
        "en": ("Progressive Ablation Study (Heliyon Table 3)",
               "Adds the architecture components one by one (baseline → LightGBM with spot/calendar → + Fingrid wind/load features → + real-time telemetry → "
               "+ conformal calibration) and reports MAE/Pinball after each stage, proving **every component contributes** to the final 43.19 €/MWh MAE."),
        "de": ("Progressive Ablation Study (Heliyon Table 3)",
               "Fügt die Architektur-Bausteine schrittweise hinzu (Baseline → LightGBM mit Spot/Kalender → + Fingrid Wind/Last → + Echtzeit-Telemetrie → "
               "+ konforme Kalibrierung) und zeigt MAE/Pinball nach jeder Stufe – Beleg, dass **jeder Baustein** zum finalen MAE von 43,19 €/MWh beiträgt."),
    },
    {
        "id": "significance",
        "keywords": ["wilcoxon", "t-test", "t test", "significance", "signifikanz", "signifikant", "statistical", "statistisch", "p-value", "p-wert"],
        "en": ("Statistical Significance: Paired t-Test & Wilcoxon (Heliyon Table 5)",
               "Compares the per-row absolute errors of our model and the baseline on the same validation rows. Both the **paired Student's t-test** "
               "and the non-parametric **Wilcoxon signed-rank test** give `p < 0.0001` – the improvement is statistically significant, not luck."),
        "de": ("Statistische Signifikanz: gepaarter t-Test & Wilcoxon (Heliyon Table 5)",
               "Vergleicht die absoluten Fehler unseres Modells und der Baseline zeilenweise auf denselben Validierungsdaten. Sowohl der **gepaarte t-Test** "
               "als auch der nichtparametrische **Wilcoxon-Vorzeichen-Rang-Test** ergeben `p < 0,0001` – die Verbesserung ist statistisch signifikant, kein Zufall."),
    },
    {
        "id": "calibration_scatter",
        "keywords": ["calibration scatter", "scatter", "45°", "45 degree", "45-grad", "kalibrierung", "calibration chart", "streudiagramm", "predicted vs"],
        "en": ("45° Calibration Scatter (Heliyon Fig. 10)",
               "Each dot = one validation quarter-hour: x = predicted P50, y = realized price. Dots on the **45° diagonal** are perfect forecasts. "
               "The chart also verifies that **80.0 %** of realized prices fall inside the P10–P90 band."),
        "de": ("45°-Kalibrierungs-Scatterplot (Heliyon Fig. 10)",
               "Jeder Punkt = eine Validierungs-Viertelstunde: x = prognostizierter P50, y = echter Preis. Punkte auf der **45°-Diagonale** sind perfekte Prognosen. "
               "Die Grafik bestätigt außerdem, dass **80,0 %** der echten Preise im P10–P90-Band liegen."),
    },
    # ======================= 8. TAB 4 =======================
    {
        "id": "data_table",
        "keywords": ["data table", "40-step table", "40 step table", "datentabelle", "tabelle", "rohdaten", "raw data"],
        "en": ("40-Step Data Table (Tab 4)",
               "Shows all 40 rows for origin `{origin}` with the exact submission columns: `delivery_start_utc`, `step`, `p10`, `p50`, `p90`, "
               "`p_up`, `p_down`, `p_none`, `p_spike_high`, `p_spike_low`, plus Spot price and `top_drivers`."),
        "de": ("40-Step Data Table (Tab 4)",
               "Zeigt alle 40 Zeilen für Origin `{origin}` mit den exakten Submission-Spalten: `delivery_start_utc`, `step`, `p10`, `p50`, `p90`, "
               "`p_up`, `p_down`, `p_none`, `p_spike_high`, `p_spike_low` sowie Spotpreis und `top_drivers`."),
    },
    {
        "id": "json_api",
        "keywords": ["json", "api payload", "download", "herunterladen", "rest api", "export"],
        "en": ("⬇️ Download Origin Forecast as JSON API Payload (Tab 4)",
               "Downloads `forecast_<origin>.json` containing the origin, data cutoff `{cutoff}` and all 40 steps – ready for SCADA, trading or "
               "BI integration. *Preview JSON API Response in Browser* shows the same payload inline."),
        "de": ("⬇️ Download Origin Forecast as JSON API Payload (Tab 4)",
               "Lädt `forecast_<origin>.json` mit Origin, Daten-Cutoff `{cutoff}` und allen 40 Schritten herunter – bereit zur Anbindung an SCADA, Handel "
               "oder BI. *Preview JSON API Response in Browser* zeigt denselben Payload direkt an."),
    },
    # ======================= 9. AI ASSISTANT CONTROLS =======================
    {
        "id": "gear",
        "keywords": ["gear", "zahnrad", "⚙", "settings", "einstellungen", "api key", "api-key", "api schlüssel", "clear chat", "chat löschen", "which model", "welches modell", "ai model"],
        "en": ("⚙️ AI Assistant Settings (Gear Button)",
               "The gear in the top-right of the AI box opens:\n"
               "* **AI model selector** – `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash` (GPT-4o / Llama 3.3 also supported via key).\n"
               "* **API key field** (masked, not stored by the browser password manager).\n"
               "* **🗑️ Clear Chat History**.\n\n"
               "Without a key the built-in, bilingual knowledge engine answers all website questions."),
        "de": ("⚙️ KI-Einstellungen (Zahnrad-Button)",
               "Das Zahnrad oben rechts in der KI-Box öffnet:\n"
               "* **KI-Modell-Auswahl** – `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash` (GPT-4o / Llama 3.3 ebenfalls per Key möglich).\n"
               "* **API-Key-Feld** (maskiert, wird nicht vom Browser-Passwortmanager gespeichert).\n"
               "* **🗑️ Clear Chat History**.\n\n"
               "Ohne Key beantwortet die eingebaute, zweisprachige Wissens-Engine alle Fragen zur Website."),
    },
    {
        "id": "quick_questions",
        "keywords": ["quick question", "schnellfrage", "vorgefertigte frage", "predefined question", "preset question"],
        "en": ("✨ Quick Questions",
               "Four one-click prompts above the input: **Drivers at Peak**, **Spike Risk Summary**, **Change vs. t−1h** and **Review Actions**. "
               "Clicking one sends it to the chat and auto-scrolls to the answer."),
        "de": ("✨ Quick Questions",
               "Vier Ein-Klick-Fragen über dem Eingabefeld: **Drivers at Peak**, **Spike Risk Summary**, **Change vs. t−1h** und **Review Actions**. "
               "Ein Klick schickt die Frage in den Chat und scrollt automatisch zur Antwort."),
    },
    {
        "id": "reasoning",
        "keywords": ["reasoning", "chain-of-thought", "chain of thought", "gedankengang", "denkprozess", "begründung"],
        "en": ("🧠 Show Model Reasoning (Chain-of-Thought)",
               "When enabled, every answer gets a collapsible **reasoning trace** with 4 steps: (1) question & language detection, "
               "(2) data context used (origin, period, cutoff – Rule #1 checked), (3) key evidence (SHAP drivers, spike/direction probabilities), (4) conclusion."),
        "de": ("🧠 Show Model Reasoning (Chain-of-Thought)",
               "Ist der Schalter aktiv, erhält jede Antwort einen aufklappbaren **Gedankengang** in 4 Schritten: (1) Frage & Spracherkennung, "
               "(2) genutzter Datenkontext (Origin, Zeitraum, Cutoff – Regel #1 geprüft), (3) zentrale Belege (SHAP-Treiber, Spike-/Richtungswahrscheinlichkeiten), (4) Schlussfolgerung."),
    },
    {
        "id": "imbalance_price",
        "keywords": ["imbalance price", "ausgleichspreis", "ausgleichsenergie", "balancing price", "what are we forecasting", "was wird prognostiziert"],
        "en": ("What Is the Imbalance Price?",
               "Fingrid's **15-minute settlement price** for deviations between a balance party's schedule and actual metered energy in Finland. "
               "It is set by the activated mFRR balancing energy, so it can be far above Spot (deficit) or far below / negative (surplus). "
               "We forecast it 10 h ahead: P50 now averages **{p50_mean:.2f} €/MWh** vs. Spot `{spot_mean:.2f} €/MWh`."),
        "de": ("Was ist der Ausgleichspreis (Imbalance Price)?",
               "Fingrids **15-Minuten-Abrechnungspreis** für Abweichungen zwischen Fahrplan und tatsächlich gemessener Energie eines Bilanzkreises in Finnland. "
               "Er ergibt sich aus der aktivierten mFRR-Regelenergie und kann daher weit über dem Spot (Defizit) oder weit darunter / negativ (Überschuss) liegen. "
               "Wir prognostizieren ihn 10 h voraus: P50 aktuell Ø **{p50_mean:.2f} €/MWh** vs. Spot `{spot_mean:.2f} €/MWh`."),
    },
]


class _SafeDict(dict):
    def __missing__(self, key):  # leave unknown placeholders visible instead of crashing
        return "{" + key + "}"


def match_kb_entry(question: str) -> dict | None:
    """Return the best-matching KB entry (longest total keyword overlap) or None."""
    q = question.lower()
    best, best_score = None, 0
    for entry in KB_ENTRIES:
        if any(x in q for x in entry.get("exclude", ())):
            continue
        score = sum(len(k) for k in entry["keywords"] if k in q)
        if score > best_score:
            best, best_score = entry, score
    return best


def render_kb_entry(entry: dict, is_de: bool, ctx: dict) -> str:
    """Render a KB entry as markdown in the requested language."""
    title, body = entry["de" if is_de else "en"]
    try:
        body_txt = body.format_map(_SafeDict(ctx))
    except (ValueError, KeyError, IndexError):
        body_txt = body
    return f"**{title}**\n\n{body_txt}"


def kb_digest_for_llm() -> str:
    """Compact English digest of every UI element, injected into live LLM prompts."""
    lines = []
    for e in KB_ENTRIES:
        title, body = e["en"]
        clean = re.sub(r"\{[a-z_]+(?::[^}]*)?\}", "<live>", body).replace("\n", " ")
        lines.append(f"- {title}: {clean}")
    return "\n".join(lines)
