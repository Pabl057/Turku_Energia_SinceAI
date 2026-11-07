# ⚡ Turku Energia Imbalance Price Forecasting – Der komplette Leitfaden & Architektur-Guide

Dieser Guide fasst das gesamte **Turku Energia Imbalance Price Challenge**-Projekt (Since AI Hackathon 2026) zusammen – inklusive **visueller Zeitstrahl-Logik**, **System-Architektur**, **Zusammenhang zwischen Day-Ahead Spotpreis, aFRR/mFRR und unserer Prognose** sowie einer **Schritt-für-Schritt-Anleitung** für die Umsetzung im Code.

---

## 1. Das Wichtigste zuerst: Wie hängen Day-Ahead-Prognose, Spotpreis, aFRR/mFRR und unsere Prediction zusammen?

Hier liegt oft das größte Missverständnis. Deshalb klären wir zuerst die **zeitliche Kausalität**:

> [!IMPORTANT]
> **Wir müssen NICHT den Spotpreis vorhersagen!**
> Den **Day-Ahead Spotpreis** bekommen wir von der Strombörse Nord Pool **bereits am Vortag um 14:00 Uhr fertig geliefert** (`data/spot_nordpool.csv`).
> Was wir vorhersagen müssen, ist der **Ausgleichsenergiepreis (`Imbalance Price`)** – also wie stark der echte Netzpreis in den nächsten 10 Stunden vom bereits bekannten Spotpreis nach oben oder unten abweichen wird!

### 🕒 Der exakte 24-Stunden-Zeitstrahl (vom Vortag bis zu unserer Prediction)

#### A. Chronologischer Zeitgraph (Alle Uhrzeiten & Lags im Detail)

Beispiel: Wir stehen **heute um 15:00 Uhr (`t`)** und sagen die Preise bis **01:00 Uhr nachts (`t + 10h`)** vorher:

```mermaid
flowchart TD
    subgraph T_YESTERDAY["VORTAG (D-1): Fahrplan & Börsenpreis werden fixiert"]
        T1["13:00 Uhr (D-1): Fingrid veröffentlicht Wind- (246), Solar- (247) & Verbrauchsprognose (165)"]
        T2["14:00 Uhr (D-1): Nord Pool fixiert Day-Ahead Spotpreis + ENTSO-E Kapazitäten (11.1) für den ganzen Folgetag"]
        T1 --> T2
    end

    subgraph T_HISTORY["HEUTE VOR 15:00 UHR: Was im echten Stromnetz passiert ist (Lag-Fenster)"]
        T3["Bis 13:00 Uhr (t - 120 Min): ENTSO-E Nachbarländer-Imbalance & Last (Schweden SE1/SE3, Estland) sichtbar"]
        T4["Bis 14:00 Uhr (t - 60 Min): Echter Ausgleichspreis (319) + echte aFRR (347/348) & mFRR (244/106/369) sichtbar"]
        T5["Bis 14:30 Uhr (t - 30 Min): Echter Verbrauch (124), Erzeugung (74) & Grenzflüsse (61/57/60/55) sichtbar"]
        T6["14:30 bis 15:00 Uhr: Blindflug-Zone (Messwerte noch nicht veröffentlicht -> Anti-Leakage!)"]
        T3 --> T4 --> T5 --> T6
    end

    subgraph T_ORIGIN["HEUTE 15:00 UHR (Zeitpunkt t = Forecast Origin)"]
        T7["LightGBM-Inferenz startet um Punkt 15:00 Uhr"]
    end

    subgraph T_HORIZON["ZUKUNFT: 15:00 UHR BIS 01:00 UHR NACHTS (10 Stunden = 40 Viertelstunden-Schritte)"]
        S1["Step 1 (15:00 - 15:15 Uhr): Kurzfrist-Horizont -> stark geprägt vom aktuellen aFRR/mFRR-Stress vor 15:00 Uhr"]
        S4["Step 4 (15:45 - 16:00 Uhr): Ende Stunde 1 -> Übergang von Echtzeit-Trägheit zu Wetter-/Lastprognose"]
        S16["Step 16 (18:45 - 19:00 Uhr): Abendspitze (Stunde 4) -> Spotpreis + Residuallast-Rampe treiben Spike-Risiko"]
        S40["Step 40 (00:45 - 01:00 Uhr): Ende Stunde 10 -> orientiert sich primär am bekannten Spotpreis + Windprognose"]
        S1 --> S4 --> S16 --> S40
    end

    T2 -->|"Bekannter Zukunfts-Anker für Steps 1..40"| T7
    T4 & T5 -->|"Aktueller Netzstress-Indikator"| T7
    T7 --> S1
```

#### B. Preis-Zeitgraph: Spotpreis vs. Echter Ausgleichspreis (durch aFRR/mFRR) vs. Unsere Prognose

Dieser Graph zeigt anschaulich am Beispiel eines Nachmittags (mit Windflaute ab 14:00 Uhr), wie **aFRR & mFRR** den Preis vor 15:00 Uhr nach oben treiben und wie unser Modell daraus den Verlauf für **15:00 bis 21:00 Uhr** vorhersagt:

```mermaid
xychart-beta
    title "Zeitverlauf (EUR/MWh): Bekannter Spotpreis vs. Ausgleichspreis / P50-Prognose (ab 15:00 Uhr)"
    x-axis ["12:00", "13:00", "14:00 (Lag-Ende)", "15:00 (Origin t)", "16:00", "17:00", "18:00 (Abendspitze)", "19:00", "20:00", "21:00"]
    y-axis "Preis in EUR/MWh" 0 --> 220
    line "Day-Ahead Spotpreis (seit gestern 14h bekannt)" [45, 48, 55, 65, 75, 95, 115, 90, 60, 45]
    line "Ausgleichspreis (bis 14h echt durch aFRR/mFRR, ab 15h unsere P50-Prognose)" [46, 70, 125, 135, 130, 155, 185, 120, 70, 48]
```

* **12:00 Uhr:** Netz ist ausgeglichen (`mFRR = none`) $\rightarrow$ Ausgleichspreis ($46\text{ €}$) $\approx$ Spotpreis ($45\text{ €}$).
* **13:00 – 14:00 Uhr (vor $t$):** Weniger Wind als geplant! Fingrid aktiviert `aFRR Up` und `mFRR Up` $\rightarrow$ Echter Ausgleichspreis springt auf **$125\text{ €/MWh}$** (obwohl der Spotpreis nur bei $55\text{ €}$ lag!).
* **15:00 Uhr (unser Zeitpunkt $t$):** Unser Modell sieht den Sprung bis 14:00 Uhr **und** sieht, dass der bekannte Spotpreis bis 18:00 Uhr auf $115\text{ €}$ klettert. Es prognostiziert deshalb für 18:00 Uhr einen Ausgleichspreis von **$P_{50} = 185\text{ €/MWh}$** und schlägt Alarm (`p_spike_high > 0.60`, da $185\text{ €} > 115\text{ €} + 50\text{ €}$).
* **Ab 20:00 Uhr:** Wind nimmt laut Prognose wieder zu, Last sinkt $\rightarrow$ unsere $P_{50}$-Prognose nähert sich wieder dem bekannten Spotpreis ($48\text{ €} \approx 45\text{ €}$) an.

```mermaid
sequenceDiagram
    autonumber
    participant FG as "Fingrid (Vortag 13h)"
    participant NP as "Nord Pool (Vortag 14h)"
    participant RT as "Stromnetz (Heute vor t)"
    participant ML as "LightGBM (Heute bei t)"
    participant OUT as "Prognose (t + 10h)"

    FG->>NP: Wind-, Solar- & Lastprognose
    NP->>ML: Bekannter Spotpreis (für Zielstunden)
    RT->>ML: Aktuelle aFRR / mFRR Werte (60 Min Lag)
    ML->>OUT: Berechnet P10, P50, P90, Richtung & Spikes
```

### 🔍 Was passiert in den einzelnen Schritten genau?

#### Schritt A: Vortag 13:00 Uhr & 14:00 Uhr – Der Plan wird gemacht (Zukunfts-Features!)
1. **Um 13:00 Uhr am Vortag (D-1)** veröffentlicht Fingrid die Day-Ahead-Prognosen für den gesamten nächsten Tag:
   * Wie viel Windstrom wird morgen je Viertelstunde erwartet? (`dataset 246`)
   * Wie viel Solarstrom wird erwartet? (`dataset 247`)
   * Wie hoch wird der Stromverbrauch sein? (`dataset 165`)
2. **Um 14:00 Uhr am Vortag (D-1)** bildet sich an der Börse Nord Pool daraus der **Spotpreis (`fi_spot_price`)** für jede einzelne Viertelstunde des nächsten Tages.
3. **Warum ist das für uns Gold wert?**
   Wenn unser Modell heute um z. B. **15:00 Uhr** (`forecast_origin_utc = t`) die nächsten **10 Stunden** (`15:00` bis `01:00` nachts, Schritte 1 bis 40) vorhersagen soll, **kennen wir den Spotpreis und die Wind-/Verbrauchsprognosen für diese zukünftigen 10 Stunden bereits!** Sie stehen für die Zukunft bereit, ohne gegen die `first_usable_origin_utc`-Regel zu verstoßen.

#### Schritt B: Warum weicht der Ausgleichspreis überhaupt vom Spotpreis ab?
Zwischen 14:00 Uhr gestern (als der Spotpreis festgelegt wurde) und heute passiert die echte Physik im Stromnetz:
* Es weht plötzlich **weniger Wind** als gestern um 13:00 Uhr gedacht, oder es ist **kälter** (Verbraucher heizen mehr), oder ein Kraftwerk fällt aus.
* Jetzt fehlt im finnischen Netz plötzlich Strom (Frequenz fällt unter 50 Hz).
* Fingrid muss sofort **Regelenergie (`aFRR` automatisch in Sekunden, `mFRR` manuell in Minuten)** hochfahren (**Up-Regulation**).
* Wer kurzfristig Kraftwerke hochfährt, verlangt hohe Preise (oft +100 bis +500 €/MWh über dem Spotpreis).
* Aus dem **volumengewichteten Durchschnitt** von `mFRR` und `aFRR` entsteht der **Ausgleichspreis (`Imbalance Price`)**.

#### Schritt C: Wie spielen `aFRR` und `mFRR` in unsere eigene Prediction hinein?
Da `aFRR` und `mFRR` erst in Echtzeit aktiviert werden, kennen wir zum Zeitpunkt $t$ natürlich noch **nicht** die zukünftigen `aFRR`- und `mFRR`-Volumina für $t+1$ bis $t+10$ Stunden (diese sind erst 60 Minuten *nach* der jeweiligen Viertelstunde bekannt!).

**Aber `aFRR` und `mFRR` helfen unserer Prediction auf zwei entscheidenden Wegen:**

1. **Als „Fieberthermometer“ für den aktuellen Netz-Zustand (Lag-Features aus den Stunden vor $t$):**
   * Zum Zeitpunkt $t$ kennen wir alle echten `aFRR`- und `mFRR`-Aktivierungen, Preise und Richtungen bis zu $t - 60\text{ Minuten}$ (und den echten Verbrauch/Grenzfluss bis $t - 30\text{ Minuten}$).
   * **Beispiel:** Wenn in den letzten 2 Stunden vor $t$ bereits durchgehend 400 MW `mFRR Up-Regulation` aktiviert werden mussten und die Differenz zwischen echtem Verbrauch (`124`) und geplanter Verbrauchsprognose (`165`) riesig ist, weiß das Modell: *„Das Netz hat gerade ein strukturelles Stromdefizit – der Wetterbericht lag heute falsch!“*
   * Da Wetterfronten und Kraftwerksausfälle mehrere Stunden anhalten (**Autokorrelation / Trägheit**), überträgt das Modell diesen aktuellen Stress aus `aFRR`/`mFRR` auf die kommenden Stunden (besonders stark für die Schritte 1 bis 16, also die ersten 4 Stunden).

2. **Als gelernte Muster für zukünftige Engpässe (für Schritte 1 bis 40):**
   * Im **Training** sieht LightGBM tausende historische Stunden und lernt:
     * *„Immer wenn für Stunde $t+k$ ein hoher Spotpreis + steiler Anstieg der Verbrauchsprognose (Abendspitze 17–19 Uhr) + wenig Wind prognostiziert ist UND in den Stunden vor $t$ schon `aFRR`/`mFRR` nach oben geregelt hat, schießt der Ausgleichspreis mit hoher Wahrscheinlichkeit um $>50\text{ €/MWh}$ über den Spotpreis (`p_spike_high`).“*

---

## 2. Übersicht aller Datenquellen & wann wir sie nutzen dürfen (`availability.csv`)

Diese Tabelle zeigt exakt, welche Daten im Ordner `data/` liegen und wann sie für eine Prognose zum Zeitpunkt $t$ (`forecast_origin_utc`) sichtbar sind:

| Datenquelle (`series_group`) | Fingrid / ENTSO-E ID | Wann verfügbar? (`first_usable_origin_utc`) | Rolle für unsere Prediction zum Zeitpunkt $t$ |
| :--- | :--- | :--- | :--- |
| **Kalender** (`calendar`) | Wochentag, Stunde, Feiertag | **Immer im Voraus bekannt** | **Zukunfts-Feature** für Zielstunde $t+\text{step}$ (Tageszeit- & Wochenend-Muster) |
| **Day-Ahead Prognosen** (`wind`, `solar`, `consumption_forecast_da`) | `246`, `247`, `165` | **Vortag 13:00 Uhr** (Finnische Zeit) | **Zukunfts-Feature** für Zielstunde $t+\text{step}$ (Geplante Einspeisung & Last) |
| **Day-Ahead Spotpreis** (`spot_price`) | `spot_nordpool.csv` | **Vortag 14:00 Uhr** (Finnische Zeit) | **Wichtigstes Zukunfts-Feature** (Ankerpreis für Zielstunde $t+\text{step}$) |
| **Übertragungskapazität** (`transfer_capacity`) | `11.1` (FI–EE) | **Vortag 14:00 Uhr** (Finnische Zeit) | **Zukunfts-Feature** (Wie viel Strom kann über Kabel importiert/exportiert werden?) |
| **Echter Verbrauch & Erzeugung** | `124`, `74` | **30 Min nach Viertelstunden-Ende** | **Gegenwarts-/Lag-Feature** (zeigt aktuelle Abweichung vom Day-Ahead-Plan!) |
| **Grenzüberschreitende Flüsse** (`cross_border_flow`) | `61`, `57`, `60`, `55` | **30 Min nach Viertelstunden-Ende** | **Gegenwarts-/Lag-Feature** (Aktueller Import/Export mit Schweden/Estland/Norwegen) |
| **Echter Ausgleichspreis** (`imbalance_price`) | `319` | **60 Min nach Viertelstunden-Ende** | **Zielvariable ($y$)** + **wichtigstes Lag-Feature** (letzter bekannter Preis bei $t$) |
| **mFRR Preis, Volumen & Richtung** | `244`, `106`, `375`, `376`, `369` | **60 Min nach Viertelstunden-Ende** | **Zielvariable (`direction`)** + **Lag-Feature** (aktueller Regelenergie-Stress) |
| **aFRR Preis & Volumen** | `347`, `348` | **60 Min nach Viertelstunden-Ende** | **Lag-Feature** (automatische Kurzfrist-Regelung der letzten Stunden vor $t$) |
| **Nachbarländer Imbalance/Last** | `17.1.g`, `6.1.b` | **120 Min nach Perioden-Ende** | **Lag-Feature** (Netzstress in Schweden SE1/SE3 und Estland) |

---

## 3. Visuelle Gesamt-Architektur (End-to-End Pipeline)

```mermaid
flowchart TD
    A["1. Rohdaten (inputs.parquet + spot_nordpool.csv)"] --> B["2. Anti-Leakage Filter (first_usable_origin_utc <= t)"]
    B --> C1["Zukunfts-Features (Spotpreis + Wind/Last-Plan bei t+step)"]
    B --> C2["Gegenwarts-Features (aFRR/mFRR + letzter Preis vor t)"]
    C1 & C2 --> D["3. Feature-Matrix X (40 Schritte pro Origin)"]

    D --> M1["Preise: 3x Quantile LightGBM (P10, P50, P90)"]
    D --> M2["Richtung: 1x Multiclass LightGBM (p_up, p_down, p_none)"]
    D --> M3["Spikes: 2x Binary LightGBM (p_spike_high, p_spike_low)"]

    M1 & M2 & M3 --> E["4. Postprocessing & SHAP (Regel-Checks + top_drivers)"]
    E --> F["5. forecast_backtest.csv (validate_submission.py + score.py)"]
```

---

## 4. Unser smarter Trick für die Preisprognose: Den „Spread“ zum Spotpreis vorhersagen!

Anstatt das Modell blind den absoluten Ausgleichspreis schätzen zu lassen, hilft folgende Überlegung aus der Marktmechanik:

$$\text{Ausgleichspreis} = \text{Spotpreis} + \underbrace{(\text{Ausgleichspreis} - \text{Spotpreis})}_{\text{Imbalance Spread }\Delta P}$$

* Wenn im Netz **keine dominante Regelrichtung (`none`)** herrscht, ist $\Delta P = 0$, also $\text{Ausgleichspreis} = \text{Spotpreis}$.
* Wenn **Up-Regulation (`up`)** herrscht, ist $\Delta P \ge 0$ (Preis steigt über den Spotpreis).
* Wenn **Down-Regulation (`down`)** herrscht, ist $\Delta P \le 0$ (Preis fällt unter den Spotpreis).
* Auch die **Preisspitzen (`Spikes`)** sind in der Challenge exakt über diesen Spread definiert:
  * `spike_high = 1`, wenn $\Delta P > +50\text{ €/MWh}$
  * `spike_low = 1`, wenn $\Delta P < -50\text{ €/MWh}$

Daher ist der **Spotpreis der Ziel-Viertelstunde (`spot_at_delivery`)** (oder das direkte Trainieren auf den Spread $\Delta P$ und anschließendes Addieren des Spotpreises) der stärkste Hebel, um den **Baseline-MAE von 66,20 €/MWh** drastisch zu schlagen!

---

## 5. Schritt-für-Schritt-Kochbuch: Wie setzen wir das jetzt einfach um?

Unsere Projektstruktur ist bereits fertig vorbereitet:

```text
├── data/                      # Alle CSV- & Parquet-Daten (inkl. forecast_origins.csv, sample_submission.csv)
├── notebooks/
│   └── baseline.ipynb         # Offizielles Starter-Notebook (Referenz)
├── src/
│   ├── data/                  # Schritt 1: Daten laden & rows_available_at()
│   ├── features/              # Schritt 2: Feature-Matrix X bauen
│   ├── models/                # Schritt 3: Die 6 LightGBM-Modelle + SHAP
│   └── evaluation/            # Schritt 4: Postprocessing, validate_submission.py & score.py
├── apps/                      # Schritt 5: Level 2 Dashboard (Streamlit / UI)
├── score.py                   # Offizielles Scoring-Skript
└── validate_submission.py     # Offizielles Validierungs-Skript
```

### Schritt 1: Daten-Loader (`src/data/loader.py`)
Lädt `data/inputs.parquet` und `data/spot_nordpool.csv` und stellt `rows_available_at(data, origin_utc)` bereit, sodass niemals Zukunftsdaten durchsickern.

### Schritt 2: Feature-Builder (`src/features/build_features.py`)
Erzeugt für jeden Prognosezeitpunkt $t$ (`forecast_origin_utc`) und jeden der 40 Schritte (`step = 1..40`, also `delivery_start_utc = t + (step-1)*15min`) eine Zeile in der Feature-Matrix $X$:
1. **Horizont-Feature:** `step` (1 bis 40).
2. **Zukunfts-Features am Zeitpunkt `delivery_start_utc`** *(sofern `first_usable_origin_utc <= t`)*:
   * `spot_price_target`: Bekannter Nord Pool Spotpreis für die Ziel-Viertelstunde.
   * `wind_forecast_target`, `solar_forecast_target`, `consumption_forecast_target`: Fingrid Day-Ahead-Prognosen für die Ziel-Viertelstunde.
   * `residual_load_forecast`: `consumption_forecast_target - wind_forecast_target - solar_forecast_target` (wie viel Last muss durch konventionelle Kraftwerke/Importe gedeckt werden?).
   * `hour_utc`, `dayofweek`, `is_weekend`: Kalender-Features der Ziel-Viertelstunde.
3. **Lag-/Echtzeit-Features am Zeitpunkt `t`** *(die letzten bekannten Werte vor `t`)*:
   * `last_imbalance_price`: Letzter bekannter Ausgleichspreis (60 Min vor $t$) sowie Mittelwert/Std/Min/Max der letzten 4 und 12 Viertelstunden.
   * `last_imbalance_spread`: Letzter bekannter Ausgleichspreis minus damaliger Spotpreis.
   * `last_mfrr_up_vol`, `last_mfrr_down_vol`, `last_mfrr_direction`: Wie stark musste Fingrid zuletzt manuell regeln?
   * `last_afrr_up_vol`, `last_afrr_down_vol`: Wie stark musste Fingrid zuletzt automatisch regeln?
   * `consumption_forecast_error`: Letzter echter Verbrauch (`124`, 30 Min vor $t$) minus prognostizierter Verbrauch (`165`) – zeigt sofort, ob die Verbrauchsprognose heute danebenliegt!
   * `net_cross_border_flow`: Summe der aktuellen Grenzflüsse (`61`, `57`, `60`, `55`).

### Schritt 3: Training der 6 LightGBM-Modelle (`src/models/train.py`)
Wir trainieren die 6 Modelle mit `RANDOM_SEED = 42`:
* **Auf der Validierungsperiode testen:**
  * Training auf Daten vor dem **7. September 2026** (`2026-09-06T21:00:00Z`).
  * Vorhersage für die Validierungsperiode (**7. Sep – 4. Okt 2026**).
  * Bewertung mit `score.py` gegen `data/actuals_validation.csv` (Vergleich mit Baseline: `MAE 66.20`, `Pinball 24.18`).
* **Für die finale Abgabe (`forecast_backtest.csv`):**
  * Training auf allen Daten bis zum **4. Oktober 2026** (bzw. rollierendes Nachtrainieren).
  * Vorhersage für die **673 Test-Origins** (`5. Okt – 1. Nov 2026`, ergibt exakt **26.920 Zeilen**).

### Schritt 4: Postprocessing & Pflicht-Validierung
Vor dem Speichern von `forecast_backtest.csv` werden die Regeln erzwungen und geprüft:

```bash
# 1. Format-Validierung (muss VALIDATION PASSED ergeben!)
./venv/bin/python validate_submission.py --backtest forecast_backtest.csv --origins data/forecast_origins.csv

# 2. Lokales Scoring auf den Validierungsdaten
./venv/bin/python score.py --submission baseline_output/validation/forecast_backtest.csv --actuals data/actuals_validation.csv --spike-margin 50 --output-dir results/validation
```

---

## 6. Finale Ergebnisse & Status-Checkliste für das Team

- [x] **Python-Umgebung (`./venv`)** mit Python 3.12, `lightgbm`, `shap`, `pandas`, `numpy`, `scikit-learn`, `jupyter` installiert.
- [x] **Datenstruktur (`data/`)** sauber konsolidiert und in `.gitignore` vor versehentlichem Upload geschützt.
- [x] **Level 1 Pipeline (`src/`) komplett abgeschlossen & feingetunt:**
  1. `src/data/loader.py` & `src/features/build_features.py` (105 kausale Features ohne Data Leakage).
  2. `src/models/forecaster.py` (8-Modell LightGBM Suite inkl. Short-Horizon Specialists, Exponential Recency Weighting, Conformal Calibration, Empirical Bayes Brier Shrinkage & TreeSHAP `top_drivers`).
  3. Finale `forecast_backtest.csv` (26.920 Zeilen, `VALIDATION PASSED: 0 Errors, 0 Warnings`).
- [x] **Level 2 Situational Dashboard (`apps/dashboard.py`) fertiggestellt:**
  1. Interaktives Echtzeit-Dashboard (`./venv/bin/python apps/dashboard.py 8050` auf `http://localhost:8050`).
  2. Dual-Mode: **Validation Replay (mit echten Ist-Preisen & Live-Fehlermetriken)** + **Official Test Backtest (673 Origins)**.
  3. „Time Machine“-Slider, Quick-Jump zu Top-Spikes, Actionable Alerts für Trader & Shift Supervisors (`p_spike > 0.30`) und interaktive TreeSHAP `top_drivers`.

### 🏆 Offizielle Validierungsergebnisse (`score.py` auf 672 Origins / 26.880 Viertelstunden)

| Bewertungs-Kategorie (Gewichtung) | Offizielle Baseline | Reine Spotpreis-Benchmark | **Unser Fine-Tuned System** | Verbesserung ggü. Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **MAE auf `p50`** *(30% Gewicht)* | 66,20 €/MWh | 46,47 €/MWh | **43,65 €/MWh** *(Stunde 1: **40,83 €**)* | **-34,1 % Fehler** (-22,55 €/MWh) |
| **P10–P90 Coverage** *(Ziel: 80,0%)* | 68,29 % | — | **80,09 %** | **Punktlandung (+0,09 % am Ziel)** |
| **Mean Pinball Loss (`p10`/`p90`)** *(20% Gewicht)* | 24,18 | — | **13,03** (`p10`: 7,63 / `p90`: 18,43) | **-46,1 % Fehler** |
| **Spike Brier Score (`±50 €`)** *(20% Gewicht)* | 0,1132 | — | **0,0965** (ROC-AUC Low: `0.872`) | **-14,8 % besser** |
| **Direction 3-Class Brier** *(10% Gewicht)* | 0,4509 | — | **0,4134** *(Stunde 1: **0,3808**)* | **-8,3 % besser** (66,3 % Acc.) |

---

## 7. Ehrliche & Kritische Bestandsaufnahme: Was ist gut, was ist schlecht?

Um als Team genau zu wissen, wo wir im Wettbewerb stehen (und auf welche kritischen Fragen der Jury wir vorbereitet sein müssen), ist hier die ungeschönte Analyse unseres aktuellen Systems:

### ✅ Was wirklich stark ist (Unsere Wettbewerbsvorteile)

1. **Perfekte Unsicherheits-Kalibrierung (`80,09 %` Coverage & `-46,1 %` Pinball Loss – 20 % der Gesamtnote):**
   * Unkalibrierte GBDT-Quantilmodelle sind Out-of-Sample fast immer zu eng (`65–72 %` Coverage, Baseline lag bei `68,29 %`).
   * Durch unsere **Split Conformal Quantile Calibration** treffen wir das offizielle `80,0 %`-Ziel der Challenge auf **`0,09 %` genau (`80,09 %`)** und senken den Pinball Loss von `24,18` auf **`13,03`**. In dieser Bewertungskategorie holen wir nahezu das theoretische Maximum.
2. **Starke Kurzfrist-Prognose in Stunde 1–2 (`MAE 40,83 €/MWh`, `Direction Brier 0,3808`):**
   * Während der reine Nord Pool Spotpreis einen MAE von `46,47 €/MWh` hat, schlägt unser Modell den Spotpreis in der ersten Stunde um **`5,64 €/MWh` (-12,1 % unter Spot)** und über den gesamten 10h-Horizont um **`2,82 €/MWh` (-6,1 % unter Spot)**.
   * Möglich wird das durch die Kombination aus **30-Minuten-Verbrauchsprognosefehler (`Fingrid 124 vs. 165`)**, **60-Minuten aFRR/mFRR-Echtzeit-Lags** und unseren **Short-Horizon Specialist-Modellen** für die Schritte `1..8`.
3. **Exzellente Erkennung negativer Preisspitzen (`spike_low` ROC-AUC `0.872`):**
   * Überangebot-Situationen (Starkwind + niedrige Last $\rightarrow$ Preise stürzen um $>50\text{ €/MWh}$ unter Spot) erkennt unser Modell mit hoher Zuverlässigkeit (`Brier 0.0921`).
4. **100 % Regelkonformität + Komplettes End-to-End-Produkt (Level 1 + Level 2 + SHAP):**
   * **Kein Data Leakage:** Streng nach `first_usable_origin_utc <= t` gefiltert (`validate_submission.py` $\rightarrow$ `0 Errors, 0 Warnings`).
   * **Echte Erklärbarkeit:** Exakte `TreeSHAP`-Werte (`top_drivers`) in allen 26.920 Vorhersage-Zeilen.
   * **Einsatzbereites Level-2-Dashboard (`apps/dashboard.py`):** Mit Time-Machine, Spike-Quick-Jumps, Trader-Ampel-Alerts (`p_spike > 0.30`) und direktem Soll-/Ist-Vergleich auf der Validierungsperiode.

---

### ⚠️ Was noch schwach / kritisch ist (Ehrliche Schwachstellen & Grenzen)

1. **`P50`-Punktprognose ist bei extremen Preis-Spikes zu konservativ („Median-Trägheit“):**
   * **Das Problem:** Wenn der echte Ausgleichspreis bei einem Netz-Notfall auf `+500` oder `+700 €/MWh` explodiert (z. B. am `11.09.2026`), schlagen zwar unsere **`p_spike_high`-Wahrscheinlichkeit (`> 30–50 %`)** und das **`P90`-Band (`~250–350 €`)** deutlich an – aber unsere **`P50`-Kurve (Median)** geht oft nur um `+20` bis `+45 €/MWh` über den Spotpreis.
   * **Warum passiert das?** In **64 % aller Viertelstunden** herrscht keine dominante Regelrichtung (`direction = none`), sodass `Ausgleichspreis == Spotpreis` (`Spread = 0`) gilt. Ein `L1`/Median-Modell wird mathematisch hart bestraft, wenn es im `P50` mutig `+300 €` vorhersagt und der Spike eine Viertelstunde früher oder später eintritt.
2. **Nachlassende Prognosekraft ab Stunde 4 bis 10 (`Steps 13..40`):**
   * Ab Stunde 4 (`MAE ~44,20 €/MWh`) sind die aktuellen `aFRR`/`mFRR`-Störungen weitgehend abgeklungen. Da im offiziellen Datensatz **keine Intraday-Wetter-Updates** enthalten sind (nur die Fingrid Day-Ahead-Prognose vom Vortag 13:00 Uhr), kann das Modell plötzliche Wetterumschwünge in Stunde 5–10 nicht voraussehen und fällt stark auf den bekannten Spotpreis zurück.
3. **Positive Spikes (`spike_high` ROC-AUC `0.734`) sind schwerer vorherzusagen als negative (`0.872`):**
   * Während Wind-Überschüsse (`spike_low`) schon am Vortag gut im Wetterbericht sichtbar sind, entstehen Mangel-Spikes (`spike_high`) häufig durch kurzfristige Kraftwerks- oder Interkonnektor-Ausfälle sowie plötzliche Windflauten, die in den Day-Ahead-Features ab Stunde 3 kaum angekündigt werden.
4. **Statisches Training vor dem Test-Monat (Kein Rolling Retraining im Oktober):**
   * Aktuell wird das finale Test-Modell **einmalig** mit Daten bis zum `4. Oktober 2026` trainiert und sagt dann alle 673 Origins bis zum `1. November 2026` vorher. Es nutzt zwar zu jedem Zeitpunkt $t$ die frischen Echtzeit-Lags, passt aber seine Baumgewichte während des Oktobers nicht an das kältere Herbst-/Heizregime an.
5. **Monokultur bei der Modell-Architektur (Reines LightGBM):**
   * Alle 8 Teilmodelle basieren aktuell auf LightGBM. Es fehlt noch ein zweites, strukturell anderes Modell (z. B. `CatBoost` oder ein regularisiertes `Ridge`/`Huber`-Linearmodell) zur Varianzreduktion im Ensemble.

---

## 8. Roadmap: Wie wir jetzt konkret weiter vorgehen müssen

Wir teilen das weitere Vorgehen in **technische Modell-Upgrades (falls wir noch Zeit für weitere Score-Verbesserungen haben)** und **Pflicht-Schritte für Abgabe & Jury-Pitch** auf:

```mermaid
flowchart LR
    A["Phase 1: Abgabe-Sicherheit (Erledigt)"] --> B["Phase 2: High-Impact Modell-Upgrades (Optional, 1-2h)"]
    B --> C["Phase 3: Storyline, Jury-Report & Live-Demo (20% + 20% Note!)"]
```

### 🛠️ Phase 1: Technische High-Impact Upgrades (Was noch echte Punkte bringt)

Falls wir am Code noch weiter feilen wollen, sind das die **3 Maßnahmen mit dem höchsten Return-on-Investment**:

1. **Wöchentliches Rolling Retraining im Test-Backtest (`src/evaluation/pipeline.py`):**
   * **Idee:** Statt das Modell am `04.10.2026` einzufrieren, teilen wir die 673 Test-Origins in **4 Wochen-Blöcke** (Woche 1: `05.–11.10.`, Woche 2: `12.–18.10.`, Woche 3: `19.–25.10.`, Woche 4: `26.10.–01.11.`) und trainieren vor jeder Woche mit allen bis dahin verfügbaren Daten (`first_usable_origin_utc <= week_start`) neu.
   * **Erwarteter Gewinn:** ca. **`0,5 bis 1,2 €/MWh` besserer MAE** in den letzten zwei Oktoberwochen, weil das Modell die steigende Herbst-Heizlast lernt.
2. **Regime-Switching / Spike-Boost für `P50` in Stunde 1–2 (`src/models/forecaster.py`):**
   * **Idee:** Wenn in `Steps 1..6` sowohl `p_spike_high > 0.45` als auch `p_up > 0.55` aktiv sind, mischen wir einen kleinen Anteil (`z. B. 10–15 %`) des oberen Quantils (`P90`) in die `P50`-Prognose ein. Dadurch reagiert die `P50`-Kurve bei akuten Netz-Notfällen mutiger, ohne in den 64 % ruhigen Stunden zu verrauschen.
3. **CatBoost / Huber-Regressor Blend:**
   * **Idee:** Einen zusätzlichen `CatBoostRegressor(loss_function='MAE')` auf denselben 105 Features trainieren und `50/50` mit LightGBM mitteln. Reduziert Ausreißer-Bäume auf ungesehenen Testwochen.

---

### 🎤 Phase 2: Vorbereitung für Jury-Report & Demo (Zählt zusammen 40 % der Gesamtbewertung!)

Vergesst nicht die offizielle Punkteverteilung aus `AGENTS.md`:
* **20 % Reasoning Chain & Documentation** (Level 1 Begründung & Methodik)
* **20 % Situational Dashboard & Decision Support** (Level 2 UI)

**Unsere konkrete Checkliste für die Präsentation:**

1. **Die „Ehrliche Story“ im Pitch offensiv nutzen (Jury liebt Transparenz):**
   * Erklärt der Jury aktiv: *„Warum ist der Ausgleichspreis ab Stunde 4 so schwer vorherzusagen? Weil in Stunde 1–2 die Echtzeit-Trägheit aus `aFRR`/`mFRR` dominiert (wo unser Spezialisten-Modell den Spotpreis um fast 6 €/MWh schlägt!), während ab Stunde 4 die Vortags-Wetterprognosen dominieren. Deshalb haben wir unser System als **Horizont-Decay-Brücke** gebaut und ab Stunde 4 den Fokus auf **perfekt kalibrierte 80%-Risikobänder (`80,09 %` Coverage)** gelegt!“*
   * Damit verwandelt ihr eine physikalische Grenze des Datensatzes in den stärksten Beweis eurer Domänen-Kompetenz!
2. **Live-Demo im Dashboard (`http://localhost:8050`) vorführen:**
   * **Schritt 1:** Im Dashboard oben links auf **`Validation Period (with Actuals)`** schalten.
   * **Schritt 2:** Rechts auf den Quick-Jump-Button **`11 Sep 10:00 (+700€)`** klicken.
   * **Schritt 3:** Der Jury zeigen, wie die **Actionable Alert Box** für Schichtleiter & Trader sofort auf Rot (`HIGH UPWARD SPIKE RISK`) springt, die goldene Ist-Preiskurve im `P10–P90`-Trichter verläuft und unten rechts im **SHAP-Panel** exakt erklärt wird, *warum* das Modell gewarnt hat (Echtzeit-Spread + Residuallast).
   * **Schritt 4:** Umschalten auf **`Official Test Backtest (Oct 2026)`** und zeigen, dass alle 673 Test-Origins (26.920 Zeilen) live durchklickbar und zu 100 % validiert sind.
