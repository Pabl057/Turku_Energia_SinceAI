# High-Level Overview — Presentation-Ready Architecture & Functionality

This document provides clean, **slide-ready Mermaid diagrams** and concise explanations at a high level of abstraction. Every diagram is designed so it can be screenshotted or copied directly onto a presentation slide.

---

## 1. The Big Picture (4-Stage Pipeline)

At any hourly forecast origin $t$, our system transforms raw grid and market data into a calibrated 10-hour (40 quarter-hour) probabilistic forecast and actionable decision support in four steps:

```mermaid
flowchart LR
    A["1. Market & Grid Data<br/>Fingrid (incl. RT 181/188) · ENTSO-E · Nord Pool<br/>(Strict Anti-Leakage Filter)"] --> B["2. Physical Feature Engine<br/>197 Features Combining<br/>Real-Time Stress + D-1 Plans"]
    B --> C["3. Calibrated AI Suite<br/>Dual-Depth Adaptive Stacking +<br/>Conformal 80% Uncertainty + Birnbaum XAI"]
    C --> D["4. Decision Support<br/>forecast_backtest.csv +<br/>Situational Desk & AI Copilot"]
```

| Stage | What It Does | Why It Matters |
| :--- | :--- | :--- |
| **1. Market & Grid Data** | Loads Fingrid (including 3-min RT Wind `181`, Nuclear `188`, Production `191`, Grid State `177`, and pre-June `aFRR/mFRR` lag buffer), ENTSO-E, and Nord Pool series filtered by `first_usable_origin_utc <= t`. | Guarantees **100% zero future data leakage** (`Rule #1`). |
| **2. Physical Feature Engine** | Combines **what just happened on the grid** (last 30–120 min, including real-time wind forecast error & implied aFRR volume) with **what is scheduled for tomorrow** (known Day-Ahead spot price, wind, solar, and load). | Captures both short-term grid inertia and long-term supply/demand ramps. |
| **3. Calibrated AI Suite** | Predicts the price deviation ($\Delta P$) via a dual-depth adaptive stacking + short-horizon ensemble, calibrates the 80% uncertainty band (`P10–P90`), couples spike risk to regulation direction, and explains drivers via **TreeSHAP** & **Birnbaum Reliability ($I_B, I_{CR}, I_{FV}$)**. | Reduces error by **34.8% (MAE)** and **48.6% (Pinball)** while hitting **exact 80.0% coverage** ($p < 10^{-100}$). |
| **4. Decision Support** | Outputs the validated submission (`forecast_backtest.csv`) and powers the interactive **Level 2 Situational Desk, Time Machine, Birnbaum XAI & AI Copilot**. | Gives shift supervisors and traders early spike warnings and concrete review suggestions. |

---

## 2. Core Forecasting Idea: Predict the Spread ($\Delta P$), Not Raw Price

Under Finnish market rules, the day-ahead **Spot Price** is already locked in at 14:00 the day before, and whenever there is no dominant mFRR balancing need, the **Imbalance Price equals the Spot Price**. Instead of guessing raw prices from scratch, our model predicts how much the grid will **deviate** from the known spot price:

```mermaid
flowchart LR
    SPOT["Known Day-Ahead Spot Price<br/>(Published Yesterday at 14:00)"] --> PLUS(("+"))
    STRESS["Real-Time Grid Stress +<br/>Wind / Load Forecast Errors"] --> ML["LightGBM Suite<br/>Predicts Price Spread (ΔP)"]
    ML --> PLUS
    PLUS --> OUT["10-Hour Imbalance Forecast<br/>P10 · P50 · P90 · Spikes · Direction"]
```

$$P_{\text{imbalance}}(t + k) = \underbrace{P_{\text{spot}}(t + k)}_{\text{Known since D-1 14:00}} + \underbrace{\Delta P(t + k)}_{\text{Predicted by AI}}$$

---

## 3. Bridging the 10-Hour Horizon (Two Physical Regimes)

A 10-hour forecast (40 steps of 15 minutes) spans two very different physical regimes. Our architecture blends specialized models for each:

```mermaid
flowchart TB
    subgraph INPUTS["What We Know at Forecast Origin t"]
        PAST["Realized Grid State (Before t)<br/>• t - 30m: Actual Load Error & Border Flows<br/>• t - 60m: Realized Imbalance, mFRR & aFRR<br/>• t - 120m: Sweden & Estonia Imbalance"]
        FUTURE["Published Day-Ahead Plan (For t + 15m to t + 10h)<br/>• Nord Pool Spot Price & 1h/3h Ramps<br/>• Fingrid Wind, Solar & Load Forecasts<br/>• Estlink (FI–EE) Transfer Capacity"]
    end

    PAST ==>|"Strongest Driver"| H1["Hours 1–2 (Steps 1–8)<br/>Short-Horizon Grid Inertia<br/>(Active balancing stress persists)"]
    FUTURE ==>|"Strongest Driver"| H2["Hours 3–10 (Steps 9–40)<br/>Forward Structural Regime<br/>(Evening load peaks & wind ramps)"]

    H1 & H2 --> BLEND["Blended 40-Step Forecast Trajectory"]
```

- **Hours 1–2 (`Steps 1–8` — Grid Inertia):** Weather timing errors and plant outages persist for 1–2 hours. Dedicated **Short-Horizon Specialists** focus on the latest 30m consumption forecast error and 60m mFRR/aFRR activations.
- **Hours 3–10 (`Steps 9–40` — Forward Structure):** Real-time persistence fades. **Full-Horizon Models** rely on the known D-1 spot curve, residual load, and 3-hour wind ramps—beating the baseline by **40.1%** at Hour 10.

---

## 4. How the AI Model Suite Works Together

Instead of a single black-box model, five specialized modules produce a mathematically consistent, explainable forecast for every 15-minute step:

```mermaid
flowchart TD
    FEAT["197 Leak-Free Features per Step"] --> M1 & M2 & M3 & M4

    M1["1. Point Forecast (p50)<br/>Dual-Depth Adaptive Stacking (60/40) + Short-Horizon L1<br/>→ -34.8% MAE (43.19 €/MWh)"]
    M2["2. Uncertainty Band (p10, p90)<br/>Quantile GBDT + Conformal Calibration<br/>→ -48.6% Pinball & Exact 80.0% Coverage"]
    M3["3. Spike Detector (±50 €/MWh)<br/>Classifier + Tail Blend + Direction Coupling<br/>→ 0.0928 Brier (-18.0% vs Baseline)"]
    M4["4. Balancing Direction (up/down/none)<br/>3-Class GBDT + Base-Rate Shrinkage<br/>→ 0.4080 Brier (-9.5% vs Baseline)"]

    M1 --> SHAP["5. Exact TreeSHAP & Birnbaum Reliability XAI<br/>Top 3 Drivers per Step + Subsystem Criticality (I_B / I_CR / I_FV)"]
    M1 & M2 & M3 & M4 & SHAP --> GUARD["Post-Processing Guardrails<br/>p10 ≤ p50 ≤ p90 · Probabilities Sum to 1.0"]
```

### Key Innovations in Plain Language
1. **Split Conformal Calibration (`p10`, `p90`):** Standard decision trees are overconfident (capturing only ~68% of actual prices). We scale the interval width using recent out-of-sample errors so **exactly 80.0%** of actual prices fall inside `[P10, P90]`.
2. **Uncertainty-Aware & Direction-Coupled Spike Probabilities (`p_spike_high`, `p_spike_low`):** Combines a dedicated spike classifier with the width of our `P10–P90` interval and physical mFRR regulation direction (`up` for positive spikes, `down` for negative spikes).
3. **TreeSHAP & Birnbaum Reliability XAI (`top_drivers` + $I_B, I_{CR}, I_{FV}$):** Every forecast row names the top 3 physical factors driving that specific quarter-hour, while the dashboard decomposes grid spike risk into Reliability Engineering importance measures and a 40-step Temporal Driver Attention Heatmap.

---

## 5. Level 2 Situational Awareness Desk & Time Machine

The Level 2 Streamlit application (`apps/dashboard.py`) turns the 40-step forecast into an operational cockpit for **Production Shift Supervisors** and **Electricity Traders**:

```mermaid
flowchart LR
    FC["40-Step Forecast<br/>+ TreeSHAP & Birnbaum XAI"] --> TRIAGE["Automated Risk Triage<br/>• Spike Risk ≥ 30%<br/>• Uncertainty Band ≥ 120 €<br/>• Forecast Shift |Δp50| ≥ 15 €"]

    TRIAGE --> DESK["Live Situational Desk<br/>Fan Chart + Alert Banner +<br/>Birnbaum Reliability & Temporal Heatmap"]
    TRIAGE --> TM["Historical Time Machine & AI Copilot<br/>Step Through 1,345 Past Origins +<br/>Ablation & Wilcoxon Significance Proof"]
```

| User Persona | What They See | Example Suggested Action to Review |
| :--- | :--- | :--- |
| **Production Shift Supervisor** *(Power plants, electric boilers, heat pumps)* | 10-hour P10–P50–P90 price fan chart, high/low spike alert banners, and dominant mFRR direction. | **High Up-Spike Risk (`>30%`):** Consider deferring flexible electric boiler / heat pump consumption or checking fast-start reserve readiness.<br/>**Low / Negative Price Risk:** Opportunity to absorb cheap surplus electricity in district heating boilers or thermal storage. |
| **Electricity Trading Specialist** *(Intraday & balancing markets)* | Hour-over-hour forecast revision deltas ($t - 1\text{h} \to t$), TreeSHAP drivers, Birnbaum subsystem criticality, and cross-border flow telemetry. | **Material Upward Revision (`Δp50 ≥ +15 €/MWh`):** Review intraday short exposure before gate closure as real-time Fingrid consumption error tightens the grid. |

---

## 6. Headline Results at a Glance (Slide Scorecard)

| Evaluation Criterion | Official Weight | Starter Baseline | Our Solution | Impact & Statistical Significance |
| :--- | :---: | ---: | ---: | :--- |
| **Point Accuracy (`p50` MAE)** | **30%** | 66.20 €/MWh | **43.19 €/MWh** | **-34.8% error** (**39.44 €/MWh** at Hour 1; $t=27.46,\ p<10^{-100}$) |
| **Uncertainty (`p10`–`p90` Pinball)** | **20%** | 24.18 | **12.44** | **-48.6% pinball loss** ($t=29.26,\ p<10^{-100}$) |
| **P10–P90 Interval Coverage** | *(Target 80%)* | 68.29% | **80.00%** | **Exact 80.0% calibration target** |
| **Spike Identification (`±50 €` Brier)** | **20%** | 0.1132 | **0.0928** | **-18.0% Brier** (`p_spike_low: 0.0862`; $t=37.87,\ p<10^{-100}$) |
| **Balancing Direction (3-Class Brier)** | **10%** | 0.4509 | **0.4080** | **-9.5% Brier** (**0.3608** at Hour 1; $t=32.95,\ p<10^{-100}$) |
