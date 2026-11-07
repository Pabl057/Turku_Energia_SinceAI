# Challenge Brief — Anticipate Power System Imbalance: Build an Imbalance Price Forecast

**Event:** Since AI Hackathon 2026 (6–8 Nov 2026, Turku)  
**Challenge Partner:** Turku Energia  
**Template Contact:** Riku Lauttia · [riku.lauttia@sinceai.fi](mailto:riku.lauttia@sinceai.fi) · `+358 40 938 8820`

---

## 01 Challenge Title

> **Anticipate power system imbalance: build an imbalance price forecast**

---

## 02 Problem Statement

### Background & Market Context

Keeping the power system in balance requires production and consumption to be matched continuously. Electricity can hardly be stored in the grid, so an imbalance shows up immediately as a deviation in grid frequency. The power system corrects normal deviations automatically with **reserves**—production or consumption capacity that power system actors have reserved in advance for use—so the frequency deviation itself usually goes unnoticed by an ordinary electricity user. It only becomes a problem if the deviation grows so large that the operation of the power system has to be safeguarded, for example by disconnecting consumption.

As the share of wind and solar power grows, the balancing need of the power system varies ever more rapidly, which shows up as significant fluctuations in **imbalance prices**. An imbalance is the difference between planned and actual electricity production or consumption—in practice, a **forecast error**. If a company has planned to consume 100 units but consumes 110, an imbalance of 10 units arises.

For **market parties** (companies and other actors that buy, sell, produce, or consume electricity on the electricity market), anticipating the imbalance is challenging because the price is affected simultaneously by several factors:

- Consumption and weather conditions
- Renewable production forecasts (wind and solar)
- The transmission situation in the grid
- The availability of reserves
- The momentary balancing need of the market

The available information is scattered across different sources, and it is not easy to form an overall picture of the upcoming market situation from it.

Market parties trade on the previous day in **Day-Ahead trading** (on a power exchange such as Nord Pool) according to their consumption and production forecasts. When the forecasts are updated during the day, the forecast changes must be traded on the **Intraday markets**. If the forecast error is not corrected through intraday trading or with own resources, the remaining imbalance is settled in **imbalance settlement** at the **imbalance price**. Because electricity sales are based on the **spot price** (the bidding zone price formed on the day-ahead market), the price difference causes a financial cost to the market party, which has to be managed through well-timed and well-priced trading or by adjusting own assets (production or consumption resources).

Limited visibility into future balancing energy prices increases the risk of imbalance costs and makes it harder to plan trading, production, and demand response. Decisions often have to be made under uncertainty, without a reliable estimate of how the market will develop over the coming hours.

!!! abstract "Challenge Objective"
    In this challenge, the team's goal is to find out whether historical market, production, consumption, and weather data can be used to build an AI-based forecasting model that:

    1. **Produces a well-reasoned estimate** of the imbalance price for the next 10 hours (40 quarter-hour periods),
    2. **Identifies the most important factors** affecting the price, and
    3. **Describes the uncertainty** associated with the forecast.

---

### What Makes Up the Price to Be Forecast?

Electricity production and consumption are balanced with two balancing mechanisms: **mFRR** and **aFRR**:

- **aFRR** is the *automatically* activated frequency restoration reserve of the power system.
- **mFRR** is the *manually* activated frequency restoration reserve.

They respond in different ways and are handled on different markets. Their energy prices affect the imbalance price, which is determined separately for each **15-minute period**.

The **balancing direction** is decisive in the price calculation: is more production or less consumption needed (**upward regulation**), or the opposite (**downward regulation**)? The dominant direction is determined on the basis of mFRR balancing. In practice, balancing steers the price:

- **Upward regulation:** A higher price encourages increasing production or reducing consumption $\rightarrow$ the price is **higher** than the Day-Ahead spot price.
- **Downward regulation:** A lower price encourages reducing production or increasing consumption $\rightarrow$ the balancing price is **lower** than the Day-Ahead spot price.
- **Volume-weighted pricing (effective 1 June 2026):** When the mFRR activated in Finland and the fulfilled balancing need of aFRR are both in the dominant direction, the price is calculated as an **average weighted by their balancing volumes** (the larger balancing volume gets a larger weight in the calculation—not a sum of the prices).
- **No dominant mFRR direction:** If a quarter-hour has no dominant mFRR direction, the price is based on the **day-ahead spot price**, even if aFRR balancing has been used. In other situations, separate pricing rules apply.

---

### Scope Boundary & Turku Energia's Role

With the solution, market actors can better anticipate the balancing need of the grid and make informed operational decisions. As a scope boundary, the task is to produce **forecasts, observations, and decision support** for a person working at Turku Energia in production or electricity trading—for example, a **production shift supervisor** or an **electricity trading specialist**.

**Turku Energia's role:** Turku Energia plans electricity procurement, trading, and the operation of its available production and consumption resources based on market forecasts. When forecasts change, the company has to assess whether trading or the production plan needs to be changed and whether it is worthwhile to offer available flexible capacity to the reserve markets. Better visibility into the balancing need of the system and into future prices helps identify significant situations earlier and supports timely decision-making.

---

### Concrete Deliverables

#### Level 1: Mandatory Forecast Data

The output produces, for each upcoming 15-minute period over the next 10 hours:

- The **forecast imbalance price** (`p50`)
- An **uncertainty interval** for the forecast (`p10`, `p90`)
- An **estimate of the direction** of the system balancing need (`p_up`, `p_down`, `p_none`)
- A **probability (`0–1`), separately, of an exceptionally high and of an exceptionally low price** (exceptional = the imbalance price deviates from the spot price by more than **50 €/MWh** upwards or downwards: `p_spike_high`, `p_spike_low`)
- The **most important factors** affecting the forecast (`top_drivers`)
- Information on **when the forecast was made** (`forecast_origin_utc`) and **how fresh the data used is** (`data_cutoff_utc`)

The result is produced as a machine-readable time series in the CSV format described in the starter package (`forecast_backtest.csv`). The solution may additionally provide the result through an API, but no production integration needs to be built.

#### Level 2: Situational Overview for the User

From the forecast data, the solution surfaces only the situations that are relevant for decision-making, for example:

- The risk of a price spike has increased
- The system is expected to need upward or downward regulation
- The forecast has changed materially since the previous update
- Different data sources or models give a conflicting signal
- The situation is worth an expert review

!!! warning "Example Situational Alert"
    *"The risk of an exceptionally high price has increased for 16:00–18:00. The forecast is driven in particular by a reduced wind power forecast, increased consumption and the cross-border transmission situation. The forecast involves high uncertainty, so it is worth checking the production plan and the market position."*

The solution may also give recommendations for action, but in the first phase as **suggestions to review**—not as automatic trading or operating instructions—because open market data alone does not tell:

- What capacity Turku Energia has available at that moment
- What the marginal cost of the plant is
- How quickly the output of the plant can be changed
- What heat, contractual, or ownership constraints apply to the operation
- Whether the capacity is already committed to another market

Therefore, a solution based on open data can reliably say *"check this situation"*, but not yet with certainty *"offer this amount from this plant to this market"*. An actual capacity recommendation is a natural next development step in which Turku Energia's internal plant and cost data is brought in.

The result is a **user interface / page** that presents this information with the necessary graphical elements, plus an **API** from which this information can be retrieved for display.

---

## 03 Guiding Question

> **How can AI help anticipate the balancing need of the power system and the imbalance price accurately enough to support operational decision-making?**

---

## 04 Recommended AI Approaches *(Optional)*

Methods, models, and technologies the team considers best.

---

## 05 Dataset or API Provided

The challenge uses publicly available market, production, and consumption data as well as the spot price licensed by Turku Energia. The effect of weather is represented by Fingrid's next-day consumption, wind power, and solar power forecasts. **The Level 1 deliverable can be produced entirely with the starter package data.**

### Starter Package

Turku Energia provides the teams with a pre-loaded starter package (CSV / Parquet files) containing historical data from **1 June 2026** onwards at **15-minute resolution**:

- **Imbalance price** (the target variable to forecast) and **mFRR and aFRR balancing data**
- **Spot price (day-ahead)**, source Nord Pool
- **Finnish consumption and production data**, as well as **consumption, wind power, and solar power forecasts**
- **Cross-border flows** between Finland and Sweden (`SE1`, `SE3`), Estonia (`EE`), and Norway (`NO4`), as well as the **available transmission capacity** between Finland and Estonia
- **Imbalance prices and consumption forecasts** of neighboring areas (`SE1`, `SE3`, `EE`)
- **Calendar variables** (weekday, public holidays). In the package, the effect of weather is represented by Fingrid's next-day consumption, wind power, and solar power forecasts. The package contains no separate weather data.

The package also includes a description of each file: source, unit, time zone, and publication delay (`availability.csv`). Every data row has a column `first_usable_origin_utc`, which states the first forecast origin at which the row's information may be used. **A forecast may only use information that would have been available at the forecast origin.**

### Data Outside the Starter Package

Teams may also use other open or paid data. Each team is itself responsible for the usage rights of the data and lists all sources it has used in its submission. For example, processing of an ENTSO-E Transparency Platform API token can take several working days, so it is worth applying for one right away.

### Primary Data Sources

| Source | Data | Use in the Challenge | In the Starter Package? |
| :--- | :--- | :--- | :--- |
| **Fingrid Open Data** | Imbalance price, mFRR and aFRR balancing data, consumption and production, consumption forecast, wind power and solar power forecasts, cross-border flows between Finland and Sweden (`SE1`, `SE3`), Estonia (`EE`), and Norway (`NO4`) | Target variable to forecast and the state of the Finnish power system | **Yes** (`CC BY 4.0`) |
| **ENTSO-E Transparency Platform** | Imbalance prices and consumption forecasts of neighboring areas (`SE1`, `SE3`, Estonia) and the transmission capacity between Finland and Estonia | Market and system situation in neighboring areas | **Yes** (`CC BY 4.0`) |
| **Nord Pool** | Spot price (day-ahead), Finland | Reference level for exceptional prices, and the basis of the price in quarter-hours with no dominant mFRR direction | **Yes.** Licensed by Turku Energia: for use during the hackathon only, no redistribution and no publication in public repositories. The data must be deleted after the event. |
| **Finnish Meteorological Institute (FMI)** | Temperature, wind, solar radiation | Effect of weather on consumption and on wind and solar production | **No.** Teams can fetch weather data themselves. |
| **Calendar Data** | Weekday, time of day, public holidays, daylight saving time | Variation of consumption and price over time | **Yes** |

---

## 06 Evaluation Criteria

**Evaluation weighting:** **Level 1 = 80%**, **Level 2 = 20%**. Success at Level 1 matters most; Level 2 brings added value with which a team can stand out.

### Level 1: Forecast Quality and Reasoning (80% of Total Score)

| Sub-Criterion | Share of L1 | Evaluation Method & Description |
| :--- | :---: | :--- |
| **1. Forecast Accuracy** | **30%** | Mean Absolute Error (MAE) of the forecast imbalance price (`p50`) compared with the actual price over the test period. |
| **2. Uncertainty Calibration** | **20%** | How well the uncertainty interval (`P10–P90`) describes the actual variation. With a well-calibrated forecast, **about 80%** of actual prices fall within the interval (too narrow = overconfident; too wide = useless). Scored using the mean **pinball loss** of the `P10` and `P90` forecasts. |
| **3. Identification of Exceptional Prices** | **20%** | How well the model anticipates quarter-hours in which the imbalance price deviates significantly from the spot price (**by more than 50 €/MWh upwards or downwards**), scored via Brier score. |
| **4. Direction of the Balancing Need** | **10%** | Accuracy (3-class Brier score) of the forecast dominant balancing direction (`up` / `down` / `none`), based on mFRR activations (Fingrid dataset `369`). |
| **5. Description of the Reasoning Chain** | **20%** | Clarity, consistency, and inventiveness of the description of how the data sources are used and of the factors affecting the forecast. Assessed by the jury based on the evaluation report, presentation slides, and demo video. |

> *The forecast is not expected to be right all the time. What matters is that the uncertainty is calibrated and that the solution recognizes situations in which the error may be large.*

### Forecast Verification & Anti-Leakage Rule

!!! danger "Mandatory Anti-Leakage Rule (`first_usable_origin_utc`)"
    Forecasts are evaluated over the test period (`5 Oct – 1 Nov 2026`). The starter package also contains the input data for the test period, while the actual values used in scoring are held only by the organizer.

    A forecast may only use information that would have been available at the forecast origin. A data row of the starter package may be used **only from its `first_usable_origin_utc` time onwards**. **If a row is used before this time, the submission is not a forecast, and its automatic evaluation score is 0.**

### Scoring Process

- **Two official scripts** are shared in the starter package: `validate_submission.py` (checks the format of the submission file) and `score.py` (calculates the metrics), along with `baseline.ipynb`.
- Each team checks its submission with `validate_submission.py` before submitting. Validation checks structural compliance, not forecast quality.
- Teams can calculate metrics themselves for the **validation period (`7 Sep – 4 Oct 2026`)** using `score.py` and `data/actuals_validation.csv`.
- The organizer calculates the official results against the actual values of the test period with the same locked scripts for all submissions.
- The source code of the best solutions will be reviewed (potentially assisted by AI tools) to ensure that no data was used before its `first_usable_origin_utc`.

### Level 2: Situational Overview and Usability (20% of Total Score)

- **Clarity and user-friendliness** of the user interface for an energy sector expert (e.g., a production shift supervisor or an electricity trading specialist).
- **Two time horizons:** What is worth checking right now and over the next 10 hours, and how the forecast has performed in hindsight (e.g., the previous day, week, or month).
- **Time-travel capability:** The ability to move the point of review into the past, so that the usefulness of the solution can be demonstrated over time and not only as real-time monitoring.
- **Inventiveness:** Insights that give energy sector experts new information.

---

## 07 Submission Format

The solution must produce a forecast of the imbalance price for the next 10 hours in 15-minute periods. During the test period, the forecast is made on the hour. In addition to the price estimate, the forecast must include an uncertainty interval, an estimate of the probability of significant price changes, and the most important factors affecting the forecast.

The hackathon does not require automatic trading, plant control, or production-ready integrations. The solution must be evaluable with the starter package data. Submissions are made via the **Since AI platform**.

### Level 1 Deliverables (Required)

- [x] **Machine-readable forecast file** (`forecast_backtest.csv`) in the format described in the starter package, checked with `validate_submission.py` before submission
- [x] **Source code and instructions** for running it (`INSTRUCTIONS.md`)
- [x] **Short evaluation report** (`docs/EVALUATION_REPORT.md`): reasoning chain, factors affecting the forecast, and validation period metrics
- [x] **List of data sources, external services, ready-made models, and software components** used, and their usage rights (`docs/DATA_MODELS_LICENSES.md`)
- [ ] **About 5–8 presentation slides**
- [ ] **Demo video of at most 5 minutes**, presenting the forecast for the next 10 hours in 15-minute periods, the forecast uncertainty, significant deviations, and the most important factors affecting the forecast

### Level 2 Added Value in Demo

- [x] A user-friendly view or alerts about situations in which the balancing need or the price forecast changes materially (`apps/dashboard.py`)
- [x] How the forecast has performed in hindsight and the ability to move the point of review into the past (*Historical Time Machine*)
- [x] A reasoned suggestion of situations in which Turku Energia should reconsider its trading, production plan, or use of capacity

---

## 08 Points of Contact & Event Schedule

### Turku Energia Contacts

| Role / Area | Name | Email / Phone | Discord Handle |
| :--- | :--- | :--- | :--- |
| **Coordination** (Business Technology Manager) | Anne-Mari "Ama" Angervo | [annemari.angervo@turkuenergia.fi](mailto:annemari.angervo@turkuenergia.fi) · `040 738 7073` | `te_ama_31257` |
| **Electricity Market Operations** (Trading Specialist) | Ville Kuusisto | [ville.kuusisto@turkuenergia.fi](mailto:ville.kuusisto@turkuenergia.fi) | `te_ville_14208` |
| **Production** (Production Manager) | Kalle Karjala | [kalle.karjala@turkuenergia.fi](mailto:kalle.karjala@turkuenergia.fi) | `te_kalle` |
| **AI, Data, Technology & Implementation** (Development Specialist) | Vesa Lukka | [vesa.lukka@turkuenergia.fi](mailto:vesa.lukka@turkuenergia.fi) | `te_vesa_99724` |
| **Communications** | Nora Forsman *(Fri)* / Vilma Matero *(Sat–Sun)* | — | — |

### On-Site Schedule at EduCity (Turku)

=== "Friday, 6 Nov 2026"
    - **From 15:30:** On site at EduCity: Ville, Kalle, Vesa, Ama + communications (Nora or Vilma)
    - **17:00:** Opening ceremony — Since AI presents the challenges
    - **18:50–20:00:** Answering questions in the Turku Energia challenge room (**Room 3006**)
    - Possibility of dinner

=== "Saturday, 7 Nov 2026"
    - **09:00–12:00:** On site: Ville, Kalle, Ama
    - **14:30–18:00:** On site: Ville, Kalle, Vesa + communications
    - Possibility to eat during the event
    - At other times, mentors answer questions via Discord as far as possible

=== "Sunday, 8 Nov 2026"
    - **08:45:** Arrival at EduCity; **09:00** briefing *(on site: Ama, Vesa)*
    - **10:00–13:00:** Evaluation of solutions (Places 1, 2, and 3) *(on site: Ville, Kalle, Vesa, Ama)*
    - **13:30:** Top 3 announced
    - **14:00–15:30:** Final presentations *(on site: everyone + communications)*

---

## 09 Glossary

| Term | Plain-Language Explanation |
| :--- | :--- |
| **Power system balance** | Electricity must be produced at every moment in the same amount as it is consumed. If production and consumption do not match, the frequency of the power system starts to deviate from its target level (50 Hz). |
| **Imbalance** | The difference between the planned and the actually realized electricity production or consumption. For example, if a company has planned to consume 100 units but consumes 110, an imbalance of 10 units arises. |
| **Market party** | A company or other actor that buys, sells, produces, or consumes electricity on the electricity market. Turku Energia acts in different roles within this whole. |
| **Balance responsible party (BRP)** | A market party that is financially responsible for ensuring that the electricity procurement and delivery under its responsibility are in balance. Imbalances are settled in imbalance settlement. Turku Energia is not necessarily the balance responsible party itself for every plant or operation. |
| **Imbalance price** | The price at which the imbalance between plan and actual is settled. The price may be unfavorable compared with having bought or sold the required electricity on the market in time. |
| **Reserve** | Production or consumption capacity reserved in advance for use, which can be activated quickly to balance the power system. For example, a power plant may commit to being able to increase or decrease its output when needed. |
| **Balancing energy** | Electricity that is added or reduced to balance the system. Upward regulation means increasing production or reducing consumption, and downward regulation the opposite. Same as mFRR. |
| **mFRR and aFRR** | Different frequency restoration reserves. **aFRR** is the *automatically* activated frequency restoration reserve and **mFRR** the *manually* activated frequency restoration reserve. They respond in different ways and are handled on different markets. |
| **Spot price** | The bidding zone price usually formed on the next-day electricity market (the day-ahead market). It should not be confused with intraday prices or imbalance prices. |
| **Day-ahead market** | A market on which electricity is bought and sold for the hours or quarter-hour periods of the next day, based on the forecasts available. |
| **Intraday market** | The within-day market on which earlier plans can be corrected closer to the actual delivery time of the electricity. |
| **Asset (production or consumption resource)** | A physical resource whose production or consumption can be controlled. A production asset can be, for example, a power plant; a consumption asset can be an electric boiler, a heat pump, a battery, or a flexible industrial load. |
| **Imbalance cost** | The financial cost arising from an imbalance. It depends on the volume, the direction, and the price of the imbalance. |
| **Demand response** | Increasing, reducing, or shifting consumption to another time based on the price or the needs of the power system. Total consumption does not necessarily have to decrease; it can instead be timed better. |
