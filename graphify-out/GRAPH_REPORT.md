# Graph Report - Turku_Energia_SinceAI  (2026-11-07)

## Corpus Check
- Corpus is ~32,601 words - fits in a single context window. You may not need a graph.

## Summary
- 279 nodes · 661 edges · 16 communities (12 shown, 4 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 38 edges (avg confidence: 0.88)
- Token cost: 53,000 input · 9,000 output

## Community Hubs (Navigation)
- Template Generation & Submission Validation
- Availability Rules & Dataset Assembly
- Challenge Rules & Forecasting Architecture
- Entire.io OpenCode Plugin & Workflow
- Submission Scoring & Evaluation Metrics
- Entire.io Pi Extension & Session Hooks
- Fingrid & ENTSO-E Series Definitions
- Fingrid & ENTSO-E API Fetching
- Time Window & Duration Parsing
- ENTSO-E XML Parsing
- Environment Config & Secret Masking
- Dataset Builder CLI Entrypoint
- Nord Pool Spot Timezone Conversion
- Rate-Limited HTTP Client
- Helsinki Monthly Window Chunking

## God Nodes (most connected - your core abstractions)
1. `run()` - 31 edges
2. `entireSetup()` - 25 edges
3. `fetch_entsoe()` - 19 edges
4. `EntirePlugin()` - 18 edges
5. `handleEvent()` - 16 edges
6. `BuildError` - 16 edges
7. `read_spot_file()` - 15 edges
8. `fetch_fingrid()` - 14 edges
9. `validate()` - 14 edges
10. `parse_entsoe()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `Prediction Post-Processing Constraints` --semantically_similar_to--> `validate()`  [INFERRED] [semantically similar]
  .agents/skills/turku-imbalance-forecasting/SKILL.md → validate_submission.py
- `24-Hour Timeline & Publication Lag Causality` --semantically_similar_to--> `compute_first_usable()`  [INFERRED] [semantically similar]
  PROJECT_GUIDE.md → build_dataset.py
- `Entire.io CLI (entire)` --conceptually_related_to--> `EntirePlugin()`  [INFERRED]
  .agents/skills/entire-workflow/SKILL.md → .opencode/plugin/entire.ts
- `Rule #2: Nord Pool Spot Price Licensing (spot_nordpool.csv)` --conceptually_related_to--> `read_spot_file()`  [INFERRED]
  AGENTS.md → build_dataset.py
- `Level 1 Deliverable: 10-Hour Probabilistic Forecast (80% Weight)` --conceptually_related_to--> `score_submission()`  [INFERRED]
  CHALLENGE_BRIEF.md → score.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Anti-Leakage Data Availability Protocol (first_usable_origin_utc <= origin)** — agents_rule1_first_usable_origin_utc, agents_skills_turku_imbalance_forecasting_skill_rows_available_at, project_guide_temporal_causality, readme_inputs_long_schema, build_dataset_compute_first_usable [EXTRACTED 1.00]
- **Submission Validation and Official Scoring Pipeline** — agents_rule3_forecast_backtest_csv, agents_skills_turku_imbalance_forecasting_skill_postprocessing_constraints, validate_submission_validate, score_score_submission, readme_scoring_formula [EXTRACTED 1.00]
- **Imbalance Price Formation & LightGBM Spread Forecasting Architecture** — challenge_brief_imbalance_price_mechanism, project_guide_imbalance_spread_strategy, project_guide_feature_matrix_x, project_guide_six_lightgbm_models, agents_skills_turku_imbalance_forecasting_skill_shap_top_drivers [INFERRED 0.95]

## Communities (16 total, 4 thin omitted)

### Community 0 - "Template Generation & Submission Validation"
Cohesion: 0.08
Nodes (16): build_origins(), build_template(), main(), parse_args(), Dataset Schedule (Training, Validation, Test & 673 Forecast Origins), Finding, load_origins(), main() (+8 more)

### Community 1 - "Availability Rules & Dataset Assembly"
Cohesion: 0.13
Nodes (22): _atomic_write_bytes(), AvailabilityRule, build_actuals(), build_calendar(), compute_first_usable(), expand_fingrid_rows(), find_rule(), fmt_utc() (+14 more)

### Community 2 - "Challenge Rules & Forecasting Architecture"
Cohesion: 0.07
Nodes (17): AGENTS.md Instruction Set, Level 2 Situational Dashboard & Time Machine, Rule #3: Submission File Integrity (forecast_backtest.csv), Turku Energia Imbalance Price Forecasting Skill, SHAP Explainability (top_drivers), Turku Energia Challenge Brief (Since AI Hackathon 2026), Level 1 Deliverable: 10-Hour Probabilistic Forecast (80% Weight), Turku Energia (Challenge Partner) (+9 more)

### Community 3 - "Entire.io OpenCode Plugin & Workflow"
Cohesion: 0.10
Nodes (22): Checkpoints, Sessions & Trails, Entire.io Agent Workflow Skill, Entire.io CLI (entire), EntirePlugin(), announcedKey(), announceTask(), backgroundSessionAborted(), callHook() (+14 more)

### Community 4 - "Submission Scoring & Evaluation Metrics"
Cohesion: 0.16
Nodes (18): compute_metrics(), compute_points(), criterion_points(), fmt(), load_actuals(), load_submission(), main(), _mean_or_nan() (+10 more)

### Community 5 - "Entire.io Pi Extension & Session Hooks"
Cohesion: 0.20
Nodes (25): entireSetup(), announceTask(), callKey(), classify(), drainBackgroundTasks(), endSession(), endSessions(), endTurn() (+17 more)

### Community 6 - "Fingrid & ENTSO-E Series Definitions"
Cohesion: 0.18
Nodes (9): _capacity(), easter_sunday(), EntsoeSeries, _expected_quarters(), finnish_holidays(), _imbalance(), _load_fc(), _saturday_between() (+1 more)

### Community 7 - "Fingrid & ENTSO-E API Fetching"
Cohesion: 0.24
Nodes (8): compact(), fetch_entsoe(), fetch_fingrid(), FingridSeries, _iso_z(), KeyStore, RawCache, Starter Package Open & Licensed Data Sources (Fingrid, ENTSO-E, Nord Pool, FMI)

### Community 8 - "Time Window & Duration Parsing"
Cohesion: 0.33
Nodes (3): BuildError, _duration_minutes(), parse_window_arg()

### Community 9 - "ENTSO-E XML Parsing"
Cohesion: 0.43
Nodes (4): _child(), _children(), _local(), parse_entsoe()

### Community 11 - "Dataset Builder CLI Entrypoint"
Cohesion: 0.33
Nodes (3): log(), main(), parse_args()

### Community 12 - "Nord Pool Spot Timezone Conversion"
Cohesion: 0.40
Nodes (3): local_to_utc_by_order(), _parse_local_naive(), SpotFileError

## Knowledge Gaps
- **3 isolated node(s):** `Checkpoints, Sessions & Trails`, `Turku Energia (Challenge Partner)`, `Darts Time-Series Library (u8darts)`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 62 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AGENTS.md Instruction Set` connect `Challenge Rules & Forecasting Architecture` to `Template Generation & Submission Validation`, `Entire.io OpenCode Plugin & Workflow`, `Submission Scoring & Evaluation Metrics`, `Fingrid & ENTSO-E Series Definitions`?**
  _High betweenness centrality (0.393) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `fetch_entsoe()` (e.g. with `.dataset_id()` and `.get()`) actually correct?**
  _`fetch_entsoe()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Checkpoints, Sessions & Trails`, `Turku Energia (Challenge Partner)`, `Darts Time-Series Library (u8darts)` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Template Generation & Submission Validation` be split into smaller, more focused modules?**
  _Cohesion score 0.08416389811738649 - nodes in this community are weakly interconnected._
- **Why does `Entire.io Agent Workflow Skill` connect `Entire.io OpenCode Plugin & Workflow` to `Challenge Rules & Forecasting Architecture`?**
  _High betweenness centrality (0.326) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `EntirePlugin()` (e.g. with `Entire.io CLI (entire)` and `entire.ts`) actually correct?**
  _`EntirePlugin()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Should `Availability Rules & Dataset Assembly` be split into smaller, more focused modules?**
  _Cohesion score 0.13174603174603175 - nodes in this community are weakly interconnected._