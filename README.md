# Intelligent Energy & Equipment Monitoring — YUKTHI 2026

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501),
and upload a CSV conforming to the Data Specification (11 fields:
`timestamp`, `equipment_id`, `Chilled Water Rate (L/sec)`,
`Cooling Water Temperature (C)`, `Building Load (RT)`,
`Chiller Energy Consumption (kWh)`, `Outside Temperature (F)`,
`Dew Point (F)`, `Humidity (%)`, `Wind Speed (mph)`, `Pressure (in)`).

## Architecture

```
Operational Data (CSV)
    │
    ▼
1. Ingestion & Validation  (pipeline.py: load_and_validate)
    - Schema check against Data Specification
    - Per-equipment chronological sort
    │
    ▼
2. Missing Value Handling  (handle_missing_values)
    - Time-based interpolation, independently per equipment_id
    - No cross-equipment leakage
    │
    ▼
3. Feature Engineering  (engineer_features)
    - Cyclic time features (hour, day-of-week)
    - Efficiency proxy (kWh / RT)
    - Causal rolling stats (24h window: mean/std of load, efficiency)
    - Lag feature (previous energy reading)
    - Gap-aware: temporal gaps are tracked, not treated as anomalies
    │
    ▼
4. Expected-Behavior Model  (train_expected_behavior_models)
    - One RandomForestRegressor per equipment_id
    - Predicts energy consumption from load + weather + time
    - This IS the "understanding of expected behaviour under
      relevant operating and environmental conditions" (Spec Sec. 4)
    - Validated with time-respecting split (no future leakage)
    │
    ▼
5. Contextual Anomaly Scoring  (compute_residuals, score_anomalies)
    - residual = actual − predicted (the ML contribution: deviation
      is computed relative to conditions, not a fixed threshold)
    - IsolationForest on residuals + context → anomaly_score, severity
    │
    ▼
6. Persistence Detection  (add_persistence_flag)
    - Distinguishes isolated blips from sustained abnormal conditions
    │
    ▼
7. Streamlit Application  (app.py)
    - Time Series tab: actual vs expected energy, anomalies overlaid
    - Anomaly Table tab: sortable/filterable, exportable to CSV
    - Drill-Down tab: evidence view — conditions at the anomaly vs.
      typical conditions for similar load, ±12h context chart,
      plain-language interpretation (isolated vs persistent)
```

## Design choices mapped to requirements

| Requirement (Problem Statement / Data Spec) | How it's addressed |
|---|---|
| ML must be a meaningful component, not thresholds | Two-stage ML: regression for expected behavior + IsolationForest for anomaly scoring on residuals |
| Contextual anomaly detection (Sec. 4) | Anomalies are scored on deviation from *predicted* energy given load/weather, not raw value thresholds |
| No hard-coded observations/dataset assumptions (Sec. 10) | Pipeline operates on `equipment_id`/`timestamp` programmatically; schema-validated; works for any conforming CSV |
| Gaps ≠ anomalies (Data Spec Sec. 5) | `gap_hours` tracked as a feature; missing-value interpolation is time-aware, not gap-blind |
| No leakage (Data Spec Sec. 8) | Rolling/lag features are strictly causal; train/val split is chronological |
| Persistent vs isolated deviations (Sec. 4) | `persistent_anomaly` flag via rolling anomaly count |
| Evidence-based insights (Sec. 8) | Drill-down tab shows actual vs. typical conditions for comparable load, plus surrounding time context |
| Functional application, not just a notebook (Sec. 8) | Streamlit app with upload, filters, three interactive views |

## Known limitations / honest caveats

- Validation R² for the expected-behavior model is moderate (~0.5–0.9
  depending on equipment and season) using time-series cross-validation.
  This is partly due to genuine seasonal extrapolation (training data
  skews toward cooler months) and may partly reflect real behavioral
  drift in the equipment — which itself is a relevant finding worth
  discussing in your presentation, not just an error to fix.
- Severity thresholds (0.85 / 0.70 anomaly-score cutoffs) are a
  reasonable starting point tuned on this dataset's score distribution,
  not derived from a labeled ground truth (none is provided, per Data
  Spec Sec. 9). Consider justifying/adjusting them live if asked.
- IsolationForest `contamination=0.03` is an assumption (~3% of data
  flagged) — worth being ready to explain why, and how it'd change with
  a different value.
