"""
YUKTHI 2026 - Intelligent Energy & Equipment Monitoring
Data pipeline + contextual anomaly detection for chiller operational data.

Design principles (per Data Specification):
- No hard-coded row counts, timestamps, or dataset-specific assumptions.
- equipment_id treated as categorical identifier; each unit is its own
  chronological series.
- Missing values handled per-equipment, without cross-equipment leakage.
- Temporal gaps are NOT treated as anomalies; features are gap-aware.
- Rolling/lag features are strictly causal (no future leakage).
- No exposed target column is assumed to exist - unsupervised approach.
"""
"""
YUKTHI 2026 - Intelligent Energy & Equipment Monitoring
Data pipeline + contextual anomaly detection for chiller operational data.

Design principles (per Data Specification):
- No hard-coded row counts, timestamps, or dataset-specific assumptions.
- equipment_id treated as categorical identifier; each unit is its own
  chronological series.
- Missing values handled per-equipment, without cross-equipment leakage.
- Temporal gaps are NOT treated as anomalies; features are gap-aware.
- Rolling/lag features are strictly causal (no future leakage).
- No exposed target column is assumed to exist - unsupervised approach.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. INGESTION & VALIDATION
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    'timestamp': 'datetime',
    'equipment_id': 'categorical',
    'Chilled Water Rate (L/sec)': 'numeric',
    'Cooling Water Temperature (C)': 'numeric',
    'Building Load (RT)': 'numeric',
    'Chiller Energy Consumption (kWh)': 'numeric',
    'Outside Temperature (F)': 'numeric',
    'Dew Point (F)': 'numeric',
    'Humidity (%)': 'numeric',
    'Wind Speed (mph)': 'numeric',
    'Pressure (in)': 'numeric',
}

def load_and_validate(path: str) -> pd.DataFrame:
    """Load CSV and validate it conforms to the Data Specification schema."""
    df = pd.read_csv(path)

    missing_cols = set(REQUIRED_COLUMNS.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['equipment_id'] = df['equipment_id'].astype(str)

    numeric_cols = [c for c, t in REQUIRED_COLUMNS.items() if t == 'numeric']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Sort chronologically WITHIN each equipment unit (per spec 3.1)
    df = df.sort_values(['equipment_id', 'timestamp']).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# 2. MISSING VALUE HANDLING (per-equipment, gap-aware, no leakage across units)
# ---------------------------------------------------------------------------

MEASUREMENT_COLS = [
    'Chilled Water Rate (L/sec)',
    'Cooling Water Temperature (C)',
    'Building Load (RT)',
    'Chiller Energy Consumption (kWh)',
    'Outside Temperature (F)',
    'Dew Point (F)',
    'Humidity (%)',
    'Wind Speed (mph)',
    'Pressure (in)',
]

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate missing values in time order, independently per equipment_id.
    Uses time-based linear interpolation (respects actual gap size), with
    forward/backward fill only at series edges.
    """
    out = []
    for eq_id, group in df.groupby('equipment_id', sort=False):
        g = group.set_index('timestamp').sort_index()
        for col in MEASUREMENT_COLS:
            g[col] = g[col].interpolate(method='time', limit_direction='both')
        g['equipment_id'] = eq_id
        out.append(g.reset_index())
    result = pd.concat(out, ignore_index=True)
    return result.sort_values(['equipment_id', 'timestamp']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING (causal only - no future leakage)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time features, efficiency proxy, and causal rolling/lag statistics.
    All rolling windows use only past data (min_periods allows partial windows
    at series start rather than dropping rows).
    """
    df = df.copy()

    # --- Time features (cyclic encoding for hour to capture periodicity) ---
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # --- Efficiency proxy: kWh per RT of cooling delivered ---
    # Guard against divide-by-zero / near-zero load (idle periods)
    df['efficiency_kw_per_rt'] = np.where(
        df['Building Load (RT)'] > 1.0,
        df['Chiller Energy Consumption (kWh)'] / df['Building Load (RT)'],
        np.nan
    )

    # --- Per-equipment causal rolling/lag features ---
    feat_frames = []
    for eq_id, group in df.groupby('equipment_id', sort=False):
        g = group.sort_values('timestamp').copy()

        # Time since previous observation (hours) - lets model account for gaps
        g['gap_hours'] = g['timestamp'].diff().dt.total_seconds() / 3600.0
        g['gap_hours'] = g['gap_hours'].fillna(0.5)  # nominal interval at start

        # Rolling stats over ~24h (48 obs at 30-min interval), causal (shift(1)
        # not needed since rolling() with default is already trailing/causal)
        window = 48
        g['load_roll_mean_24h'] = (
            g['Building Load (RT)'].rolling(window, min_periods=6).mean()
        )
        g['load_roll_std_24h'] = (
            g['Building Load (RT)'].rolling(window, min_periods=6).std()
        )
        g['efficiency_roll_mean_24h'] = (
            g['efficiency_kw_per_rt'].rolling(window, min_periods=6).mean()
        )
        g['efficiency_roll_std_24h'] = (
            g['efficiency_kw_per_rt'].rolling(window, min_periods=6).std()
        )

        # Lag feature: previous observation's energy consumption
        g['energy_lag1'] = g['Chiller Energy Consumption (kWh)'].shift(1)

        feat_frames.append(g)

    result = pd.concat(feat_frames, ignore_index=True)

    # Backfill remaining NaNs in rolling features (start-of-series) with
    # per-equipment median so early rows aren't dropped
    roll_cols = ['load_roll_mean_24h', 'load_roll_std_24h',
                 'efficiency_roll_mean_24h', 'efficiency_roll_std_24h',
                 'energy_lag1']
    for col in roll_cols:
        result[col] = result.groupby('equipment_id')[col].transform(
            lambda s: s.fillna(s.median())
        )

    return result


# ---------------------------------------------------------------------------
# 4. EXPECTED-BEHAVIOR MODEL (per-equipment regressor)
# ---------------------------------------------------------------------------

REGRESSOR_FEATURES = [
    'Building Load (RT)',
    'Cooling Water Temperature (C)',
    'Chilled Water Rate (L/sec)',
    'Outside Temperature (F)',
    'Dew Point (F)',
    'Humidity (%)',
    'hour_sin',
    'hour_cos',
    'day_of_week',
]

def train_expected_behavior_models(df: pd.DataFrame) -> dict:
    """
    Trains one RandomForestRegressor per equipment_id to predict expected
    Chiller Energy Consumption given load & environmental conditions.
    Returns dict of {equipment_id: (model, feature_list)}.
    """
    models = {}
    for eq_id, group in df.groupby('equipment_id', sort=False):
        g = group.dropna(subset=REGRESSOR_FEATURES + ['Chiller Energy Consumption (kWh)'])

        X = g[REGRESSOR_FEATURES]
        y = g['Chiller Energy Consumption (kWh)']

        # Time-respecting split: train on first 80% chronologically, validate on last 20%
        split_idx = int(len(g) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        val_score = model.score(X_val, y_val)
        train_score = model.score(X_train, y_train)

        models[eq_id] = {
            'model': model,
            'features': REGRESSOR_FEATURES,
            'val_r2': val_score,
            'train_r2': train_score,
            'n_train': len(X_train),
            'n_val': len(X_val),
        }
        print(f"{eq_id}: train R2={train_score:.3f}, val R2={val_score:.3f} "
              f"(n_train={len(X_train)}, n_val={len(X_val)})")

    return models


# ---------------------------------------------------------------------------
# 5. RESIDUAL-BASED CONTEXTUAL ANOMALY SCORING
# ---------------------------------------------------------------------------

def compute_residuals(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Adds predicted energy and residual columns using per-equipment models."""
    df = df.copy()
    df['predicted_energy_kwh'] = np.nan
    df['residual'] = np.nan
    df['residual_pct'] = np.nan

    for eq_id, mdl_info in models.items():
        mask = df['equipment_id'] == eq_id
        g = df.loc[mask, mdl_info['features']]
        valid = g.dropna().index

        preds = mdl_info['model'].predict(df.loc[valid, mdl_info['features']])
        df.loc[valid, 'predicted_energy_kwh'] = preds
        df.loc[valid, 'residual'] = (
            df.loc[valid, 'Chiller Energy Consumption (kWh)'] - preds
        )
        df.loc[valid, 'residual_pct'] = (
            df.loc[valid, 'residual'] / preds.clip(min=1e-3)
        ) * 100

    return df


def score_anomalies(df: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    """
    Fits an IsolationForest per equipment on residual + context features
    to flag contextual anomalies (points that deviate from what conditions
    predict, not just high/low raw values).
    """
    df = df.copy()
    df['anomaly_score'] = np.nan       # higher = more anomalous (0-1 scale)
    df['is_anomaly'] = False
    df['severity'] = 'Normal'

    iso_features = ['residual', 'residual_pct', 'Building Load (RT)',
                     'load_roll_std_24h', 'efficiency_roll_std_24h']

    for eq_id, group in df.groupby('equipment_id', sort=False):
        g = group.dropna(subset=iso_features)
        if len(g) < 20:
            continue

        iso = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        iso.fit(g[iso_features])

        raw_scores = iso.decision_function(g[iso_features])  # higher = more normal
        preds = iso.predict(g[iso_features])                  # -1 = anomaly

        # Normalize decision_function to 0-1 "anomaly score" (invert + rescale)
        norm_score = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-9)

        df.loc[g.index, 'anomaly_score'] = norm_score
        df.loc[g.index, 'is_anomaly'] = preds == -1

    # Severity buckets from anomaly_score percentile (only among flagged anomalies context)
    def bucket(row):
        if not row['is_anomaly'] or pd.isna(row['anomaly_score']):
            return 'Normal'
        s = row['anomaly_score']
        if s >= 0.85:
            return 'High'
        elif s >= 0.7:
            return 'Medium'
        else:
            return 'Low'

    df['severity'] = df.apply(bucket, axis=1)

    return df


def add_persistence_flag(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Flags anomalies that recur within a rolling window (persistence),
    distinguishing isolated blips from sustained abnormal conditions.
    window=4 -> looks at last 4 observations (~2h at 30-min interval).
    """
    df = df.copy()
    df['persistent_anomaly'] = False

    for eq_id, group in df.groupby('equipment_id', sort=False):
        g = group.sort_values('timestamp')
        rolling_count = g['is_anomaly'].rolling(window, min_periods=1).sum()
        persistent = rolling_count >= 3  # 3+ of last 4 obs flagged
        df.loc[g.index, 'persistent_anomaly'] = persistent.values

    return df


# ---------------------------------------------------------------------------
# 6. ANOMALY TYPE LABELING (behavioral, not diagnostic - see Data Spec Sec.6)
# ---------------------------------------------------------------------------

def add_anomaly_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a human-readable behavioral label to each flagged anomaly,
    derived from severity + persistence + direction of deviation, plus
    a separate multi-unit event flag.

    IMPORTANT: These are behavioral pattern names, not physical fault
    diagnoses. The Data Specification (Sec. 9) confirms no fault/health
    ground truth is provided, so we cannot name specific equipment faults
    (e.g. "compressor failure"). Per Data Spec Sec. 6, where measurements
    don't support identifying a specific physical fault, teams may instead
    label the abnormal behaviour and the condition warranting investigation
    - that is what this function does.
    """
    df = df.copy()
    df['anomaly_type'] = 'Normal'
    df['is_multi_unit_event'] = False

    def _label(row):
        if not row['is_anomaly']:
            return 'Normal'
        direction = 'Over-Consumption' if row['residual'] > 0 else 'Under-Consumption'
        if row['severity'] == 'High' and row['persistent_anomaly']:
            return f'Critical Sustained {direction}'
        elif row['persistent_anomaly']:
            return f'Sustained {direction} Drift'
        elif row['severity'] == 'High' and not row['persistent_anomaly']:
            return f'Isolated {direction} Spike'
        else:
            return f'Minor {direction} Deviation'

    anomaly_mask = df['is_anomaly']
    df.loc[anomaly_mask, 'anomaly_type'] = df.loc[anomaly_mask].apply(_label, axis=1)

    # Multi-unit event: 2+ equipment units anomalous at the same timestamp
    anomalies = df[df['is_anomaly']]
    simul = anomalies.groupby('timestamp')['equipment_id'].nunique()
    multi_unit_times = set(simul[simul >= 2].index)
    df.loc[df['timestamp'].isin(multi_unit_times) & anomaly_mask, 'is_multi_unit_event'] = True

    return df


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(csv_path: str):
    print("1. Loading & validating...")
    df = load_and_validate(csv_path)
    print(f"   Loaded {len(df)} rows, {df['equipment_id'].nunique()} equipment units")

    print("\n2. Handling missing values...")
    df = handle_missing_values(df)
    print(f"   Remaining NaNs:\n{df[MEASUREMENT_COLS].isna().sum().to_dict()}")

    print("\n3. Engineering features...")
    df = engineer_features(df)

    print("\n4. Training expected-behavior models...")
    models = train_expected_behavior_models(df)

    print("\n5. Computing residuals...")
    df = compute_residuals(df, models)

    print("\n6. Scoring contextual anomalies...")
    df = score_anomalies(df)

    print("\n7. Adding persistence flags...")
    df = add_persistence_flag(df)

    print("\n8. Labeling anomaly types...")
    df = add_anomaly_type(df)

    print(f"\nTotal anomalies flagged: {df['is_anomaly'].sum()} "
          f"({100*df['is_anomaly'].mean():.2f}% of rows)")
    print(f"Persistent anomalies: {df['persistent_anomaly'].sum()} "
          f"({100*df['persistent_anomaly'].mean():.2f}% of rows)")
    print("\nSeverity breakdown:")
    print(df['severity'].value_counts())

    return df, models


if __name__ == '__main__':
    result_df, trained_models = run_pipeline('/mnt/user-data/uploads/development_dataset.csv')
    result_df.to_pickle('/home/claude/scored_dataset.pkl')
    print("\nSaved to scored_dataset.pkl")
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. INGESTION & VALIDATION
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    'timestamp': 'datetime',
    'equipment_id': 'categorical',
    'Chilled Water Rate (L/sec)': 'numeric',
    'Cooling Water Temperature (C)': 'numeric',
    'Building Load (RT)': 'numeric',
    'Chiller Energy Consumption (kWh)': 'numeric',
    'Outside Temperature (F)': 'numeric',
    'Dew Point (F)': 'numeric',
    'Humidity (%)': 'numeric',
    'Wind Speed (mph)': 'numeric',
    'Pressure (in)': 'numeric',
}

def load_and_validate(path: str) -> pd.DataFrame:
    """Load CSV and validate it conforms to the Data Specification schema."""
    df = pd.read_csv(path)

    missing_cols = set(REQUIRED_COLUMNS.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['equipment_id'] = df['equipment_id'].astype(str)

    numeric_cols = [c for c, t in REQUIRED_COLUMNS.items() if t == 'numeric']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Sort chronologically WITHIN each equipment unit (per spec 3.1)
    df = df.sort_values(['equipment_id', 'timestamp']).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# 2. MISSING VALUE HANDLING (per-equipment, gap-aware, no leakage across units)
# ---------------------------------------------------------------------------

MEASUREMENT_COLS = [
    'Chilled Water Rate (L/sec)',
    'Cooling Water Temperature (C)',
    'Building Load (RT)',
    'Chiller Energy Consumption (kWh)',
    'Outside Temperature (F)',
    'Dew Point (F)',
    'Humidity (%)',
    'Wind Speed (mph)',
    'Pressure (in)',
]

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate missing values in time order, independently per equipment_id.
    Uses time-based linear interpolation (respects actual gap size), with
    forward/backward fill only at series edges.
    """
    out = []
    for eq_id, group in df.groupby('equipment_id', sort=False):
        g = group.set_index('timestamp').sort_index()
        for col in MEASUREMENT_COLS:
            g[col] = g[col].interpolate(method='time', limit_direction='both')
        g['equipment_id'] = eq_id
        out.append(g.reset_index())
    result = pd.concat(out, ignore_index=True)
    return result.sort_values(['equipment_id', 'timestamp']).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING (causal only - no future leakage)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time features, efficiency proxy, and causal rolling/lag statistics.
    All rolling windows use only past data (min_periods allows partial windows
    at series start rather than dropping rows).
    """
    df = df.copy()

    # --- Time features (cyclic encoding for hour to capture periodicity) ---
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # --- Efficiency proxy: kWh per RT of cooling delivered ---
    # Guard against divide-by-zero / near-zero load (idle periods)
    df['efficiency_kw_per_rt'] = np.where(
        df['Building Load (RT)'] > 1.0,
        df['Chiller Energy Consumption (kWh)'] / df['Building Load (RT)'],
        np.nan
    )

    # --- Per-equipment causal rolling/lag features ---
    feat_frames = []
    for eq_id, group in df.groupby('equipment_id', sort=False):
        g = group.sort_values('timestamp').copy()

        # Time since previous observation (hours) - lets model account for gaps
        g['gap_hours'] = g['timestamp'].diff().dt.total_seconds() / 3600.0
        g['gap_hours'] = g['gap_hours'].fillna(0.5)  # nominal interval at start

        # Rolling stats over ~24h (48 obs at 30-min interval), causal (shift(1)
        # not needed since rolling() with default is already trailing/causal)
        window = 48
        g['load_roll_mean_24h'] = (
            g['Building Load (RT)'].rolling(window, min_periods=6).mean()
        )
        g['load_roll_std_24h'] = (
            g['Building Load (RT)'].rolling(window, min_periods=6).std()
        )
        g['efficiency_roll_mean_24h'] = (
            g['efficiency_kw_per_rt'].rolling(window, min_periods=6).mean()
        )
        g['efficiency_roll_std_24h'] = (
            g['efficiency_kw_per_rt'].rolling(window, min_periods=6).std()
        )

        # Lag feature: previous observation's energy consumption
        g['energy_lag1'] = g['Chiller Energy Consumption (kWh)'].shift(1)

        feat_frames.append(g)

    result = pd.concat(feat_frames, ignore_index=True)

    # Backfill remaining NaNs in rolling features (start-of-series) with
    # per-equipment median so early rows aren't dropped
    roll_cols = ['load_roll_mean_24h', 'load_roll_std_24h',
                 'efficiency_roll_mean_24h', 'efficiency_roll_std_24h',
                 'energy_lag1']
    for col in roll_cols:
        result[col] = result.groupby('equipment_id')[col].transform(
            lambda s: s.fillna(s.median())
        )

    return result


# ---------------------------------------------------------------------------
# 4. EXPECTED-BEHAVIOR MODEL (per-equipment regressor)
# ---------------------------------------------------------------------------

REGRESSOR_FEATURES = [
    'Building Load (RT)',
    'Cooling Water Temperature (C)',
    'Chilled Water Rate (L/sec)',
    'Outside Temperature (F)',
    'Dew Point (F)',
    'Humidity (%)',
    'hour_sin',
    'hour_cos',
    'day_of_week',
]

def train_expected_behavior_models(df: pd.DataFrame) -> dict:
    """
    Trains one RandomForestRegressor per equipment_id to predict expected
    Chiller Energy Consumption given load & environmental conditions.
    Returns dict of {equipment_id: (model, feature_list)}.
    """
    models = {}
    for eq_id, group in df.groupby('equipment_id', sort=False):
        g = group.dropna(subset=REGRESSOR_FEATURES + ['Chiller Energy Consumption (kWh)'])

        X = g[REGRESSOR_FEATURES]
        y = g['Chiller Energy Consumption (kWh)']

        # Time-respecting split: train on first 80% chronologically, validate on last 20%
        split_idx = int(len(g) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        val_score = model.score(X_val, y_val)
        train_score = model.score(X_train, y_train)

        models[eq_id] = {
            'model': model,
            'features': REGRESSOR_FEATURES,
            'val_r2': val_score,
            'train_r2': train_score,
            'n_train': len(X_train),
            'n_val': len(X_val),
        }
        print(f"{eq_id}: train R2={train_score:.3f}, val R2={val_score:.3f} "
              f"(n_train={len(X_train)}, n_val={len(X_val)})")

    return models


# ---------------------------------------------------------------------------
# 5. RESIDUAL-BASED CONTEXTUAL ANOMALY SCORING
# ---------------------------------------------------------------------------

def compute_residuals(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Adds predicted energy and residual columns using per-equipment models."""
    df = df.copy()
    df['predicted_energy_kwh'] = np.nan
    df['residual'] = np.nan
    df['residual_pct'] = np.nan

    for eq_id, mdl_info in models.items():
        mask = df['equipment_id'] == eq_id
        g = df.loc[mask, mdl_info['features']]
        valid = g.dropna().index

        preds = mdl_info['model'].predict(df.loc[valid, mdl_info['features']])
        df.loc[valid, 'predicted_energy_kwh'] = preds
        df.loc[valid, 'residual'] = (
            df.loc[valid, 'Chiller Energy Consumption (kWh)'] - preds
        )
        df.loc[valid, 'residual_pct'] = (
            df.loc[valid, 'residual'] / preds.clip(min=1e-3)
        ) * 100

    return df


def score_anomalies(df: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    """
    Fits an IsolationForest per equipment on residual + context features
    to flag contextual anomalies (points that deviate from what conditions
    predict, not just high/low raw values).
    """
    df = df.copy()
    df['anomaly_score'] = np.nan       # higher = more anomalous (0-1 scale)
    df['is_anomaly'] = False
    df['severity'] = 'Normal'

    iso_features = ['residual', 'residual_pct', 'Building Load (RT)',
                     'load_roll_std_24h', 'efficiency_roll_std_24h']

    for eq_id, group in df.groupby('equipment_id', sort=False):
        g = group.dropna(subset=iso_features)
        if len(g) < 20:
            continue

        iso = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        iso.fit(g[iso_features])

        raw_scores = iso.decision_function(g[iso_features])  # higher = more normal
        preds = iso.predict(g[iso_features])                  # -1 = anomaly

        # Normalize decision_function to 0-1 "anomaly score" (invert + rescale)
        norm_score = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-9)

        df.loc[g.index, 'anomaly_score'] = norm_score
        df.loc[g.index, 'is_anomaly'] = preds == -1

    # Severity buckets from anomaly_score percentile (only among flagged anomalies context)
    def bucket(row):
        if not row['is_anomaly'] or pd.isna(row['anomaly_score']):
            return 'Normal'
        s = row['anomaly_score']
        if s >= 0.85:
            return 'High'
        elif s >= 0.7:
            return 'Medium'
        else:
            return 'Low'

    df['severity'] = df.apply(bucket, axis=1)

    return df


def add_persistence_flag(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Flags anomalies that recur within a rolling window (persistence),
    distinguishing isolated blips from sustained abnormal conditions.
    window=4 -> looks at last 4 observations (~2h at 30-min interval).
    """
    df = df.copy()
    df['persistent_anomaly'] = False

    for eq_id, group in df.groupby('equipment_id', sort=False):
        g = group.sort_values('timestamp')
        rolling_count = g['is_anomaly'].rolling(window, min_periods=1).sum()
        persistent = rolling_count >= 3  # 3+ of last 4 obs flagged
        df.loc[g.index, 'persistent_anomaly'] = persistent.values

    return df


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(csv_path: str):
    print("1. Loading & validating...")
    df = load_and_validate(csv_path)
    print(f"   Loaded {len(df)} rows, {df['equipment_id'].nunique()} equipment units")

    print("\n2. Handling missing values...")
    df = handle_missing_values(df)
    print(f"   Remaining NaNs:\n{df[MEASUREMENT_COLS].isna().sum().to_dict()}")

    print("\n3. Engineering features...")
    df = engineer_features(df)

    print("\n4. Training expected-behavior models...")
    models = train_expected_behavior_models(df)

    print("\n5. Computing residuals...")
    df = compute_residuals(df, models)

    print("\n6. Scoring contextual anomalies...")
    df = score_anomalies(df)

    print("\n7. Adding persistence flags...")
    df = add_persistence_flag(df)

    print(f"\nTotal anomalies flagged: {df['is_anomaly'].sum()} "
          f"({100*df['is_anomaly'].mean():.2f}% of rows)")
    print(f"Persistent anomalies: {df['persistent_anomaly'].sum()} "
          f"({100*df['persistent_anomaly'].mean():.2f}% of rows)")
    print("\nSeverity breakdown:")
    print(df['severity'].value_counts())

    return df, models


if __name__ == '__main__':
    result_df, trained_models = run_pipeline('/mnt/user-data/uploads/development_dataset.csv')
    result_df.to_pickle('/home/claude/scored_dataset.pkl')
    print("\nSaved to scored_dataset.pkl")
