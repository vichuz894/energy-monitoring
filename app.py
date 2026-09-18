"""
YUKTHI 2026 - Intelligent Energy & Equipment Monitoring
Streamlit application: Detect -> Understand -> Assess -> Act

Run with:  streamlit run app.py
Requires:  pip install streamlit pandas numpy scikit-learn plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from pipeline import (
    load_and_validate,
    handle_missing_values,
    engineer_features,
    train_expected_behavior_models,
    compute_residuals,
    score_anomalies,
    add_persistence_flag,
    REGRESSOR_FEATURES,
)

st.set_page_config(
    page_title="Intelligent Energy & Equipment Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CACHED PIPELINE EXECUTION
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_full_pipeline(file_bytes: bytes):
    """
    Runs the full data-to-insight pipeline on an uploaded CSV.
    Cached on file content so re-uploading the same file doesn't retrain.
    """
    import io
    df = load_and_validate(io.BytesIO(file_bytes))
    df = handle_missing_values(df)
    df = engineer_features(df)
    models = train_expected_behavior_models(df)
    df = compute_residuals(df, models)
    df = score_anomalies(df)
    df = add_persistence_flag(df)

    model_diagnostics = {
        eq_id: {'train_r2': m['train_r2'], 'val_r2': m['val_r2'],
                'n_train': m['n_train'], 'n_val': m['n_val']}
        for eq_id, m in models.items()
    }
    return df, model_diagnostics


# ---------------------------------------------------------------------------
# SIDEBAR - DATA INPUT & FILTERS
# ---------------------------------------------------------------------------

st.sidebar.title("⚙️ Energy & Equipment Monitoring")
st.sidebar.caption("Detect → Understand → Assess → Act")

uploaded_file = st.sidebar.file_uploader(
    "Upload operational dataset (CSV, per Data Specification)",
    type=['csv']
)

if uploaded_file is None:
    st.title("Intelligent Energy & Equipment Monitoring")
    st.info(
        "👈 Upload a CSV conforming to the Data Specification to begin.\n\n"
        "Required fields: `timestamp`, `equipment_id`, `Chilled Water Rate (L/sec)`, "
        "`Cooling Water Temperature (C)`, `Building Load (RT)`, "
        "`Chiller Energy Consumption (kWh)`, `Outside Temperature (F)`, "
        "`Dew Point (F)`, `Humidity (%)`, `Wind Speed (mph)`, `Pressure (in)`."
    )
    st.stop()

with st.spinner("Running data pipeline: ingest → clean → engineer features → "
                 "train expected-behavior models → score contextual anomalies..."):
    df, diagnostics = run_full_pipeline(uploaded_file.getvalue())

equipment_list = sorted(df['equipment_id'].unique())
selected_equipment = st.sidebar.multiselect(
    "Equipment", equipment_list, default=equipment_list
)

min_date, max_date = df['timestamp'].min(), df['timestamp'].max()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date(),
)

severity_filter = st.sidebar.multiselect(
    "Severity", ['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low']
)

persistent_only = st.sidebar.checkbox("Persistent anomalies only", value=False)

# Apply filters
mask = df['equipment_id'].isin(selected_equipment)
if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    mask &= (df['timestamp'] >= start) & (df['timestamp'] < end)

filtered = df[mask].copy()
anomalies = filtered[filtered['is_anomaly'] & filtered['severity'].isin(severity_filter)]
if persistent_only:
    anomalies = anomalies[anomalies['persistent_anomaly']]

with st.sidebar.expander("ℹ️ Model diagnostics"):
    for eq_id, d in diagnostics.items():
        st.write(f"**{eq_id}**: val R²={d['val_r2']:.2f}, "
                 f"n_train={d['n_train']}, n_val={d['n_val']}")

# ---------------------------------------------------------------------------
# HEADER METRICS
# ---------------------------------------------------------------------------

st.title("Intelligent Energy & Equipment Monitoring")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Observations (filtered)", f"{len(filtered):,}")
c2.metric("Anomalies flagged", f"{len(anomalies):,}",
          f"{100*len(anomalies)/max(len(filtered),1):.1f}% of rows")
c3.metric("Persistent anomalies",
          f"{int(filtered['persistent_anomaly'].sum()):,}")
c4.metric("High severity", f"{int((filtered['severity']=='High').sum()):,}")

st.caption(
    "Anomalies are contextual: they reflect deviation from **energy consumption "
    "predicted for the observed load and weather conditions**, not simply high "
    "raw values."
)

tab1, tab2, tab3 = st.tabs(
    ["📈 Time Series", "📋 Anomaly Table", "🔍 Drill-Down / Evidence"]
)

# ---------------------------------------------------------------------------
# TAB 1: TIME SERIES - ACTUAL VS EXPECTED, ANOMALIES OVERLAID
# ---------------------------------------------------------------------------

with tab1:
    for eq_id in selected_equipment:
        g = filtered[filtered['equipment_id'] == eq_id].sort_values('timestamp')
        if g.empty:
            continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=g['timestamp'], y=g['Chiller Energy Consumption (kWh)'],
            mode='lines', name='Actual Energy (kWh)',
            line=dict(color='#1f77b4', width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=g['timestamp'], y=g['predicted_energy_kwh'],
            mode='lines', name='Expected Energy (model)',
            line=dict(color='#7f7f7f', width=1, dash='dot')
        ))

        g_anom = g[g['is_anomaly']]
        for sev, color in [('High', '#d62728'), ('Medium', '#ff7f0e'), ('Low', '#f7d038')]:
            sub = g_anom[g_anom['severity'] == sev]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub['timestamp'], y=sub['Chiller Energy Consumption (kWh)'],
                    mode='markers', name=f'{sev} severity anomaly',
                    marker=dict(color=color, size=8,
                                symbol='circle-open' if sev != 'High' else 'circle',
                                line=dict(width=2))
                ))

        fig.update_layout(
            title=f"{eq_id} — Actual vs Expected Energy Consumption",
            xaxis_title="Time", yaxis_title="Energy (kWh)",
            height=380, hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2: ANOMALY TABLE
# ---------------------------------------------------------------------------

with tab2:
    st.subheader(f"Flagged anomalies ({len(anomalies)})")

    if anomalies.empty:
        st.info("No anomalies match the current filters.")
    else:
        display_cols = [
            'timestamp', 'equipment_id', 'severity', 'persistent_anomaly',
            'Building Load (RT)', 'Chiller Energy Consumption (kWh)',
            'predicted_energy_kwh', 'residual', 'residual_pct', 'anomaly_score'
        ]
        table = anomalies[display_cols].sort_values('anomaly_score', ascending=False).copy()
        table = table.rename(columns={
            'predicted_energy_kwh': 'Expected Energy (kWh)',
            'residual': 'Residual (kWh)',
            'residual_pct': 'Residual (%)',
            'anomaly_score': 'Anomaly Score',
            'persistent_anomaly': 'Persistent',
        })
        for col in ['Building Load (RT)', 'Chiller Energy Consumption (kWh)',
                    'Expected Energy (kWh)', 'Residual (kWh)']:
            table[col] = table[col].round(1)
        table['Residual (%)'] = table['Residual (%)'].round(1)
        table['Anomaly Score'] = table['Anomaly Score'].round(3)

        st.dataframe(table, use_container_width=True, height=500)

        csv = table.to_csv(index=False).encode('utf-8')
        st.download_button("Download anomaly table (CSV)", csv,
                            "anomalies.csv", "text/csv")

# ---------------------------------------------------------------------------
# TAB 3: DRILL-DOWN / EVIDENCE VIEW
# ---------------------------------------------------------------------------

with tab3:
    st.subheader("Investigate a specific anomaly")

    if anomalies.empty:
        st.info("No anomalies to investigate under current filters.")
    else:
        options = anomalies.sort_values('anomaly_score', ascending=False)
        options['label'] = (
            options['timestamp'].astype(str) + " — " + options['equipment_id'] +
            " — " + options['severity'] + " (score " +
            options['anomaly_score'].round(2).astype(str) + ")"
        )
        choice = st.selectbox("Select an anomaly", options['label'].tolist())
        row = options[options['label'] == choice].iloc[0]

        eq_id = row['equipment_id']
        ts = row['timestamp']

        colA, colB = st.columns([1, 1])

        with colA:
            st.markdown(f"### {eq_id} at {ts}")
            st.markdown(f"**Severity:** {row['severity']} &nbsp;|&nbsp; "
                        f"**Persistent:** {'Yes' if row['persistent_anomaly'] else 'No'}")
            st.markdown(
                f"- Actual energy: **{row['Chiller Energy Consumption (kWh)']:.1f} kWh**\n"
                f"- Expected energy (given load/weather): **{row['predicted_energy_kwh']:.1f} kWh**\n"
                f"- Deviation: **{row['residual']:+.1f} kWh "
                f"({row['residual_pct']:+.1f}%)**\n"
                f"- Anomaly score: **{row['anomaly_score']:.3f}**"
            )

            # Evidence: compare this observation's conditions to typical
            # conditions for similar load on this equipment
            eq_hist = df[df['equipment_id'] == eq_id]
            similar_load = eq_hist[
                (eq_hist['Building Load (RT)'] >= row['Building Load (RT)'] * 0.9) &
                (eq_hist['Building Load (RT)'] <= row['Building Load (RT)'] * 1.1)
            ]

            st.markdown("**Conditions at this observation vs. typical for similar load:**")
            compare_rows = []
            for feat in ['Cooling Water Temperature (C)', 'Outside Temperature (F)',
                         'Humidity (%)', 'Chilled Water Rate (L/sec)']:
                compare_rows.append({
                    'Variable': feat,
                    'At anomaly': round(row[feat], 1),
                    'Typical (similar load)': round(similar_load[feat].median(), 1),
                })
            st.table(pd.DataFrame(compare_rows).set_index('Variable'))

        with colB:
            window = pd.Timedelta(hours=12)
            context = eq_hist[
                (eq_hist['timestamp'] >= ts - window) &
                (eq_hist['timestamp'] <= ts + window)
            ].sort_values('timestamp')

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=context['timestamp'], y=context['Chiller Energy Consumption (kWh)'],
                mode='lines+markers', name='Actual'
            ))
            fig.add_trace(go.Scatter(
                x=context['timestamp'], y=context['predicted_energy_kwh'],
                mode='lines', name='Expected', line=dict(dash='dot')
            ))
            fig.add_vline(x=ts, line_dash='dash', line_color='red')
            fig.update_layout(
                title="±12h context around this anomaly",
                height=320, xaxis_title="Time", yaxis_title="Energy (kWh)"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown(
            "**Suggested interpretation:** " +
            (
                "This deviation recurs across consecutive observations, indicating a "
                "sustained condition rather than a single sensor blip — recommend "
                "investigating equipment-level causes (e.g. fouling, control fault, "
                "sensor calibration) rather than dismissing as noise."
                if row['persistent_anomaly'] else
                "This appears to be an isolated deviation. Monitor for recurrence "
                "before prioritizing investigation."
            )
        )
