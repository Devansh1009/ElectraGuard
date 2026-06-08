"""
ElectraGuard ⚡ — Electricity Theft Detection System
Streamlit Application

Based on: Kocaman, B. & Tümen, V. (2020).
"Detection of electricity theft using data processing and LSTM method"
Sadhana, 45, 286.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from theft_detection import TheftDetectionEngine
from sgcc_generator import generate_dataset, to_xlsx_bytes

# ─── Page Config ───
st.set_page_config(
    page_title="ElectraGuard — Electricity Theft Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    /* Global */
    .stApp { background: #0a0e1a; }
    section[data-testid="stSidebar"] { background: #0f1423; border-right: 1px solid rgba(0,212,255,0.1); }
    h1, h2, h3 { color: #e8ecf4 !important; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(15,20,40,0.7);
        border: 1px solid rgba(0,212,255,0.1);
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(0,212,255,0.3);
    }
    [data-testid="stMetricLabel"] { color: #8892a8 !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #e8ecf4 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(15,20,40,0.7);
        border: 1px solid rgba(0,212,255,0.1);
        border-radius: 8px;
        color: #8892a8;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0,212,255,0.08);
        border-color: rgba(0,212,255,0.3);
        color: #00d4ff !important;
    }

    /* DataFrame */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* Buttons */
    .stButton > button {
        border: 1px solid rgba(0,212,255,0.2);
        background: rgba(0,212,255,0.05);
        color: #00d4ff;
        border-radius: 10px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: rgba(0,212,255,0.12);
        border-color: rgba(0,212,255,0.4);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 20px 0 10px;
    }
    .main-header h1 {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    .gradient-text {
        background: linear-gradient(135deg, #00d4ff 0%, #3b82f6 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-critical { background: rgba(239,68,68,0.12); color: #ff6b6b; border: 1px solid rgba(239,68,68,0.2); }
    .badge-high { background: rgba(249,115,22,0.12); color: #fb923c; border: 1px solid rgba(249,115,22,0.2); }
    .badge-medium { background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(245,158,11,0.2); }
    .badge-low { background: rgba(16,185,129,0.12); color: #5de0b5; border: 1px solid rgba(16,185,129,0.2); }

    /* Info boxes */
    .info-card {
        background: rgba(15,20,40,0.7);
        border: 1px solid rgba(0,212,255,0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ───
if "engine" not in st.session_state:
    st.session_state.engine = None
if "df" not in st.session_state:
    st.session_state.df = None
if "source_name" not in st.session_state:
    st.session_state.source_name = None
if "lstm_results" not in st.session_state:
    st.session_state.lstm_results = None


def run_detection(df, source_name):
    """Run the full detection pipeline."""
    engine = TheftDetectionEngine()
    engine.process_data(df)
    engine.calculate_stats()
    engine.calculate_risk_scores()
    st.session_state.engine = engine
    st.session_state.df = engine.df
    st.session_state.source_name = source_name
    st.session_state.lstm_results = None


def run_lstm_training():
    """Run LSTM training on the loaded data."""
    engine = st.session_state.engine
    if engine is None or len(engine.ts_cols) < 7:
        st.warning("Need ≥ 7 time-series columns for LSTM training.")
        return

    ts_matrix = engine.get_time_series_matrix()
    labels = engine.auto_label_for_lstm()

    if ts_matrix is None:
        st.warning("No time-series data found.")
        return

    progress_bar = st.progress(0, text="Initializing LSTM...")
    status_text = st.empty()

    def on_progress(fold, total, metrics):
        pct = fold / total
        progress_bar.progress(pct, text=f"Fold {fold}/{total} — Acc: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
        status_text.info(f"✅ Fold {fold} complete: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

    try:
        from lstm_model import train_and_evaluate
        results = train_and_evaluate(ts_matrix, labels, progress_callback=on_progress)

        if results:
            st.session_state.lstm_results = results
            engine.add_lstm_scores(results["consumer_probs"])
            st.session_state.df = engine.df
            progress_bar.progress(1.0, text="Model training complete!")
            st.success(f"🎉 Model trained! Avg Accuracy: {results['avg_metrics']['accuracy']:.4f}, F1: {results['avg_metrics']['f1']:.4f}")
        else:
            st.warning("Training returned no results — ensure data has both normal and theft consumers.")
    except Exception as e:
        st.error(f"Model training failed: {e}")


# ─── Sidebar ───
with st.sidebar:
    st.markdown("## ⚡ ElectraGuard")
    st.markdown("*AI-Powered Theft Detection*")
    st.divider()

    st.markdown("### 📂 Load Data")

    # File upload
    uploaded = st.file_uploader(
        "Upload Excel/CSV", type=["xlsx", "xls", "csv"],
        help="Upload electricity consumption data"
    )

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            run_detection(df, uploaded.name)
            st.success(f"✅ Loaded {len(df)} records from {uploaded.name}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎲 Sample Data", use_container_width=True):
            df, stats = generate_dataset(150, 30, 0.15)
            df = df.drop(columns=["_attack_type"], errors="ignore")
            run_detection(df, "Sample Data (150)")
            st.rerun()
    with col2:
        if st.button("🧬 SGCC (500)", use_container_width=True):
            df, stats = generate_dataset(500, 60, 0.15)
            df = df.drop(columns=["_attack_type"], errors="ignore")
            run_detection(df, f"SGCC Dataset ({stats['total_consumers']})")
            st.rerun()

    if st.button("📥 Download SGCC .xlsx", use_container_width=True):
        df, stats = generate_dataset(500, 60, 0.15)
        xlsx_bytes = to_xlsx_bytes(df)
        st.download_button(
            "⬇️ Save File",
            data=xlsx_bytes,
            file_name="SGCC_ElectricityTheft_500consumers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.divider()

    # LSTM Training
    if st.session_state.engine is not None:
        engine = st.session_state.engine
        has_ts = len(engine.ts_cols) >= 7
        st.markdown("### 🧠 LSTM Training")
        if has_ts:
            st.info(f"📊 {len(engine.ts_cols)} time-series columns detected")
            if st.button("🚀 Train LSTM Model", use_container_width=True, type="primary"):
                run_lstm_training()
                st.rerun()
        else:
            st.warning("Need ≥ 7 time-series columns (day_1, day_2, ...)")

        if st.session_state.lstm_results:
            avg = st.session_state.lstm_results["avg_metrics"]
            st.success(f"Model trained! F1: {avg['f1']:.3f}")

    st.divider()
    st.caption("Based on Kocaman & Tümen (2020)")
    st.caption("[View Paper](https://www.ias.ac.in/article/fulltext/sadh/045/0286)")


# ─── Main Content ───
if st.session_state.engine is None:
    # Landing page
    st.markdown("""
    <div class="main-header">
        <p><span class="badge" style="background:rgba(0,212,255,0.08); color:#00d4ff; border:1px solid rgba(0,212,255,0.15);">
            ⚡ AI-Powered Detection System
        </span></p>
        <h1>Electricity Theft<br><span class="gradient-text">Detection System</span></h1>
        <p style="color:#8892a8; max-width:600px; margin:16px auto;">
            Upload your electricity consumption data (XLSX/CSV) to detect potential theft
            using hybrid LSTM deep learning and statistical anomaly detection.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="info-card">
            <h4>🧠 LSTM Engine</h4>
            <p style="color:#8892a8; font-size:0.85rem;">Deep learning with 5-fold CV and per-consumer theft probability</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="info-card">
            <h4>📊 6 Detectors</h4>
            <p style="color:#8892a8; font-size:0.85rem;">Z-Score, IQR, billing ratio, load, consumption drop, meter status</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="info-card">
            <h4>📈 Live Analytics</h4>
            <p style="color:#8892a8; font-size:0.85rem;">Interactive charts, risk distribution, regional breakdown</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="info-card">
            <h4>📥 SGCC Dataset</h4>
            <p style="color:#8892a8; font-size:0.85rem;">Built-in benchmark with 6 theft attack types from the research paper</p>
        </div>""", unsafe_allow_html=True)

    st.info("👈 **Get started** — upload a file or click **Sample Data / SGCC** in the sidebar.")

else:
    # ─── Dashboard ───
    engine = st.session_state.engine
    df = st.session_state.df
    summary = engine.get_summary()

    st.markdown(f"### ⚡ {st.session_state.source_name}")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "📋 Data View", "⚠️ Alerts", "📈 Analytics", "📖 Guide"
    ])

    # ── Tab 1: Dashboard ──
    with tab1:
        # Stats row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Consumers", summary["total"])
        c2.metric("🔴 Critical", summary["critical"])
        c3.metric("🟠 High Risk", summary["high"])
        c4.metric("🟡 Medium Risk", summary["medium"])
        c5.metric("🟢 Low Risk", summary["low"])

        st.divider()

        # Charts
        col_left, col_right = st.columns(2)

        with col_left:
            # Risk distribution
            risk_data = pd.DataFrame({
                "Level": ["Critical", "High", "Medium", "Low"],
                "Count": [summary["critical"], summary["high"], summary["medium"], summary["low"]],
            })
            fig = px.bar(
                risk_data, x="Level", y="Count",
                color="Level",
                color_discrete_map={
                    "Critical": "#ef4444", "High": "#f97316",
                    "Medium": "#f59e0b", "Low": "#10b981"
                },
                title="Risk Level Distribution",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(15,20,40,0.7)",
                font_color="#8892a8",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            # Score distribution histogram
            fig2 = px.histogram(
                df, x="_risk_score", nbins=20,
                title="Risk Score Distribution",
                color_discrete_sequence=["#00d4ff"],
            )
            fig2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(15,20,40,0.7)",
                font_color="#8892a8",
                xaxis_title="Risk Score",
                yaxis_title="Count",
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Regional breakdown
        if "by_region" in summary and summary["by_region"]:
            region_df = pd.DataFrame(
                list(summary["by_region"].items()),
                columns=["Region", "Avg Risk Score"]
            )
            fig3 = px.bar(
                region_df, x="Region", y="Avg Risk Score",
                title="Average Risk Score by Region",
                color="Avg Risk Score",
                color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
            )
            fig3.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(15,20,40,0.7)",
                font_color="#8892a8",
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 2: Data View ──
    with tab2:
        st.markdown("#### 📋 Consumer Data Table")

        # Filters
        col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
        with col_f1:
            search = st.text_input("🔍 Search by ID or Name", "")
        with col_f2:
            risk_filter = st.selectbox("Risk Level", ["All", "Critical", "High", "Medium", "Low"])
        with col_f3:
            sort_by = st.selectbox("Sort by", ["Risk Score (High→Low)", "Risk Score (Low→High)", "Consumer ID"])

        display_df = df.copy()

        # Apply filters
        if search:
            id_col = engine.col_map.get("consumer_id", "")
            name_col = engine.col_map.get("name", "")
            mask = pd.Series([False] * len(display_df))
            if id_col:
                mask |= display_df[id_col].astype(str).str.contains(search, case=False, na=False)
            if name_col:
                mask |= display_df[name_col].astype(str).str.contains(search, case=False, na=False)
            display_df = display_df[mask]

        if risk_filter != "All":
            display_df = display_df[display_df["_risk_level"] == risk_filter]

        if "High→Low" in sort_by:
            display_df = display_df.sort_values("_risk_score", ascending=False)
        elif "Low→High" in sort_by:
            display_df = display_df.sort_values("_risk_score", ascending=True)

        # Select display columns
        display_cols = []
        for field in ["consumer_id", "name", "consumption", "billing", "region", "category"]:
            if field in engine.col_map:
                display_cols.append(engine.col_map[field])
        display_cols.extend(["_risk_score", "_risk_level", "_flags"])
        if "_lstm_prob" in display_df.columns:
            display_cols.append("_lstm_prob")

        existing_cols = [c for c in display_cols if c in display_df.columns]
        st.dataframe(
            display_df[existing_cols].reset_index(drop=True),
            use_container_width=True,
            height=500,
        )
        st.caption(f"Showing {len(display_df)} of {len(df)} records")

        # Export
        csv = display_df[existing_cols].to_csv(index=False)
        st.download_button("📥 Export as CSV", csv, "electraguard_results.csv", "text/csv")

    # ── Tab 3: Alerts ──
    with tab3:
        st.markdown("#### ⚠️ Theft Alerts")

        alerts_df = df[df["_risk_level"].isin(["Critical", "High"])].sort_values("_risk_score", ascending=False)

        if len(alerts_df) == 0:
            st.success("✅ No critical or high-risk consumers detected.")
        else:
            st.error(f"🚨 **{len(alerts_df)} suspicious consumers** detected")

            for _, row in alerts_df.head(20).iterrows():
                id_col = engine.col_map.get("consumer_id", "")
                name_col = engine.col_map.get("name", "")
                cid = row[id_col] if id_col else "?"
                cname = row[name_col] if name_col else ""
                score = row["_risk_score"]
                level = row["_risk_level"]
                flags = row["_flags"]

                color = "#ef4444" if level == "Critical" else "#f97316"
                emoji = "🔴" if level == "Critical" else "🟠"

                with st.expander(f"{emoji} {cid} — {cname} | Score: {score:.0f}/100 ({level})"):
                    cols = st.columns(3)
                    cons_col = engine.col_map.get("consumption", "")
                    bill_col = engine.col_map.get("billing", "")
                    if cons_col:
                        cols[0].metric("Consumption", f"{row[cons_col]:,.0f} kWh")
                    if bill_col:
                        cols[1].metric("Billing", f"₹{row[bill_col]:,.0f}")
                    cols[2].metric("Risk Score", f"{score:.0f}/100")

                    st.markdown(f"**Flags:** {flags}")

                    if "_lstm_prob" in row and not pd.isna(row["_lstm_prob"]):
                        st.progress(float(row["_lstm_prob"]), text=f"LSTM Theft Probability: {row['_lstm_prob']*100:.1f}%")

    # ── Tab 4: Analytics ──
    with tab4:
        st.markdown("#### 📈 Detection Analytics")

        # Category breakdown
        if "by_category" in summary and summary["by_category"]:
            cat_df = pd.DataFrame(
                list(summary["by_category"].items()),
                columns=["Category", "Avg Risk Score"]
            )
            fig_cat = px.pie(
                cat_df, names="Category", values="Avg Risk Score",
                title="Risk Distribution by Category",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            fig_cat.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(15,20,40,0.7)",
                font_color="#8892a8",
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        # Top suspicious
        st.markdown("##### 🔝 Top 10 Most Suspicious")
        top10 = df.nlargest(10, "_risk_score")
        cols_to_show = []
        for f in ["consumer_id", "name", "consumption", "region"]:
            if f in engine.col_map:
                cols_to_show.append(engine.col_map[f])
        cols_to_show.extend(["_risk_score", "_risk_level", "_flags"])
        existing = [c for c in cols_to_show if c in top10.columns]
        st.dataframe(top10[existing].reset_index(drop=True), use_container_width=True)

        # LSTM Results
        if st.session_state.lstm_results:
            st.divider()
            st.markdown("##### 🧠 LSTM Model Performance")
            results = st.session_state.lstm_results
            avg = results["avg_metrics"]

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy", f"{avg['accuracy']:.4f}")
            mc2.metric("Precision", f"{avg['precision']:.4f}")
            mc3.metric("Recall", f"{avg['recall']:.4f}")
            mc4.metric("F1 Score", f"{avg['f1']:.4f}")

            st.markdown("**Per-Fold Metrics:**")
            fold_df = pd.DataFrame(results["fold_metrics"])
            fold_df.index = [f"Fold {i+1}" for i in range(len(fold_df))]
            st.dataframe(fold_df.style.format("{:.4f}"), use_container_width=True)

            # LSTM probability distribution
            if "_lstm_prob" in df.columns:
                fig_lstm = px.histogram(
                    df, x="_lstm_prob", nbins=30,
                    title="LSTM Theft Probability Distribution",
                    color_discrete_sequence=["#a855f7"],
                )
                fig_lstm.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(15,20,40,0.7)",
                    font_color="#8892a8",
                    xaxis_title="Theft Probability",
                    yaxis_title="Count",
                )
                st.plotly_chart(fig_lstm, use_container_width=True)

    # ── Tab 5: Guide ──
    with tab5:
        st.markdown("#### 📖 Data Labelling Guide")
        st.markdown("""
        Learn how to structure and label your Excel data for ElectraGuard.

        ---

        ##### 1️⃣ Overview
        ElectraGuard accepts `.xlsx`, `.xls`, and `.csv` files. Column names are **auto-detected**.

        > **Minimum:** At least one numeric column for electricity consumption (kWh).

        ---

        ##### 2️⃣ Column Reference

        | Field | Priority | Accepted Names | Type |
        |-------|----------|----------------|------|
        | **Consumer ID** | Recommended | `consumer_id`, `id`, `cust_id`, `meter_id` | Text/Number |
        | **Consumption** | ⚠️ Required | `consumption`, `kwh`, `units`, `energy`, `usage` | Number (kWh) |
        | **Billing** | Recommended | `billing`, `bill`, `amount`, `charges` | Number (₹) |
        | **Region** | Optional | `region`, `area`, `zone`, `district`, `feeder` | Text |
        | **Category** | Optional | `category`, `type`, `consumer_type`, `tariff` | Text |
        | **Sanctioned Load** | Optional | `sanctioned_load`, `contract_demand` | Number (kW) |
        | **Actual Load** | Optional | `actual_load`, `measured_load`, `peak_load` | Number (kW) |
        | **Meter Status** | Optional | `meter_status`, `status`, `meter_condition` | Text |
        | **Theft Flag** | 🧠 LSTM | `flag`, `label`, `theft`, `is_theft`, `fraud`, `target` | 0 or 1 |

        ---

        ##### 3️⃣ Time-Series Columns (for LSTM)
        Include **≥ 7 sequential columns** like `day_1`, `day_2`, ... `day_30` for LSTM activation.

        Accepted patterns: `day_N`, `d_N`, `reading_N`, `week_N`, `month_N`, dates as headers.

        ---

        ##### 4️⃣ Theft Flag (Label)
        | Value | Meaning |
        |-------|---------|
        | `0` | ✅ Normal / Honest consumer |
        | `1` | 🚨 Confirmed or suspected theft |

        > **No labels?** The app auto-generates pseudo-labels using statistical heuristics.

        ---

        ##### 5️⃣ Processing Pipeline (Kocaman & Tümen, 2020)
        1. **Data Cleaning** — NaN → mean, negatives → 0
        2. **Min-Max Normalization** — Scale to [0, 1]
        3. **Sliding Window** — 7-step windows for LSTM
        4. **LSTM Training** — `LSTM(128) → Dropout(0.2) → Dense(64) → Dense(1)`
        5. **5-Fold Cross-Validation** — Accuracy, Precision, Recall, F1
        6. **Hybrid Scoring** — LSTM (40 pts) + Statistical (60 pts) = 0–100

        ---

        ##### 6️⃣ Best Practices
        ✅ Use consistent column names, no merged cells
        ✅ Provide ≥ 7 daily reading columns for LSTM
        ✅ Label at least 10% as theft (Flag=1) for balanced training
        ✅ Keep data on the first sheet

        ❌ Don't use merged cells or blank header rows
        ❌ Don't mix units (kWh and MWh)
        ❌ Don't use Flag values other than 0 or 1

        ---

        *Based on: Kocaman, B. & Tümen, V. (2020). Sadhana, 45, 286.*
        *[View Paper ↗](https://www.ias.ac.in/article/fulltext/sadh/045/0286)*
        """)
