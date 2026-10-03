
import os
import json
import glob
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ================================================================================================
# CONFIGURATION
# ================================================================================================

PROJECT_ROOT = "/content/AgriBioVision_Climate"

P13 = os.path.join(
    PROJECT_ROOT,
    "reports",
    "phase13_crop_health_intelligence"
)

P14 = os.path.join(
    PROJECT_ROOT,
    "reports",
    "phase14_integrated_risk"
)

P15 = os.path.join(
    PROJECT_ROOT,
    "reports",
    "phase15_ai_recommendation_engine"
)

P16 = os.path.join(
    PROJECT_ROOT,
    "reports",
    "phase16_climate_impact_estimator"
)

P17 = os.path.join(
    PROJECT_ROOT,
    "reports",
    "phase17_explainability_scientific_evidence"
)


# ================================================================================================
# PAGE CONFIG
# ================================================================================================

st.set_page_config(
    page_title="AgriBioVision Climate",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================================================================================================
# CUSTOM CSS
# ================================================================================================

st.markdown(
    """
    <style>

    .main {
        background: #f7faf8;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero {
        padding: 28px;
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                #0f5132 0%,
                #198754 50%,
                #20c997 100%
            );
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,.10);
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 18px;
        opacity: .95;
    }

    .metric-card {
        background: white;
        border-radius: 18px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 5px 20px rgba(0,0,0,.05);
        min-height: 125px;
    }

    .metric-title {
        font-size: 14px;
        color: #667085;
        margin-bottom: 7px;
    }

    .metric-value {
        font-size: 31px;
        font-weight: 700;
        color: #153e2e;
    }

    .metric-sub {
        font-size: 13px;
        color: #667085;
        margin-top: 5px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #153e2e;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .info-box {
        padding: 18px;
        border-radius: 16px;
        background: #eef8f1;
        border-left: 5px solid #198754;
        margin: 12px 0;
    }

    .warning-box {
        padding: 18px;
        border-radius: 16px;
        background: #fff8e6;
        border-left: 5px solid #f0ad00;
        margin: 12px 0;
    }

    .danger-box {
        padding: 18px;
        border-radius: 16px;
        background: #fff0f0;
        border-left: 5px solid #dc3545;
        margin: 12px 0;
    }

    .tag {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: #e8f5e9;
        color: #176b3a;
        font-size: 13px;
        margin: 3px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ================================================================================================
# HELPERS
# ================================================================================================

@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def fmt(x, digits=1):
    try:
        return f"{float(x):.{digits}f}"
    except:
        return "—"


def risk_label(score):
    try:
        score = float(score)
    except:
        return "Unknown"

    if score < 25:
        return "Low"
    elif score < 50:
        return "Moderate"
    elif score < 75:
        return "High"
    return "Very High"


def risk_class(score):
    label = risk_label(score)

    if label == "Low":
        return "🟢"
    if label == "Moderate":
        return "🟡"
    if label == "High":
        return "🟠"
    return "🔴"


def metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ================================================================================================
# LOAD DATA
# ================================================================================================

phase14 = load_csv(
    os.path.join(
        P14,
        "processed",
        "phase14_integrated_risk_baseline.csv"
    )
)

phase14_scenarios = load_csv(
    os.path.join(
        P14,
        "processed",
        "phase14_integrated_risk_all_scenarios.csv"
    )
)

phase15 = load_csv(
    os.path.join(
        P15,
        "recommendations",
        "phase15_recommendations_daily.csv"
    )
)

phase16 = load_csv(
    os.path.join(
        P16,
        "processed",
        "phase16_daily_impact_indicators.csv"
    )
)

phase17_evidence = load_csv(
    os.path.join(
        P17,
        "evidence",
        "phase17_scientific_evidence_registry.csv"
    )
)

model_card = load_json(
    os.path.join(
        P17,
        "model_card",
        "phase17_model_card.json"
    )
)


# ================================================================================================
# HERO
# ================================================================================================

st.markdown(
    """
    <div class="hero">
        <h1>🌱 AgriBioVision Climate</h1>
        <p>
        Climate-aware agricultural decision support combining
        climate exposure, water stress, crop-health screening,
        explainable risk and actionable recommendations.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ================================================================================================
# SIDEBAR
# ================================================================================================

st.sidebar.markdown("## 🌱 AgriBioVision")

st.sidebar.caption(
    "Climate + Water + Crop Health Intelligence"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "🌦️ Climate & Water",
        "🌱 Crop Health",
        "🧠 Integrated Risk",
        "💡 Recommendations",
        "📈 Impact Indicators",
        "🔬 Scientific Evidence"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Demo context")

crop = st.sidebar.selectbox(
    "Crop",
    [
        "Tomato",
        "Potato",
        "Corn",
        "Other"
    ]
)

location = st.sidebar.text_input(
    "Location",
    value="Demo location"
)

st.sidebar.caption(
    "Current demonstration uses the project's validated "
    "screening datasets. It does not claim live farm-level validation."
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Scientific status**

    🟢 Benchmark-supported  
    🟡 Screening methodology  
    ⚠️ Field validation pending
    """
)


# ================================================================================================
# DATA PREPARATION
# ================================================================================================

if not phase14.empty:

    phase14["date"] = pd.to_datetime(
        phase14["date"],
        errors="coerce"
    )

    phase14 = phase14.sort_values("date")

if not phase16.empty:

    phase16["date"] = pd.to_datetime(
        phase16["date"],
        errors="coerce"
    )

    phase16 = phase16.sort_values("date")


# ================================================================================================
# DASHBOARD
# ================================================================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">Decision Support Dashboard</div>',
        unsafe_allow_html=True
    )

    if phase14.empty:
        st.error("Integrated-risk data unavailable.")
        st.stop()

    latest = phase14.iloc[-1]

    integrated = float(latest["Integrated Risk"])
    climate = float(latest["Climate Risk"])
    water = float(latest["Water Stress"])

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Integrated Risk",
            f"{integrated:.1f}",
            f"{risk_class(integrated)} {risk_label(integrated)}"
        )

    with c2:
        metric_card(
            "Climate Risk",
            f"{climate:.1f}",
            risk_label(climate)
        )

    with c3:
        metric_card(
            "Water Stress",
            f"{water:.1f}",
            risk_label(water)
        )

    with c4:

        if not phase16.empty and "Resource Pressure Index" in phase16.columns:
            resource = float(
                phase16.iloc[-1]["Resource Pressure Index"]
            )
        else:
            resource = np.nan

        metric_card(
            "Resource Pressure",
            fmt(resource),
            "Normalized indicator"
        )

    st.markdown(
        '<div class="section-title">Current Decision Snapshot</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            <div class="info-box">
            <b>📍 Context</b><br>
            Crop: {crop}<br>
            Location: {location}<br>
            Observation: {latest["date"].date()}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="warning-box">
            <b>⚠️ Interpretation</b><br>
            Current integrated risk is classified as
            <b>{risk_label(integrated)}</b>.
            This is a screening score, not a probability of crop failure.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        if "Dominant Risk" in latest:

            st.markdown(
                f"""
                <div class="info-box">
                <b>Dominant Risk Domain</b><br>
                {latest["Dominant Risk"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        if "Climate Water Compounding" in latest:

            st.markdown(
                f"""
                <div class="info-box">
                <b>Climate-Water Compounding</b><br>
                {latest["Climate Water Compounding"]}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">Risk Trend</div>',
        unsafe_allow_html=True
    )

    chart_df = phase14[
        [
            "date",
            "Integrated Risk",
            "Climate Risk",
            "Water Stress"
        ]
    ].copy()

    chart_df = chart_df.set_index("date")

    st.line_chart(
        chart_df,
        height=330
    )

    st.markdown(
        """
        <div class="info-box">
        <b>How to use this dashboard</b><br>
        Start with the integrated risk, inspect the dominant component,
        review the water/climate indicators, then inspect recommendations
        and scientific evidence before taking action.
        </div>
        """,
        unsafe_allow_html=True
    )


# ================================================================================================
# CLIMATE + WATER
# ================================================================================================

elif page == "🌦️ Climate & Water":

    st.markdown(
        '<div class="section-title">🌦️ Climate & Water Intelligence</div>',
        unsafe_allow_html=True
    )

    if phase16.empty:
        st.error("Climate-impact data unavailable.")
        st.stop()

    if "Climate Risk" in phase16.columns:
        climate_mean = phase16["Climate Risk"].mean()
    else:
        climate_mean = np.nan

    if "Water Stress" in phase16.columns:
        water_mean = phase16["Water Stress"].mean()
    else:
        water_mean = np.nan

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Mean Climate Risk",
            fmt(climate_mean),
            "Project observation period"
        )

    with c2:
        metric_card(
            "Mean Water Stress",
            fmt(water_mean),
            "Project observation period"
        )

    with c3:
        if "Compounding Index" in phase16.columns:
            comp = phase16["Compounding Index"].mean()
        else:
            comp = np.nan

        metric_card(
            "Mean Compounding Index",
            fmt(comp),
            "Normalized indicator"
        )

    st.markdown(
        '<div class="section-title">Climate vs Water Stress</div>',
        unsafe_allow_html=True
    )

    chart_cols = [
        c for c in [
            "Climate Risk",
            "Water Stress"
        ]
        if c in phase16.columns
    ]

    if chart_cols:

        st.line_chart(
            phase16.set_index("date")[chart_cols],
            height=350
        )

    st.markdown(
        '<div class="section-title">Climate-Water Pressure</div>',
        unsafe_allow_html=True
    )

    if "Resource Pressure Index" in phase16.columns:

        rp = phase16[
            [
                "date",
                "Resource Pressure Index"
            ]
        ].set_index("date")

        st.line_chart(
            rp,
            height=300
        )

    st.warning(
        "Resource Pressure is a normalized screening indicator. "
        "It is not a measurement of litres of water saved or actual financial savings."
    )


# ================================================================================================
# CROP HEALTH
# ================================================================================================

elif page == "🌱 Crop Health":

    st.markdown(
        '<div class="section-title">🌱 Crop Health Screening</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Upload a plant image for the Phase 13 benchmark model.

        **Important:** The model was evaluated on PlantVillage benchmark data.
        A prediction in this application should be treated as a screening signal,
        not a definitive field diagnosis.
        """
    )

    uploaded = st.file_uploader(
        "Upload crop image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded:

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                uploaded,
                caption="Uploaded crop image",
                use_container_width=True
            )

        with col2:

            st.markdown("### 🔬 Model Status")

            model_path = os.path.join(
                P13,
                "models",
                "phase13_crop_health_mobilenetv2.keras"
            )

            if os.path.exists(model_path):

                st.success(
                    "Phase 13 MobileNetV2 model available."
                )

                try:

                    from PIL import Image
                    import tensorflow as tf

                    image = Image.open(uploaded).convert("RGB")
                    image = image.resize((160, 160))

                    arr = np.asarray(image).astype("float32")

                    # MobileNetV2 preprocessing.
                    arr = tf.keras.applications.mobilenet_v2.preprocess_input(
                        arr
                    )

                    arr = np.expand_dims(
                        arr,
                        axis=0
                    )

                    @st.cache_resource
                    def load_model():

                        return tf.keras.models.load_model(
                            model_path,
                            compile=False
                        )

                    model = load_model()

                    pred = model.predict(
                        arr,
                        verbose=0
                    )

                    probs = pred[0]

                    idx = int(
                        np.argmax(probs)
                    )

                    confidence = float(
                        probs[idx]
                    )

                    # Try to recover class names.
                    index_files = glob.glob(
                        os.path.join(
                            P13,
                            "processed",
                            "*index*.csv"
                        )
                    )

                    class_name = f"Class {idx}"

                    if index_files:

                        try:

                            idx_df = pd.read_csv(
                                index_files[0]
                            )

                            possible_cols = [
                                c for c in idx_df.columns
                                if c.lower() in [
                                    "class",
                                    "class_name",
                                    "label",
                                    "category"
                                ]
                            ]

                            if possible_cols:

                                labels = (
                                    idx_df[possible_cols[0]]
                                    .dropna()
                                    .astype(str)
                                    .unique()
                                    .tolist()
                                )

                                labels = sorted(labels)

                                if idx < len(labels):
                                    class_name = labels[idx]

                        except Exception:
                            pass

                    st.metric(
                        "Predicted class",
                        class_name
                    )

                    st.metric(
                        "Model confidence",
                        f"{confidence * 100:.1f}%"
                    )

                    if confidence >= 0.80:

                        st.info(
                            "High model confidence does not mean "
                            "field-level diagnostic certainty."
                        )

                    else:

                        st.warning(
                            "Lower model confidence — human/agronomic verification "
                            "is especially important."
                        )

                except Exception as e:

                    st.error(
                        "Model inference could not be completed."
                    )

                    st.caption(
                        str(e)
                    )

            else:

                st.warning(
                    "Phase 13 model file is not currently available."
                )

    st.markdown(
        """
        <div class="warning-box">
        <b>Scientific limitation</b><br>
        Phase 13 was validated as a PlantVillage benchmark.
        Field validation and global validation were not established.
        </div>
        """,
        unsafe_allow_html=True
    )


# ================================================================================================
# INTEGRATED RISK
# ================================================================================================

elif page == "🧠 Integrated Risk":

    st.markdown(
        '<div class="section-title">🧠 Explainable Integrated Risk</div>',
        unsafe_allow_html=True
    )

    if phase14.empty:
        st.error("Integrated-risk data unavailable.")
        st.stop()

    st.markdown(
        """
        Integrated Risk combines three domains:

        **35% Climate Risk + 35% Water Stress + 30% Crop Health Concern**

        The resulting score is a transparent screening framework.
        """
    )

    selected_date = st.date_input(
        "Select observation date",
        value=phase14["date"].max().date()
    )

    selected = phase14[
        phase14["date"].dt.date == selected_date
    ]

    if selected.empty:

        st.warning(
            "No exact observation for this date. "
            "Showing latest available observation."
        )

        selected = phase14.tail(1)

    row = selected.iloc[0]

    integrated = float(row["Integrated Risk"])

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Integrated Risk",
            fmt(integrated),
            f"{risk_class(integrated)} {risk_label(integrated)}"
        )

    with c2:
        metric_card(
            "Climate",
            fmt(row["Climate Risk"]),
            "35% contribution"
        )

    with c3:
        metric_card(
            "Water",
            fmt(row["Water Stress"]),
            "35% contribution"
        )

    with c4:
        metric_card(
            "Crop Health",
            fmt(row["Crop Health Concern"]),
            "30% contribution"
        )

    st.markdown(
        '<div class="section-title">Component Contributions</div>',
        unsafe_allow_html=True
    )

    contributions = pd.DataFrame(
        {
            "Component": [
                "Climate Risk",
                "Water Stress",
                "Crop Health"
            ],
            "Contribution": [
                float(row["Climate Risk Contribution"]),
                float(row["Water Stress Contribution"]),
                float(row["Crop Health Contribution"])
            ]
        }
    )

    st.bar_chart(
        contributions.set_index("Component")
    )

    st.markdown(
        f"""
        <div class="info-box">
        <b>Dominant component:</b>
        {row.get("Dominant Risk", "Not available")}
        <br><br>
        The dominant component indicates which risk contribution
        is largest in the integrated score for this observation.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="warning-box">
        This score is not a probability of disease, crop failure,
        yield loss or financial loss.
        </div>
        """,
        unsafe_allow_html=True
    )


# ================================================================================================
# RECOMMENDATIONS
# ================================================================================================

elif page == "💡 Recommendations":

    st.markdown(
        '<div class="section-title">💡 Explainable Recommendations</div>',
        unsafe_allow_html=True
    )

    if phase15.empty:
        st.warning(
            "Recommendation dataset unavailable."
        )
        st.stop()

    rec_df = phase15.copy()

    if "date" in rec_df.columns:

        rec_df["date"] = pd.to_datetime(
            rec_df["date"],
            errors="coerce"
        )

    latest_date = rec_df["date"].max()

    latest_rec = rec_df[
        rec_df["date"] == latest_date
    ]

    st.info(
        f"Showing recommendation rules for the latest available "
        f"project observation: {latest_date.date()}"
    )

    display_cols = [
        c for c in [
            "recommendation_type",
            "recommendation",
            "rationale",
            "urgency",
            "confidence"
        ]
        if c in latest_rec.columns
    ]

    if not latest_rec.empty:

        for _, r in latest_rec.iterrows():

            urgency = str(
                r.get(
                    "urgency",
                    "Routine"
                )
            )

            if urgency.lower() in [
                "immediate",
                "high"
            ]:

                st.warning(
                    f"**{r.get('recommendation_type', 'Recommendation')}**\n\n"
                    f"{r.get('recommendation', '')}\n\n"
                    f"Rationale: {r.get('rationale', '')}"
                )

            else:

                st.info(
                    f"**{r.get('recommendation_type', 'Recommendation')}**\n\n"
                    f"{r.get('recommendation', '')}\n\n"
                    f"Rationale: {r.get('rationale', '')}"
                )

    st.markdown(
        """
        <div class="warning-box">
        Recommendations are generated by a transparent rule-based
        decision-support engine. They require local agronomic validation
        before real-world intervention.
        </div>
        """,
        unsafe_allow_html=True
    )


# ================================================================================================
# IMPACT INDICATORS
# ================================================================================================

elif page == "📈 Impact Indicators":

    st.markdown(
        '<div class="section-title">📈 Climate Impact Indicators</div>',
        unsafe_allow_html=True
    )

    if phase16.empty:
        st.error("Impact data unavailable.")
        st.stop()

    indicators = [
        "Resource Pressure Index",
        "Climate Exposure Index",
        "Compounding Index",
        "Monitoring Pressure"
    ]

    available = [
        c for c in indicators
        if c in phase16.columns
    ]

    if available:

        latest = phase16.iloc[-1]

        cols = st.columns(
            min(4, len(available))
        )

        for i, col in enumerate(available):

            with cols[i]:

                metric_card(
                    col.replace(
                        " Index",
                        ""
                    ),
                    fmt(latest[col]),
                    "Normalized screening indicator"
                )

        st.markdown(
            '<div class="section-title">Indicator Trends</div>',
            unsafe_allow_html=True
        )

        st.line_chart(
            phase16.set_index("date")[available],
            height=380
        )

        st.markdown(
            """
            <div class="danger-box">
            <b>Important:</b> These are normalized screening indicators.
            They do not represent measured litres of water saved,
            financial savings, yield increase, disease reduction or CO₂ reduction.
            </div>
            """,
            unsafe_allow_html=True
        )

    if "Heat Exposure Flag" in phase16.columns:

        st.markdown(
            '<div class="section-title">Heat Exposure</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            phase16[
                [
                    "date",
                    "Heat Exposure Flag"
                ]
            ].tail(30),
            use_container_width=True,
            hide_index=True
        )


# ================================================================================================
# SCIENTIFIC EVIDENCE
# ================================================================================================

elif page == "🔬 Scientific Evidence":

    st.markdown(
        '<div class="section-title">🔬 Scientific Evidence & Limitations</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        AgriBioVision Climate separates documented evidence,
        project methodology, assumptions and limitations.
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Evidence Sources",
            str(len(phase17_evidence)),
            "Registered sources"
        )

    with c2:

        acc = model_card.get(
            "evaluation",
            {}
        ).get(
            "accuracy",
            0.8831
        )

        metric_card(
            "Benchmark Accuracy",
            f"{float(acc) * 100:.2f}%",
            "PlantVillage test benchmark"
        )

    with c3:

        metric_card(
            "Field Validation",
            "Not established",
            "Current project status"
        )

    st.markdown(
        '<div class="section-title">Evidence Registry</div>',
        unsafe_allow_html=True
    )

    if not phase17_evidence.empty:

        for _, e in phase17_evidence.iterrows():

            st.markdown(
                f"""
                <div class="info-box">
                <b>{e.get("evidence_id", "")} — {e.get("source_name", "")}</b><br>
                <b>Type:</b> {e.get("source_type", "")}<br>
                <b>Supports:</b> {e.get("supports", "")}<br>
                <b>Limitation:</b> {e.get("limitations", "")}
                </div>
                """,
                unsafe_allow_html=True
            )

            url = e.get(
                "source_url",
                ""
            )

            if str(url).startswith("http"):

                st.link_button(
                    "Open source",
                    url
                )

    st.markdown(
        '<div class="section-title">Model Card</div>',
        unsafe_allow_html=True
    )

    if model_card:

        evaluation = model_card.get(
            "evaluation",
            {}
        )

        model_info = pd.DataFrame(
            [
                [
                    "Accuracy",
                    evaluation.get(
                        "accuracy",
                        "—"
                    )
                ],
                [
                    "Balanced Accuracy",
                    evaluation.get(
                        "balanced_accuracy",
                        "—"
                    )
                ],
                [
                    "Macro F1",
                    evaluation.get(
                        "macro_f1",
                        "—"
                    )
                ],
                [
                    "Weighted F1",
                    evaluation.get(
                        "weighted_f1",
                        "—"
                    )
                ],
                [
                    "Field validation",
                    model_card.get(
                        "validation_scope",
                        {}
                    ).get(
                        "field_validation",
                        False
                    )
                ],
                [
                    "Global validation",
                    model_card.get(
                        "validation_scope",
                        {}
                    ).get(
                        "global_validation",
                        False
                    )
                ]
            ],
            columns=[
                "Metric",
                "Value"
            ]
        )

        st.dataframe(
            model_info,
            use_container_width=True,
            hide_index=True
        )

    st.markdown(
        """
        <div class="danger-box">
        <b>Scientific limitation:</b>
        PlantVillage benchmark performance should not be interpreted
        as real-world field accuracy. Independent field validation is
        required before operational deployment.
        </div>
        """,
        unsafe_allow_html=True
    )


# ================================================================================================
# DOWNLOAD SECTION
# ================================================================================================

st.divider()

st.markdown(
    '<div class="section-title">📥 Project Data</div>',
    unsafe_allow_html=True
)

download_cols = st.columns(4)

if not phase14.empty:

    with download_cols[0]:

        st.download_button(
            "Download Risk Data",
            data=phase14.to_csv(index=False),
            file_name="agribiovision_integrated_risk.csv",
            mime="text/csv"
        )

if not phase15.empty:

    with download_cols[1]:

        st.download_button(
            "Download Recommendations",
            data=phase15.to_csv(index=False),
            file_name="agribiovision_recommendations.csv",
            mime="text/csv"
        )

if not phase16.empty:

    with download_cols[2]:

        st.download_button(
            "Download Impact Data",
            data=phase16.to_csv(index=False),
            file_name="agribiovision_impact_indicators.csv",
            mime="text/csv"
        )

if not phase17_evidence.empty:

    with download_cols[3]:

        st.download_button(
            "Download Evidence",
            data=phase17_evidence.to_csv(index=False),
            file_name="agribiovision_scientific_evidence.csv",
            mime="text/csv"
        )


# ================================================================================================
# FOOTER
# ================================================================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#667085;padding:15px;">

    <b>🌱 AgriBioVision Climate</b><br>

    Climate-aware agricultural decision support<br><br>

    Screening tool • Scientific evidence • Explainable methodology<br>

    <small>
    Current project status: benchmark-supported and screening-oriented.
    Field and global validation remain future work.
    </small>

    </div>
    """,
    unsafe_allow_html=True
)
