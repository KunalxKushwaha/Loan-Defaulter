"""
frontend/app.py  –  Streamlit UI for the Loan Defaulter Prediction system.

Run from the project root:
    streamlit run frontend/app.py
"""

import sys
import os

# Make sure the project root is on the path so backend imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from backend.predictor import predict, model_exists, get_valid_options

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Animations ─────────────────────────────────────────────── */
@keyframes fadeSlideDown {
  from { opacity: 0; transform: translateY(-18px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
  0%   { background-position: 0% center; }
  100% { background-position: 200% center; }
}
@keyframes pulseRing {
  0%   { opacity: 0.8; transform: scale(1); }
  70%  { opacity: 0;   transform: scale(1.06); }
  100% { opacity: 0;   transform: scale(1.06); }
}
@keyframes popIn {
  from { opacity: 0; transform: scale(0.75); }
  to   { opacity: 1; transform: scale(1); }
}
@keyframes slideInLeft {
  from { opacity: 0; transform: translateX(-12px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes dotPulse {
  0%, 100% { box-shadow: 0 0 0 4px rgba(239,68,68,0.2); }
  50%       { box-shadow: 0 0 0 8px rgba(239,68,68,0.05); }
}

/* ── Page backgrounds ───────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid #1e3a5f;
}

/* ── Title ──────────────────────────────────────────────────── */
.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4, #3b82f6);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 4px;
    animation: fadeSlideDown 0.7s ease both, shimmer 3s linear infinite;
}
.sub-title {
    text-align: center;
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 36px;
    animation: fadeSlideDown 0.8s ease 0.1s both;
    opacity: 0;
    animation-fill-mode: both;
}

/* ── Section header ─────────────────────────────────────────── */
.section-header {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 0.5px solid #1e3a5f;
    padding-bottom: 6px;
    margin: 20px 0 12px 0;
}

/* ── Metric cards ───────────────────────────────────────────── */
.metric-card {
    background: rgba(15, 23, 42, 0.6);
    border: 0.5px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.55s ease both;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: #3b82f6;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.35s ease;
}
.metric-card:hover::before { transform: scaleX(1); }

/* ── Result cards ───────────────────────────────────────────── */
.result-card-default {
    background: rgba(239, 68, 68, 0.08);
    border: 1.5px solid #ef4444;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    position: relative;
    animation: fadeSlideUp 0.5s ease both;
    transition: transform 0.2s;
}
.result-card-default::after {
    content: '';
    position: absolute;
    inset: -2px;
    border-radius: 18px;
    border: 2px solid #ef4444;
    animation: pulseRing 2s ease-out infinite;
    pointer-events: none;
}
.result-card-default:hover { transform: scale(1.02); }

.result-card-safe {
    background: rgba(34, 197, 94, 0.08);
    border: 1.5px solid #22c55e;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    animation: fadeSlideUp 0.5s ease both;
    transition: transform 0.2s;
}
.result-card-safe:hover { transform: scale(1.02); }

/* ── Risk gauge bar ─────────────────────────────────────────── */
.gauge-track {
    height: 10px;
    border-radius: 99px;
    background: rgba(255,255,255,0.06);
    border: 0.5px solid #1e3a5f;
    overflow: visible;
    position: relative;
    margin-bottom: 6px;
}
.gauge-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #22c55e 0%, #f59e0b 50%, #ef4444 100%);
    transition: width 1.2s cubic-bezier(0.22, 1, 0.36, 1);
    position: relative;
}
.gauge-fill::after {
    content: '';
    position: absolute; right: 0; top: 50%;
    transform: translate(50%, -50%);
    width: 14px; height: 14px;
    border-radius: 50%;
    background: white;
    border: 2px solid #ef4444;
    animation: dotPulse 1.5s ease-in-out infinite;
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #475569;
}

/* ── Warning box ────────────────────────────────────────────── */
.warning-box {
    background: rgba(245, 158, 11, 0.08);
    border: 0.5px solid #b45309;
    border-left: 3px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 14px 16px;
    color: #fbbf24;
    font-size: 0.9rem;
    margin-bottom: 16px;
    animation: slideInLeft 0.4s ease both;
}

/* ── Sidebar labels & inputs ────────────────────────────────── */
[data-testid="stSidebar"] label { color: #94a3b8 !important; }
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border-color: #1e3a5f !important;
    transition: border-color 0.2s ease;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within {
    border-color: #3b82f6 !important;
}

/* ── Predict button ─────────────────────────────────────────── */
div[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #3b82f6);
    background-size: 200% auto;
    color: white;
    font-weight: 700;
    font-size: 1.05rem;
    border: none;
    border-radius: 10px;
    padding: 14px 0;
    width: 100%;
    letter-spacing: 0.03em;
    transition: background-position 0.5s ease, transform 0.15s ease, box-shadow 0.15s ease;
    position: relative;
    overflow: hidden;
}
div[data-testid="stButton"] > button:hover {
    background-position: right center;
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(139,92,246,0.4);
}
div[data-testid="stButton"] > button:active {
    transform: scale(0.97);
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar – Applicant Form
# ─────────────────────────────────────────────────────────────────────────────
valid_opts = get_valid_options()

with st.sidebar:
    st.markdown("## 🏦 Loan Applicant Details")
    st.markdown("---")

    st.markdown('<p class="section-header">👤 Personal Information</p>', unsafe_allow_html=True)
    person_age    = st.number_input("Age",           min_value=18,  max_value=100, value=30, step=1)
    person_gender = st.selectbox("Gender",           valid_opts["person_gender"])
    person_education = st.selectbox("Education Level", valid_opts["person_education"])
    person_home_ownership = st.selectbox("Home Ownership", valid_opts["person_home_ownership"])
    person_income = st.number_input("Annual Income ($)", min_value=0, max_value=10_000_000, value=60_000, step=500)
    person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=50, value=5, step=1)

    st.markdown('<p class="section-header">💳 Credit Profile</p>', unsafe_allow_html=True)
    credit_score               = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5, step=1)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Default on File?", valid_opts["previous_loan_defaults_on_file"])

    st.markdown('<p class="section-header">📋 Loan Details</p>', unsafe_allow_html=True)
    loan_intent       = st.selectbox("Loan Purpose", valid_opts["loan_intent"])
    loan_amnt         = st.number_input("Loan Amount ($)", min_value=500, max_value=500_000, value=10_000, step=100)
    loan_int_rate     = st.number_input("Interest Rate (%)", min_value=1.0, max_value=35.0, value=12.0, step=0.1, format="%.1f")
    loan_percent_income = round(loan_amnt / person_income * 100, 2) if person_income > 0 else 0.0
    st.info(f"📊 Loan-to-Income Ratio: **{loan_percent_income:.1f}%**")

    st.markdown("---")

    threshold = st.slider(
        "Decision Threshold",
        min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Lower threshold → more conservative (catches more defaults); higher → more lenient."
    )

    predict_btn = st.button("🔍 Predict Default Risk", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Main Area
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🏦 Loan Default Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">ML-powered risk assessment using XGBoost | Fill in the applicant details on the left and click Predict</p>', unsafe_allow_html=True)

# ── Model-not-found banner ────────────────────────────────────────────────────
if not model_exists():
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Model not found.</strong>  
        Place your trained <code>model.pkl</code> in the <code>models/</code> folder, 
        or run <code>python backend/train.py</code> to train it automatically.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn:
    form_data = {
        "person_age":                      person_age,
        "person_gender":                   person_gender,
        "person_education":                person_education,
        "person_home_ownership":           person_home_ownership,
        "person_income":                   person_income,
        "person_emp_exp":                  person_emp_exp,
        "credit_score":                    credit_score,
        "cb_person_cred_hist_length":      cb_person_cred_hist_length,
        "previous_loan_defaults_on_file":  previous_loan_defaults_on_file,
        "loan_intent":                     loan_intent,
        "loan_amnt":                       loan_amnt,
        "loan_int_rate":                   loan_int_rate,
        "loan_percent_income":             loan_percent_income,
    }

    try:
        result = predict(form_data, threshold=threshold)

        # ── Top result card ───────────────────────────────────────────────────
        is_default = result["prediction"] == 1
        card_cls   = "result-card-default" if is_default else "result-card-safe"
        icon       = "🚨" if is_default else "✅"
        color      = "#ef4444" if is_default else "#22c55e"

        risk_cls_map = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}
        risk_cls = risk_cls_map[result["risk_level"]]

        st.markdown(f"""
        <div class="{card_cls}">
            <div style="font-size:3rem">{icon}</div>
            <h2 style="color:{color}; margin:8px 0 4px 0; font-size:2rem">{result['label']}</h2>
            <p style="color:#94a3b8; margin:0">Prediction confidence: <strong style="color:#e2e8f0">{result['confidence']*100:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── KPI row ───────────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Default Probability", f"{result['probability']*100:.1f}%")
        with col2:
            st.metric("Risk Level", result["risk_level"])
        with col3:
            st.metric("Decision Threshold", f"{threshold:.0%}")
        with col4:
            st.metric("Loan-to-Income", f"{loan_percent_income:.1f}%")

        st.markdown("---")

        # ── Charts row ────────────────────────────────────────────────────────
        c_gauge, c_breakdown = st.columns([1, 1])

        # Gauge chart
        with c_gauge:
            st.subheader("📊 Risk Gauge")
            prob = result["probability"]

            fig, ax = plt.subplots(figsize=(2, 2), subplot_kw={"projection": "polar"})
            fig.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")

            # background arcs
            for start, end, clr in [(0, np.pi/3, "#22c55e"), (np.pi/3, 2*np.pi/3, "#f59e0b"), (2*np.pi/3, np.pi, "#ef4444")]:
                theta = np.linspace(start, end, 100)
                ax.fill_between(theta, 0.7, 1.0, color=clr, alpha=0.25)
                ax.plot(theta, np.full_like(theta, 0.85), color=clr, linewidth=6, alpha=0.7)

            # needle
            needle_angle = np.pi * (1 - prob)
            ax.annotate("", xy=(needle_angle, 0.85), xytext=(needle_angle, 0),
                        arrowprops=dict(arrowstyle="->", color="white", lw=2))

            ax.set_ylim(0, 1.1)
            ax.set_theta_zero_location("W")
            ax.set_theta_direction(1)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines["polar"].set_visible(False)

            ax.text(np.pi/2, 1.25, f"{prob*100:.1f}%", ha="center", va="center",
                    color="white", fontsize=18, fontweight="bold", transform=ax.transData)

            low_p   = mpatches.Patch(color="#22c55e", label="Low")
            med_p   = mpatches.Patch(color="#f59e0b", label="Medium")
            high_p  = mpatches.Patch(color="#ef4444", label="High")
            ax.legend(handles=[low_p, med_p, high_p], loc="upper center",
                      ncol=3, frameon=False, labelcolor="white", fontsize=10,
                      bbox_to_anchor=(0.5, -0.08))

            st.pyplot(fig)
            plt.close(fig)

        # Feature breakdown
        with c_breakdown:
            st.subheader("📋 Application Summary")

            summary_data = {
                "Feature": [
                    "Age", "Gender", "Education", "Home Ownership",
                    "Annual Income", "Employment Exp.", "Credit Score",
                    "Credit Hist. Length", "Loan Amount", "Interest Rate",
                    "Loan Purpose", "Prior Default?"
                ],
                "Value": [
                    person_age, person_gender, person_education, person_home_ownership,
                    f"${person_income:,}", f"{person_emp_exp} yr(s)", credit_score,
                    f"{cb_person_cred_hist_length} yr(s)", f"${loan_amnt:,}",
                    f"{loan_int_rate:.1f}%", loan_intent, previous_loan_defaults_on_file
                ]
            }
            summary_df = pd.DataFrame(summary_data)

            def style_rows(row):
                return ["background-color: rgba(30,41,59,0.8); color: #e2e8f0"] * len(row)

            st.dataframe(
                summary_df.style.apply(style_rows, axis=1),
                use_container_width=True,
                hide_index=True,
                height=400,
            )

        st.markdown("---")

        # ── Risk factors ──────────────────────────────────────────────────────
        st.subheader("🔎 Risk Factor Analysis")
        factors = []

        if credit_score < 580:
            factors.append(("🔴 Poor Credit Score", f"{credit_score} is below 580 – high default risk.", "high"))
        elif credit_score < 670:
            factors.append(("🟡 Fair Credit Score", f"{credit_score} is fair (580–669) – moderate risk.", "medium"))
        else:
            factors.append(("🟢 Good Credit Score", f"{credit_score} is good – low risk indicator.", "low"))

        if loan_percent_income > 40:
            factors.append(("🔴 High Debt Burden", f"Loan-to-income ratio {loan_percent_income:.1f}% exceeds 40%.", "high"))
        elif loan_percent_income > 20:
            factors.append(("🟡 Moderate Debt Burden", f"Loan-to-income ratio {loan_percent_income:.1f}% is moderate.", "medium"))
        else:
            factors.append(("🟢 Low Debt Burden", f"Loan-to-income ratio {loan_percent_income:.1f}% is manageable.", "low"))

        if previous_loan_defaults_on_file == "Yes":
            factors.append(("🔴 Prior Default Recorded", "Historical default significantly increases risk.", "high"))
        else:
            factors.append(("🟢 No Prior Default", "Clean repayment history.", "low"))

        if loan_int_rate > 20:
            factors.append(("🔴 Very High Interest Rate", f"{loan_int_rate:.1f}% may strain repayment capacity.", "high"))
        elif loan_int_rate > 12:
            factors.append(("🟡 Elevated Interest Rate", f"{loan_int_rate:.1f}% is above average.", "medium"))

        if person_income < 30_000:
            factors.append(("🟡 Low Income", f"Annual income ${person_income:,} may limit repayment capacity.", "medium"))

        cols = st.columns(2)
        for i, (title, desc, level) in enumerate(factors):
            badge = {"high": "🔴", "medium": "🟡", "low": "🟢"}[level]
            with cols[i % 2]:
                st.markdown(f"""
                <div class="metric-card">
                    <strong style="color:#e2e8f0">{title}</strong><br>
                    <span style="color:#94a3b8; font-size:0.9rem">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

        # ── Recommendation ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💡 Recommendation")

        if is_default:
            st.error(
                f"**High Default Risk Detected.** The model predicts a **{result['probability']*100:.1f}%** "
                "probability of default. Consider requesting additional collateral, co-signer, "
                "or reducing the loan amount. A manual review is strongly recommended."
            )
        else:
            st.success(
                f"**Low Default Risk.** The model estimates only a **{result['probability']*100:.1f}%** "
                "chance of default. The applicant appears creditworthy based on the provided profile. "
                "Standard loan approval process can proceed."
            )

    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.exception(e)

# ─────────────────────────────────────────────────────────────────────────────
#  Default landing state
# ─────────────────────────────────────────────────────────────────────────────
else:
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#3b82f6">🤖 ML-Powered</h3>
            <p style="color:#94a3b8">Built on XGBoost with hyperparameter tuning via RandomizedSearchCV for maximum accuracy.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#8b5cf6">📊 13 Features</h3>
            <p style="color:#94a3b8">Analyses personal, credit, and loan attributes — numerical and categorical — through a full sklearn Pipeline.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#06b6d4">⚙️ Adjustable Threshold</h3>
            <p style="color:#94a3b8">Tune the decision boundary to balance precision vs recall based on your risk appetite.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Fill in the applicant details in the left sidebar and click **Predict Default Risk** to get started.")

    # How it works
    st.subheader("📖 How It Works")
    col1, col2, col3, col4 = st.columns(4)
    for col, num, title, desc in zip(
        [col1, col2, col3, col4],
        ["1", "2", "3", "4"],
        ["Input Data", "Preprocessing", "XGBoost Model", "Risk Report"],
        [
            "Enter applicant's personal info, credit profile, and loan details.",
            "Data passes through a sklearn Pipeline — scaling + one-hot encoding.",
            "Tuned XGBoost model outputs a default probability score.",
            "Get a detailed risk report with gauge, factors, and recommendation.",
        ]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center">
                <h2 style="color:#3b82f6; margin:0">{num}</h2>
                <h4 style="color:#e2e8f0; margin:6px 0">{title}</h4>
                <p style="color:#94a3b8; font-size:0.85rem">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#475569; font-size:0.85rem">'
    '🏦 Loan Default Predictor · Powered by XGBoost + Streamlit · '
    'For research & educational use only</p>'
    '<p style="text-align:center">Made with 💖 by Kunal Kushwaha</p>',
    unsafe_allow_html=True
)
