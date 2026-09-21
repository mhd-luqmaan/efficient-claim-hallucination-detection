"""
Streamlit Application for Efficient Claim-Level Hallucination Detection
Combines SentenceTransformer (all-MiniLM-L6-v2) and calibrated Hybrid Tree Classifiers.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
import torch
import pandas as pd

from src.predict import predict_hallucinations, load_detector_model
from src.claim_extraction import get_spacy_nlp
from src.semantic_features import get_embedding_model

st.set_page_config(
    page_title="LLM Claim-Level Hallucination Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, premium aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .claim-card-supported {
        background-color: #F0FDF4;
        border-left: 5px solid #22C55E;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .claim-card-hallu {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .badge-supported {
        display: inline-block;
        background-color: #DCFCE7;
        color: #15803D;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
    }
    .badge-hallu {
        display: inline-block;
        background-color: #FEE2E2;
        color: #B91C1C;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
    }
    .signal-item {
        font-size: 0.9rem;
        color: #334155;
        margin-bottom: 0.25rem;
    }
    .disclaimer-box {
        background-color: #F1F5F9;
        border-left: 4px solid #94A3B8;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_models():
    """Initializes and caches models once during app startup."""
    nlp = get_spacy_nlp()
    emb = get_embedding_model()
    model, feature_names = load_detector_model()
    return nlp, emb, model, feature_names


nlp, emb_model, detector_model, feature_names = init_models()

# -------------------------------------------------------------
# Sidebar: System Config & Presets
# -------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Hardware detection
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU Execution"
    st.caption(f"**Compute Device:** `{device_name}`")
    st.caption("**Embedding Model:** `all-MiniLM-L6-v2` (384-d)")
    st.caption("**Classifier:** `Calibrated Hybrid Tree Ensemble`")

    st.divider()
    st.subheader("🎯 Risk Thresholds")
    threshold = st.slider("Hallucination Risk Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    high_risk_threshold = st.slider("High Risk Flag Threshold", min_value=0.50, max_value=0.95, value=0.75, step=0.05)

    st.divider()
    st.subheader("🧪 Demo Scenarios")
    preset_choice = st.selectbox(
        "Select a preconfigured test scenario:",
        [
            "Custom Input",
            "1. Factual Answer (Expected Low Risk)",
            "2. Clearly Hallucinated (Expected High Risk)",
            "3. Mixed Multi-Claim Answer",
            "4. Long Historical Narrative"
        ]
    )

    st.divider()
    with st.expander("📊 Benchmark Highlights"):
        st.markdown("""
        **Test Set Evaluation (4,228 claims):**
        - **Hybrid Model F1:** 0.7838 (RF) / 0.7798 (XGBoost)
        - **ROC-AUC:** 0.8550
        - **Latency:** 0.001 ms / claim
        - **Throughput:** > 980,000 claims/sec
        - **Zero LLM APIs or GPUs required at inference.**
        """)

# Preset Definitions
preset_data = {
    "1. Factual Answer (Expected Low Risk)": {
        "question": "Who discovered penicillin?",
        "answer": "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital in London.",
        "context": "Sir Alexander Fleming was a Scottish physician and microbiologist who discovered penicillin in 1928."
    },
    "2. Clearly Hallucinated (Expected High Risk)": {
        "question": "Who discovered penicillin?",
        "answer": "Marie Curie discovered penicillin in 1928 while conducting radioactivity research in Paris.",
        "context": "Sir Alexander Fleming was a Scottish physician and microbiologist who discovered penicillin in 1928."
    },
    "3. Mixed Multi-Claim Answer": {
        "question": "Who founded SpaceX and what was its background?",
        "answer": "Elon Musk founded SpaceX in 2002. SpaceX is headquartered in Texas. SpaceX was originally called NASA Technologies.",
        "context": "SpaceX was founded in 2002 by Elon Musk with the goal of reducing space transportation costs."
    },
    "4. Long Historical Narrative": {
        "question": "Tell me about the Apollo 11 moon mission.",
        "answer": "The Apollo 11 mission was launched on July 16, 1969. Neil Armstrong was the mission commander. The lunar module landed on Mars after three days of travel. Armstrong became the first human to step onto the surface.",
        "context": "Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969."
    }
}

# -------------------------------------------------------------
# Main Application Content
# -------------------------------------------------------------
st.markdown('<div class="main-title">🔍 Claim-Level Hallucination Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Efficient hallucination detection in LLM answers using compact semantic embeddings and linguistic features.</div>',
    unsafe_allow_html=True
)

# Populate inputs based on preset or custom
default_q = ""
default_a = ""
default_ctx = ""

if preset_choice in preset_data:
    default_q = preset_data[preset_choice]["question"]
    default_a = preset_data[preset_choice]["answer"]
    default_ctx = preset_data[preset_choice]["context"]

col1, col2 = st.columns([1, 1])

with col1:
    question_input = st.text_input("User Question:", value=default_q, placeholder="e.g. Who discovered penicillin?")
    context_input = st.text_area(
        "Optional Reference / Context (leave blank for zero-context detection):",
        value=default_ctx,
        height=100,
        placeholder="Optional source text or knowledge snippet..."
    )

with col2:
    answer_input = st.text_area(
        "LLM Generated Answer:",
        value=default_a,
        height=188,
        placeholder="Enter the response generated by the LLM..."
    )

detect_button = st.button("🚀 Detect Hallucinations", type="primary", use_container_width=True)

# -------------------------------------------------------------
# Inference Execution & Results Display
# -------------------------------------------------------------
if detect_button or (preset_choice != "Custom Input"):
    if not question_input.strip() or not answer_input.strip():
        st.warning("Please provide both a Question and an LLM Answer.")
    else:
        with st.spinner("Analyzing semantic distance, entity alignment, and linguistic stance..."):
            results = predict_hallucinations(
                question=question_input,
                answer=answer_input,
                context=context_input,
                threshold=threshold,
                high_risk_threshold=high_risk_threshold,
                model=detector_model,
                nlp=nlp,
                emb_model=emb_model
            )

        if "error" in results:
            st.error(results["error"])
        else:
            st.divider()

            # Overall Score Section
            st.subheader("📊 Overall Answer Assessment")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)

            with m_col1:
                overall_risk = results["overall_risk_percent"]
                st.metric(
                    label="Overall Hallucination Risk",
                    value=f"{overall_risk}%",
                    delta="Suspicious" if overall_risk >= threshold * 100 else "Factual",
                    delta_color="inverse"
                )

            with m_col2:
                max_risk = results["max_risk_percent"]
                st.metric(
                    label="Peak Claim Risk",
                    value=f"{max_risk}%",
                    delta="High Risk" if max_risk >= high_risk_threshold * 100 else "Normal",
                    delta_color="inverse"
                )

            with m_col3:
                st.metric(
                    label="Total Claims Identified",
                    value=results["total_claims"]
                )

            with m_col4:
                high_risk_n = results["high_risk_claim_count"]
                st.metric(
                    label="High-Risk Claims",
                    value=f"{high_risk_n} ({results['high_risk_percentage']}%)"
                )

            # Overall Progress Bar
            risk_norm = min(1.0, max(0.0, results["overall_risk_percent"] / 100.0))
            st.progress(risk_norm)

            # -------------------------------------------------------------
            # Claim-Level Breakdown
            # -------------------------------------------------------------
            st.subheader("📑 Claim-by-Claim Verification")
            st.caption("Each sentence is independently extracted, parsed for linguistic and semantic features, and evaluated.")

            for claim in results["claims"]:
                is_hallu = claim["prediction"] == "Likely Hallucination"
                card_class = "claim-card-hallu" if is_hallu else "claim-card-supported"
                badge_class = "badge-hallu" if is_hallu else "badge-supported"
                badge_icon = "⚠" if is_hallu else "✓"

                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                        <span style="font-weight: 700; color: #1E293B;">Claim {claim['claim_id']}</span>
                        <span class="{badge_class}">{badge_icon} {claim['prediction']} (Risk: {claim['hallucination_risk']}%)</span>
                    </div>
                    <div style="font-size: 1.05rem; color: #0F172A; margin-bottom: 0.8rem; font-style: italic;">
                        "{claim['claim_text']}"
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c_cols = st.columns([2, 1])
                with c_cols[0]:
                    st.markdown("**Important Model Signals:**")
                    for sig in claim["model_signals"]:
                        st.markdown(f"• {sig}")

                with c_cols[1]:
                    with st.expander(f"Inspect Features (Claim {claim['claim_id']})"):
                        feat_df = pd.DataFrame(
                            list(claim["features"].items()),
                            columns=["Feature", "Value"]
                        )
                        st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # Disclaimer
            st.markdown(f"""
            <div class="disclaimer-box">
                <b>📌 Important Disclaimer:</b> {results['disclaimer']}
            </div>
            """, unsafe_allow_html=True)
