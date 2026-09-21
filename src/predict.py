"""
Prediction & Explainability Pipeline Module
Performs end-to-end claim extraction, feature computation, calibrated risk scoring,
multi-claim aggregation, and feature-driven explanation generation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import joblib

from src.claim_extraction import extract_claims, get_spacy_nlp
from src.semantic_features import get_embedding_model
from src.feature_fusion import (
    extract_features_single_claim,
    ALL_FEATURE_NAMES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
BEST_MODEL_PATH = "models/best_model.joblib"
FEATURE_NAMES_PATH = "models/feature_names.json"

_PRELOADED_MODEL = None
_FEATURE_NAMES = None


def load_detector_model(model_path: str = BEST_MODEL_PATH):
    """
    Lazy loads and caches the calibrated detector model.
    """
    global _PRELOADED_MODEL, _FEATURE_NAMES
    if _PRELOADED_MODEL is None:
        if not os.path.exists(model_path):
            # Fallback to hybrid_xgboost or hybrid_random_forest if best_model not found
            fallback = "models/hybrid_xgboost.joblib"
            if os.path.exists(fallback):
                model_path = fallback
            else:
                model_path = "models/hybrid_random_forest.joblib"

        logger.info(f"Loading detector model from {model_path}...")
        _PRELOADED_MODEL = joblib.load(model_path)

    if _FEATURE_NAMES is None:
        if os.path.exists(FEATURE_NAMES_PATH):
            with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
                _FEATURE_NAMES = meta.get("all_features", ALL_FEATURE_NAMES)
        else:
            _FEATURE_NAMES = ALL_FEATURE_NAMES

    return _PRELOADED_MODEL, _FEATURE_NAMES


def generate_claim_explanations(features: Dict[str, float], risk: float) -> List[str]:
    """
    Generates human-interpretable model signal explanations from actual feature values.
    """
    signals = []

    q_c_sim = features.get("question_claim_similarity", 0.0)
    q_c_dist = features.get("question_claim_distance", 0.0)
    c_a_sim = features.get("claim_answer_similarity", 1.0)
    ctx_sim = features.get("context_claim_similarity", 0.0)
    ent_count = features.get("total_entity_count", 0.0)
    has_year = features.get("has_year", 0.0)
    has_num = features.get("has_numeric", 0.0)
    cert_score = features.get("certainty_score", 0.0)
    uncert_score = features.get("uncertainty_score", 0.0)

    # Semantic alignment signals
    if q_c_sim < 0.65:
        signals.append(f"Low semantic relevance between question and claim (similarity: {q_c_sim:.2f})")
    elif q_c_sim > 0.82:
        signals.append(f"Strong semantic alignment with question topic (similarity: {q_c_sim:.2f})")

    # Entity and Specificity signals
    if ent_count >= 2:
        if risk > 0.50:
            signals.append(f"High named-entity specificity ({int(ent_count)} entities) with elevated semantic tension")
        else:
            signals.append(f"Named entities ({int(ent_count)}) contextually aligned with query")

    # Numerical/Temporal assertions
    if has_year > 0:
        signals.append("Specific chronological assertion (year detected)")
    elif has_num > 0:
        signals.append("Specific numerical or quantitative assertion")

    # Stance and certainty
    if cert_score > 0.05:
        if risk > 0.50:
            signals.append("High epistemic certainty phrasing despite lower semantic support")
        else:
            signals.append("Strong epistemic certainty supported by context")
    elif uncert_score > 0.05:
        signals.append("Hedging or uncertainty markers present in claim")

    # Reference context if available
    if ctx_sim > 0.0:
        if ctx_sim < 0.50:
            signals.append(f"Discrepancy with provided reference context (context similarity: {ctx_sim:.2f})")
        else:
            signals.append(f"Corroborated by provided reference context (similarity: {ctx_sim:.2f})")

    # Default fallback signal
    if not signals:
        if risk > 0.50:
            signals.append(f"Overall statistical pattern divergence (Distance: {q_c_dist:.2f})")
        else:
            signals.append("Consistent surface and semantic feature profile")

    return signals


def predict_hallucinations(
    question: str,
    answer: str,
    context: str = "",
    threshold: float = 0.50,
    high_risk_threshold: float = 0.75,
    model=None,
    nlp=None,
    emb_model=None
) -> Dict[str, Any]:
    """
    End-to-end claim-level prediction function.

    Args:
        question: User's question text.
        answer: LLM generated answer text.
        context: Optional reference text.
        threshold: Risk threshold for labeling a claim as hallucinated (default 0.50).
        high_risk_threshold: Threshold for high-risk categorization (default 0.75).

    Returns:
        Structured prediction dictionary with overall metrics and claim breakdowns.
    """
    if not question.strip():
        return {
            "error": "Question cannot be empty.",
            "overall_risk": 0.0,
            "overall_prediction": "N/A",
            "claims": []
        }

    if not answer.strip():
        return {
            "error": "Answer cannot be empty.",
            "overall_risk": 0.0,
            "overall_prediction": "N/A",
            "claims": []
        }

    if model is None:
        model, feature_names = load_detector_model()
    else:
        feature_names = ALL_FEATURE_NAMES

    if nlp is None:
        nlp = get_spacy_nlp()

    if emb_model is None:
        emb_model = get_embedding_model()

    # 1. Segment answer into claims
    raw_claims = extract_claims(answer, nlp=nlp)
    if not raw_claims:
        return {
            "error": "No meaningful claims could be extracted from the answer.",
            "overall_risk": 0.0,
            "overall_prediction": "N/A",
            "claims": []
        }

    # 2. Extract features and predict per claim
    claim_results = []
    risks = []

    for c in raw_claims:
        claim_text = c["claim_text"]
        feats = extract_features_single_claim(
            question=question,
            claim=claim_text,
            full_answer=answer,
            context=context,
            model=emb_model,
            nlp=nlp
        )

        # Construct ordered feature vector
        feat_vector = np.array([[feats.get(col, 0.0) for col in feature_names]])

        # Model risk probability
        prob = float(model.predict_proba(feat_vector)[0, 1])
        risks.append(prob)

        prediction_label = "Likely Hallucination" if prob >= threshold else "Supported"
        is_high_risk = prob >= high_risk_threshold

        # Feature signals
        signals = generate_claim_explanations(feats, prob)

        claim_results.append({
            "claim_id": c["claim_id"],
            "claim_text": claim_text,
            "prediction": prediction_label,
            "hallucination_risk": round(prob * 100.0, 1),
            "risk_score_raw": prob,
            "is_high_risk": is_high_risk,
            "model_signals": signals,
            "features": {
                "question_claim_similarity": round(feats.get("question_claim_similarity", 0.0), 3),
                "question_claim_distance": round(feats.get("question_claim_distance", 0.0), 3),
                "claim_answer_similarity": round(feats.get("claim_answer_similarity", 0.0), 3),
                "context_claim_similarity": round(feats.get("context_claim_similarity", 0.0), 3),
                "total_entity_count": int(feats.get("total_entity_count", 0)),
                "has_year": bool(feats.get("has_year", 0)),
                "has_numeric": bool(feats.get("has_numeric", 0)),
                "certainty_count": int(feats.get("certainty_count", 0)),
                "uncertainty_count": int(feats.get("uncertainty_count", 0)),
                "word_count": int(feats.get("word_count", 0)),
            }
        })

    # 3. Overall Aggregation
    mean_risk = float(np.mean(risks))
    max_risk = float(np.max(risks))
    high_risk_count = sum(1 for r in risks if r >= high_risk_threshold)
    high_risk_percentage = float((high_risk_count / len(risks)) * 100.0)

    overall_prediction = "Likely Hallucination" if mean_risk >= threshold else "Supported"

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "total_claims": len(claim_results),
        "overall_risk_percent": round(mean_risk * 100.0, 1),
        "max_risk_percent": round(max_risk * 100.0, 1),
        "high_risk_claim_count": high_risk_count,
        "high_risk_percentage": round(high_risk_percentage, 1),
        "overall_prediction": overall_prediction,
        "claims": claim_results,
        "disclaimer": "Model signals are indicators used by the detector and are not independent proof that a claim is factually false."
    }


if __name__ == "__main__":
    test_q = "Who discovered penicillin?"
    test_a = "Marie Curie discovered penicillin in 1928."
    res = predict_hallucinations(test_q, test_a)
    print("Prediction Result:")
    print(json.dumps(res, indent=2))
