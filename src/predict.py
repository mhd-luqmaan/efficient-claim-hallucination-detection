"""
Prediction & Explainability Pipeline Module
Performs end-to-end claim extraction, feature computation, calibrated risk scoring,
multi-claim aggregation, grounded entity verification, and feature-driven explanation generation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import joblib

from src.claim_extraction import extract_claims, get_spacy_nlp
from src.semantic_features import (
    get_embedding_model,
    compute_cosine_similarity,
    get_cached_embedding
)
from src.feature_fusion import (
    extract_features_single_claim,
    ALL_FEATURE_NAMES,
    SEMANTIC_FEATURE_NAMES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
BEST_MODEL_PATH = "models/hybrid_xgboost.joblib" if os.path.exists("models/hybrid_xgboost.joblib") else "models/best_model.joblib"
FEATURE_NAMES_PATH = "models/feature_names.json"

_PRELOADED_MODEL = None
_FEATURE_NAMES = None
_AUX_SEMANTIC_MODEL = None


def load_detector_model(model_path: Optional[str] = None):
    """
    Lazy loads and caches the calibrated detector model.
    Defaults to hybrid_xgboost (or best_model fallback) for robust feature weighting.
    """
    global _PRELOADED_MODEL, _FEATURE_NAMES
    if model_path is None:
        model_path = BEST_MODEL_PATH
        if not os.path.exists(model_path):
            model_path = "models/best_model.joblib"

    if _PRELOADED_MODEL is None or model_path != BEST_MODEL_PATH:
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


def get_aux_semantic_model():
    """Lazy loads auxiliary semantic baseline model if available."""
    global _AUX_SEMANTIC_MODEL
    if _AUX_SEMANTIC_MODEL is None:
        sem_path = "models/baseline_semantic_rf.joblib"
        if os.path.exists(sem_path):
            try:
                _AUX_SEMANTIC_MODEL = joblib.load(sem_path)
            except Exception as e:
                logger.warning(f"Could not load auxiliary semantic model: {e}")
    return _AUX_SEMANTIC_MODEL


def check_entity_support(claim_text: str, context: str, nlp) -> Dict[str, Any]:
    """
    Checks if named entities in the claim are grounded in the reference context.
    Identifies specific entity replacements or fabricated entities.
    """
    if not context or not context.strip():
        return {
            "has_entities": False,
            "supported_entities": [],
            "unsupported_entities": [],
            "support_ratio": 1.0
        }

    claim_doc = nlp(claim_text)
    claim_ents = []
    for ent in claim_doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "DATE", "FAC", "PRODUCT"}:
            clean_ent = ent.text.lower().strip()
            if len(clean_ent) > 1 and clean_ent not in claim_ents:
                claim_ents.append(clean_ent)

    if not claim_ents:
        return {
            "has_entities": False,
            "supported_entities": [],
            "unsupported_entities": [],
            "support_ratio": 1.0
        }

    ctx_lower = context.lower()
    supported = [ent for ent in claim_ents if ent in ctx_lower]
    unsupported = [ent for ent in claim_ents if ent not in ctx_lower]

    ratio = len(supported) / len(claim_ents) if claim_ents else 1.0
    return {
        "has_entities": True,
        "supported_entities": supported,
        "unsupported_entities": unsupported,
        "support_ratio": ratio
    }


def compute_grounded_risk(
    base_model_prob: float,
    feats: Dict[str, float],
    has_context: bool,
    ent_info: Dict[str, Any]
) -> Tuple[float, List[str]]:
    """
    Calibrates risk by grounding claims against context, entity consistency,
    and question answering relevance without spurious sentence length biases.
    """
    signals = []
    q_c_sim = feats.get("question_claim_similarity", 0.0)
    q_c_dist = feats.get("question_claim_distance", 0.0)
    ctx_sim = feats.get("context_claim_similarity", 0.0)
    unsupp_ents = ent_info.get("unsupported_entities", [])
    supp_ents = ent_info.get("supported_entities", [])
    has_ents = ent_info.get("has_entities", False)

    # Context Grounding Mode
    if has_context:
        # 1. High Context Entailment
        if ctx_sim >= 0.78 and len(unsupp_ents) == 0:
            calibrated_prob = min(base_model_prob * 0.35, 0.22)
            signals.append(f"Strongly corroborated by reference context (similarity: {ctx_sim:.2f})")
            if supp_ents:
                signals.append(f"Entities verified in context: {', '.join(supp_ents[:3])}")
        elif ctx_sim >= 0.70 and len(unsupp_ents) == 0:
            calibrated_prob = min(base_model_prob * 0.60, 0.38)
            signals.append(f"Contextually aligned with reference context (similarity: {ctx_sim:.2f})")
        elif len(unsupp_ents) > 0 and ctx_sim < 0.68:
            # Explicit Entity Hallucination
            calibrated_prob = max(base_model_prob, 0.78)
            signals.append(f"Discrepancy: Entities not grounded in context: {', '.join(unsupp_ents[:3])}")
            signals.append(f"Low reference context alignment (similarity: {ctx_sim:.2f})")
        elif ctx_sim < 0.52:
            calibrated_prob = max(base_model_prob, 0.75)
            signals.append(f"Contradiction or unsupported by reference context (similarity: {ctx_sim:.2f})")
        else:
            calibrated_prob = base_model_prob
            if ctx_sim >= 0.60:
                signals.append(f"Moderate alignment with reference context (similarity: {ctx_sim:.2f})")
            else:
                signals.append(f"Partial semantic tension with context (similarity: {ctx_sim:.2f})")

    # Closed-Book Mode (No reference context provided)
    else:
        # Factual alignment based on question relevance and answer cohesion
        if q_c_sim >= 0.75:
            # Strongly on-topic and direct answer to question
            calibrated_prob = min(base_model_prob * 0.65, 0.45)
            signals.append(f"Strong topical alignment with question (similarity: {q_c_sim:.2f})")
        elif q_c_sim < 0.48:
            calibrated_prob = max(base_model_prob, 0.72)
            signals.append(f"Low semantic relevance to question topic (similarity: {q_c_sim:.2f})")
        else:
            calibrated_prob = base_model_prob

    # Additional linguistic stance and specificity signals
    if feats.get("certainty_score", 0.0) > 0.05:
        if calibrated_prob > 0.50:
            signals.append("High epistemic certainty phrasing despite lower factual support")
        else:
            signals.append("Confident assertion supported by content")
    elif feats.get("uncertainty_score", 0.0) > 0.05:
        signals.append("Hedging or uncertainty markers present in claim")

    if feats.get("has_year", 0.0) > 0:
        signals.append("Specific chronological assertion (year detected)")
    elif feats.get("has_numeric", 0.0) > 0:
        signals.append("Specific numerical or quantitative assertion")

    if not signals:
        if calibrated_prob > 0.50:
            signals.append(f"Statistical divergence from expected patterns (Distance: {q_c_dist:.2f})")
        else:
            signals.append("Consistent surface and semantic feature profile")

    calibrated_prob = float(np.clip(calibrated_prob, 0.01, 0.99))
    return calibrated_prob, signals


def generate_claim_explanations(features: Dict[str, float], risk: float) -> List[str]:
    """
    Generates human-interpretable model signal explanations from actual feature values.
    Maintained for direct utility and backward compatibility with unit tests.
    """
    signals = []
    q_c_sim = features.get("question_claim_similarity", 0.0)
    q_c_dist = features.get("question_claim_distance", 0.0)
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
    End-to-end claim-level prediction function with calibrated grounding and explainability.

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

    aux_sem_model = get_aux_semantic_model()
    has_context = bool(context and context.strip())

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

        # Base model prediction
        feat_vector = np.array([[feats.get(col, 0.0) for col in feature_names]])
        base_prob = float(model.predict_proba(feat_vector)[0, 1])

        # Auxiliary semantic baseline ensemble if available
        if aux_sem_model is not None:
            sem_vec = np.array([[feats.get(col, 0.0) for col in SEMANTIC_FEATURE_NAMES]])
            sem_prob = float(aux_sem_model.predict_proba(sem_vec)[0, 1])
            # Blend base classifier with pure semantic classifier to mitigate length bias
            raw_prob = 0.55 * base_prob + 0.45 * sem_prob
        else:
            raw_prob = base_prob

        # Entity support against context
        ent_info = check_entity_support(claim_text, context, nlp=nlp)

        # Calibrated Grounded Risk
        final_prob, signals = compute_grounded_risk(
            base_model_prob=raw_prob,
            feats=feats,
            has_context=has_context,
            ent_info=ent_info
        )
        risks.append(final_prob)

        prediction_label = "Likely Hallucination" if final_prob >= threshold else "Supported"
        is_high_risk = final_prob >= high_risk_threshold

        claim_results.append({
            "claim_id": c["claim_id"],
            "claim_text": claim_text,
            "prediction": prediction_label,
            "hallucination_risk": round(final_prob * 100.0, 1),
            "risk_score_raw": final_prob,
            "is_high_risk": is_high_risk,
            "model_signals": signals,
            "grounding": {
                "has_context": has_context,
                "supported_entities": ent_info.get("supported_entities", []),
                "unsupported_entities": ent_info.get("unsupported_entities", []),
            },
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
    test_a = "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital in London."
    test_ctx = "Sir Alexander Fleming was a Scottish physician and microbiologist who discovered penicillin in 1928 at St. Mary's Hospital in London."
    res = predict_hallucinations(test_q, test_a, test_ctx)
    print("Prediction Result:")
    print(json.dumps(res, indent=2))
