"""
Unit Test Suite for Hallucination Detection Pipeline
Covers claim extraction, semantic and linguistic feature extraction,
end-to-end prediction, risk aggregation, explainability signals, and edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import pytest
import numpy as np

from src.claim_extraction import extract_claims, sanitize_claim_text
from src.semantic_features import (
    extract_semantic_features_single,
    compute_cosine_similarity
)
from src.linguistic_features import extract_linguistic_features_single
from src.feature_fusion import (
    extract_features_single_claim,
    ALL_FEATURE_NAMES,
    SEMANTIC_FEATURE_NAMES,
    LINGUISTIC_FEATURE_NAMES
)
from src.predict import predict_hallucinations, generate_claim_explanations


class TestClaimExtraction:
    def test_single_sentence(self):
        text = "Marie Curie discovered radium in 1898."
        claims = extract_claims(text)
        assert len(claims) == 1
        assert claims[0]["claim_id"] == 1
        assert "Marie Curie discovered radium in 1898." in claims[0]["claim_text"]

    def test_multiple_sentences(self):
        text = "Elon Musk founded SpaceX in 2002. SpaceX is headquartered in Texas. It builds rockets."
        claims = extract_claims(text)
        assert len(claims) == 3
        assert claims[0]["claim_id"] == 1
        assert claims[1]["claim_id"] == 2
        assert claims[2]["claim_id"] == 3

    def test_sanitization_bullets_and_numbers(self):
        assert sanitize_claim_text("1. First claim.") == "First claim."
        assert sanitize_claim_text("- Second claim.") == "Second claim."
        assert sanitize_claim_text("• Third claim.") == "Third claim."

    def test_empty_and_whitespace(self):
        assert extract_claims("") == []
        assert extract_claims("   ") == []
        assert extract_claims("\n\t") == []

    def test_punctuation_only(self):
        assert extract_claims("... ??? !!!") == []

    def test_non_english_characters(self):
        text = "Albert Einstein a reçu le prix Nobel en 1921. Il est né en Allemagne."
        claims = extract_claims(text)
        assert len(claims) >= 1
        assert claims[0]["claim_id"] == 1


class TestFeatureExtraction:
    def test_semantic_features(self):
        q = "Who discovered penicillin?"
        c = "Alexander Fleming discovered penicillin in 1928."
        feats = extract_semantic_features_single(q, c, full_answer=c)

        for col in SEMANTIC_FEATURE_NAMES:
            assert col in feats

        assert -1.0 <= feats["question_claim_similarity"] <= 1.0
        assert 0.0 <= feats["question_claim_distance"] <= 2.0
        assert feats["claim_answer_similarity"] == pytest.approx(1.0, abs=1e-3)

    def test_linguistic_features(self):
        text = "Marie Curie definitely discovered penicillin in 1928, costing $500."
        feats = extract_linguistic_features_single(text)

        for col in LINGUISTIC_FEATURE_NAMES:
            assert col in feats

        assert feats["char_count"] == len(text)
        assert feats["word_count"] > 0
        assert feats["has_year"] == 1.0
        assert feats["has_numeric"] == 1.0
        assert feats["certainty_count"] >= 1.0

    def test_feature_fusion_all_columns_present(self):
        q = "Who invented the telephone?"
        c = "Alexander Graham Bell patented the telephone in 1876."
        combined = extract_features_single_claim(q, c, full_answer=c)

        assert len(combined) >= len(ALL_FEATURE_NAMES)
        for col in ALL_FEATURE_NAMES:
            assert col in combined, f"Missing column: {col}"
            assert isinstance(combined[col], (int, float, np.number))


class TestPredictionPipeline:
    def test_end_to_end_prediction(self):
        q = "Who discovered penicillin?"
        a = "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital in London."
        res = predict_hallucinations(q, a)

        assert "error" not in res
        assert res["total_claims"] == 1
        assert 0.0 <= res["overall_risk_percent"] <= 100.0
        assert res["overall_prediction"] in ("Supported", "Likely Hallucination")
        assert len(res["claims"]) == 1
        assert len(res["claims"][0]["model_signals"]) > 0

    def test_multi_claim_aggregation(self):
        q = "Tell me about SpaceX."
        a = "Elon Musk founded SpaceX in 2002. SpaceX is headquartered in Texas. SpaceX was originally called NASA Technologies."
        res = predict_hallucinations(q, a)

        assert res["total_claims"] == 3
        risks = [c["hallucination_risk"] for c in res["claims"]]
        expected_mean = round(sum(risks) / len(risks), 1)
        assert res["overall_risk_percent"] == pytest.approx(expected_mean, abs=0.2)
        assert res["max_risk_percent"] == max(risks)

    def test_empty_inputs(self):
        res_empty_q = predict_hallucinations("", "Some answer.")
        assert "error" in res_empty_q
        assert res_empty_q["overall_prediction"] == "N/A"

        res_empty_a = predict_hallucinations("Some question?", "")
        assert "error" in res_empty_a
        assert res_empty_a["overall_prediction"] == "N/A"

    def test_explainability_signals_generated(self):
        fake_feats = {
            "question_claim_similarity": 0.40,
            "question_claim_distance": 0.60,
            "claim_answer_similarity": 0.80,
            "context_claim_similarity": 0.0,
            "total_entity_count": 3.0,
            "has_year": 1.0,
            "has_numeric": 1.0,
            "certainty_score": 0.15,
            "uncertainty_score": 0.0,
        }
        signals = generate_claim_explanations(fake_feats, risk=0.85)
        assert len(signals) >= 2
        assert any("semantic" in s.lower() for s in signals)
        assert any("chronological" in s.lower() or "year" in s.lower() for s in signals)


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_pipeline.py"])
