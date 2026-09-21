"""
Feature Fusion Module
Combines semantic and linguistic features into a unified feature table for ML models.
Manages feature subsets for ablation studies and handles scaling/persistence.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from src.semantic_features import (
    extract_semantic_features_single,
    extract_semantic_features_batch,
    get_embedding_model
)
from src.linguistic_features import (
    extract_linguistic_features_single,
    extract_linguistic_features_batch,
    ENTITY_TYPES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Categorized Feature Groups
SEMANTIC_FEATURE_NAMES = [
    "question_claim_similarity",
    "question_claim_distance",
    "claim_answer_similarity",
    "context_claim_similarity",
]

SURFACE_FEATURE_NAMES = [
    "char_count",
    "word_count",
    "token_count",
    "sentence_length",
    "avg_word_length",
]

ENTITY_FEATURE_NAMES = [
    f"entity_{ent.lower()}_count" for ent in ENTITY_TYPES
] + ["total_entity_count", "entity_density"]

NUMERICAL_FEATURE_NAMES = [
    "year_count",
    "number_count",
    "percent_count",
    "money_count",
    "has_year",
    "has_numeric",
]

POS_FEATURE_NAMES = [
    "noun_count",
    "verb_count",
    "adj_count",
    "adv_count",
    "pron_count",
    "noun_ratio",
    "verb_ratio",
    "adj_ratio",
    "adv_ratio",
    "pron_ratio",
]

CERTAINTY_FEATURE_NAMES = [
    "certainty_count",
    "uncertainty_count",
    "certainty_score",
    "uncertainty_score",
]

LINGUISTIC_FEATURE_NAMES = (
    SURFACE_FEATURE_NAMES
    + ENTITY_FEATURE_NAMES
    + NUMERICAL_FEATURE_NAMES
    + POS_FEATURE_NAMES
    + CERTAINTY_FEATURE_NAMES
)

ALL_FEATURE_NAMES = SEMANTIC_FEATURE_NAMES + LINGUISTIC_FEATURE_NAMES


def extract_features_single_claim(
    question: str,
    claim: str,
    full_answer: str = "",
    context: str = "",
    model=None,
    nlp=None
) -> Dict[str, float]:
    """
    Extracts all combined features for a single claim.
    """
    sem_feats = extract_semantic_features_single(
        question=question,
        claim=claim,
        full_answer=full_answer,
        context=context,
        model=model
    )
    ling_feats = extract_linguistic_features_single(
        text=claim,
        nlp=nlp
    )

    combined = {}
    combined.update(sem_feats)
    combined.update(ling_feats)
    return combined


def extract_features_dataset(
    df: pd.DataFrame,
    batch_size: int = 64
) -> pd.DataFrame:
    """
    Extracts unified feature DataFrame for a given dataset containing:
    ['question', 'claim_text', 'full_answer', 'context']
    """
    logger.info(f"Extracting features for {len(df)} records...")
    questions = df["question"].fillna("").astype(str).tolist()
    claims = df["claim_text"].fillna("").astype(str).tolist()
    full_answers = df["full_answer"].fillna("").astype(str).tolist() if "full_answer" in df.columns else [""] * len(df)
    contexts = df["context"].fillna("").astype(str).tolist() if "context" in df.columns else [""] * len(df)

    # 1. Semantic features (batched)
    logger.info("Computing semantic features...")
    sem_dict = extract_semantic_features_batch(
        questions=questions,
        claims=claims,
        full_answers=full_answers,
        contexts=contexts,
        batch_size=batch_size
    )
    sem_df = pd.DataFrame(sem_dict)

    # 2. Linguistic features (batched)
    logger.info("Computing linguistic features...")
    ling_list = extract_linguistic_features_batch(claims, batch_size=256)
    ling_df = pd.DataFrame(ling_list)

    # Combine into unified feature DataFrame
    features_df = pd.concat([sem_df, ling_df], axis=1)

    # Ensure all columns are present and ordered
    for col in ALL_FEATURE_NAMES:
        if col not in features_df.columns:
            features_df[col] = 0.0

    features_df = features_df[ALL_FEATURE_NAMES]

    # Append metadata and labels if available
    if "label" in df.columns:
        features_df["label"] = df["label"].values
    if "question_id" in df.columns:
        features_df["question_id"] = df["question_id"].values
    if "response_id" in df.columns:
        features_df["response_id"] = df["response_id"].values
    if "claim_id" in df.columns:
        features_df["claim_id"] = df["claim_id"].values
    if "claim_text" in df.columns:
        features_df["claim_text"] = df["claim_text"].values

    return features_df


def save_feature_names(output_path: str = "models/feature_names.json"):
    """
    Persists feature names and group definitions to JSON.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    meta = {
        "all_features": ALL_FEATURE_NAMES,
        "semantic_features": SEMANTIC_FEATURE_NAMES,
        "linguistic_features": LINGUISTIC_FEATURE_NAMES,
        "surface_features": SURFACE_FEATURE_NAMES,
        "entity_features": ENTITY_FEATURE_NAMES,
        "numerical_features": NUMERICAL_FEATURE_NAMES,
        "pos_features": POS_FEATURE_NAMES,
        "certainty_features": CERTAINTY_FEATURE_NAMES,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved feature metadata to {output_path}")


def process_and_save_all_features():
    """
    Loads train, val, and test claim CSVs, generates features, and saves them.
    Also fits and saves a StandardScaler on the training set.
    """
    train_path = "data/train/train_claims.csv"
    val_path = "data/processed/val_claims.csv"
    test_path = "data/test/test_claims.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at {train_path}. Run preprocessing first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    logger.info("Extracting Train features...")
    train_feat_df = extract_features_dataset(train_df)
    logger.info("Extracting Val features...")
    val_feat_df = extract_features_dataset(val_df)
    logger.info("Extracting Test features...")
    test_feat_df = extract_features_dataset(test_df)

    # Fit feature scaler on training features
    scaler = StandardScaler()
    scaler.fit(train_feat_df[ALL_FEATURE_NAMES].values)
    os.makedirs("models", exist_ok=True)
    scaler_path = "models/feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Fitted and saved feature scaler to {scaler_path}")

    # Save feature datasets
    train_feat_df.to_csv("data/train/train_features.csv", index=False)
    val_feat_df.to_csv("data/processed/val_features.csv", index=False)
    test_feat_df.to_csv("data/test/test_features.csv", index=False)

    save_feature_names()
    logger.info("All features extracted and persisted successfully!")


if __name__ == "__main__":
    process_and_save_all_features()
