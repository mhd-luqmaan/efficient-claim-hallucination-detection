"""
Dataset Preprocessing Module for Hallucination Detection Pipeline
Downloads, parses, and formats the HaluEval QA dataset into claim-level training/eval splits.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import json
import logging
from typing import Dict, Any, List, Tuple
import requests
import pandas as pd
from sklearn.model_selection import train_test_split

from src.claim_extraction import extract_claims, get_spacy_nlp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HALUEVAL_QA_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
HALUEVAL_DIALOGUE_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/dialogue_data.json"
RAW_QA_PATH = "data/raw/qa_data.json"
RAW_DIALOGUE_PATH = "data/raw/dialogue_data.json"
PROCESSED_DIR = "data/processed"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"


def download_file_if_missing(url: str, dest_path: str) -> str:
    """Downloads dataset file if missing."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        logger.info(f"Dataset already exists at {dest_path} ({os.path.getsize(dest_path)} bytes).")
        return dest_path

    logger.info(f"Downloading from {url}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    logger.info(f"Downloaded to {dest_path} ({os.path.getsize(dest_path)} bytes).")
    return dest_path


def parse_and_extract_claims(
    qa_path: str = RAW_QA_PATH,
    dialogue_path: str = RAW_DIALOGUE_PATH,
    max_qa_samples: int = 5000,
    max_dialogue_samples: int = 5000
) -> pd.DataFrame:
    """
    Parses both QA and Dialogue JSON records, decomposes answers into individual claims,
    and maps labels:
    0 = Supported (from right_answer / right_response)
    1 = Hallucinated (from hallucinated_answer / hallucinated_response)
    Combining QA and Dialogue balances claim lengths across both classes.
    """
    logger.info("Parsing QA and Dialogue records...")
    nlp = get_spacy_nlp()
    all_claims: List[Dict[str, Any]] = []

    # 1. Parse QA Data
    if os.path.exists(qa_path):
        logger.info(f"Parsing QA samples from {qa_path}...")
        with open(qa_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if max_qa_samples and line_idx >= max_qa_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                qid = f"qa_{line_idx:05d}"
                question = data.get("question", "").strip()
                context = data.get("knowledge", "").strip()
                right_ans = data.get("right_answer", "").strip()
                hallu_ans = data.get("hallucinated_answer", "").strip()

                if not question or not (right_ans or hallu_ans):
                    continue

                # Extract full factual sentence from knowledge context containing right_ans
                factual_sents = []
                if context and right_ans:
                    for s in nlp(context).sents:
                        s_clean = s.text.strip()
                        if right_ans.lower() in s_clean.lower() and len(s_clean) > 10:
                            factual_sents.append(s_clean)

                if factual_sents:
                    # Use full factual sentence
                    for c in extract_claims(factual_sents[0], response_id=f"{qid}_supported", nlp=nlp):
                        all_claims.append({
                            "question_id": qid,
                            "response_id": c["response_id"],
                            "claim_id": c["claim_id"],
                            "question": question,
                            "context": context,
                            "claim_text": c["claim_text"],
                            "full_answer": factual_sents[0],
                            "label": 0,
                            "label_name": "Supported"
                        })
                elif right_ans:
                    for c in extract_claims(right_ans, response_id=f"{qid}_supported", nlp=nlp):
                        all_claims.append({
                            "question_id": qid,
                            "response_id": c["response_id"],
                            "claim_id": c["claim_id"],
                            "question": question,
                            "context": context,
                            "claim_text": c["claim_text"],
                            "full_answer": right_ans,
                            "label": 0,
                            "label_name": "Supported"
                        })

                if hallu_ans:
                    for c in extract_claims(hallu_ans, response_id=f"{qid}_hallucinated", nlp=nlp):
                        all_claims.append({
                            "question_id": qid,
                            "response_id": c["response_id"],
                            "claim_id": c["claim_id"],
                            "question": question,
                            "context": context,
                            "claim_text": c["claim_text"],
                            "full_answer": hallu_ans,
                            "label": 1,
                            "label_name": "Hallucinated"
                        })

    # 2. Parse Dialogue Data
    if os.path.exists(dialogue_path):
        logger.info(f"Parsing Dialogue samples from {dialogue_path}...")
        with open(dialogue_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if max_dialogue_samples and line_idx >= max_dialogue_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                did = f"dia_{line_idx:05d}"
                history = data.get("dialogue_history", "").strip()
                context = data.get("knowledge", "").strip()
                right_resp = data.get("right_response", "").strip()
                hallu_resp = data.get("hallucinated_response", "").strip()

                # Extract last user query from dialogue history
                turns = history.split("[Human]:")
                last_turn = turns[-1].split("[Assistant]:")[0].strip() if turns else history

                if not last_turn or not (right_resp or hallu_resp):
                    continue

                if right_resp:
                    for c in extract_claims(right_resp, response_id=f"{did}_supported", nlp=nlp):
                        all_claims.append({
                            "question_id": did,
                            "response_id": c["response_id"],
                            "claim_id": c["claim_id"],
                            "question": last_turn,
                            "context": context,
                            "claim_text": c["claim_text"],
                            "full_answer": right_resp,
                            "label": 0,
                            "label_name": "Supported"
                        })

                if hallu_resp:
                    for c in extract_claims(hallu_resp, response_id=f"{did}_hallucinated", nlp=nlp):
                        all_claims.append({
                            "question_id": did,
                            "response_id": c["response_id"],
                            "claim_id": c["claim_id"],
                            "question": last_turn,
                            "context": context,
                            "claim_text": c["claim_text"],
                            "full_answer": hallu_resp,
                            "label": 1,
                            "label_name": "Hallucinated"
                        })

    df = pd.DataFrame(all_claims)
    num_unique_q = df["question_id"].nunique() if not df.empty else 0
    logger.info(f"Processed {num_unique_q} QA/Dialogue contexts into {len(df)} claim records.")
    return df


def split_and_save_data(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset at the question level to eliminate data leakage.
    Saves splits to data/train/, data/processed/, and data/test/.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Question-level grouping
    unique_qids = df["question_id"].unique()
    val_test_size = val_ratio + test_ratio
    train_qids, val_test_qids = train_test_split(
        unique_qids, test_size=val_test_size, random_state=random_state
    )

    relative_test_size = test_ratio / val_test_size
    val_qids, test_qids = train_test_split(
        val_test_qids, test_size=relative_test_size, random_state=random_state
    )

    train_df = df[df["question_id"].isin(train_qids)].copy().reset_index(drop=True)
    val_df = df[df["question_id"].isin(val_qids)].copy().reset_index(drop=True)
    test_df = df[df["question_id"].isin(test_qids)].copy().reset_index(drop=True)

    # Save to CSV files
    all_path = os.path.join(PROCESSED_DIR, "claims_all.csv")
    train_path = os.path.join(TRAIN_DIR, "train_claims.csv")
    val_path = os.path.join(PROCESSED_DIR, "val_claims.csv")
    test_path = os.path.join(TEST_DIR, "test_claims.csv")

    df.to_csv(all_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Saved data splits successfully:")
    logger.info(f" - Full: {len(df)} claims -> {all_path}")
    logger.info(f" - Train: {len(train_df)} claims ({train_df['label'].value_counts().to_dict()}) -> {train_path}")
    logger.info(f" - Val:   {len(val_df)} claims ({val_df['label'].value_counts().to_dict()}) -> {val_path}")
    logger.info(f" - Test:  {len(test_df)} claims ({test_df['label'].value_counts().to_dict()}) -> {test_path}")

    return train_df, val_df, test_df


def run_preprocessing(max_qa: int = 5000, max_dia: int = 5000):
    """
    Main entrypoint for Phase 2 dataset preprocessing.
    """
    qa_file = download_file_if_missing(HALUEVAL_QA_URL, RAW_QA_PATH)
    dia_file = download_file_if_missing(HALUEVAL_DIALOGUE_URL, RAW_DIALOGUE_PATH)
    claims_df = parse_and_extract_claims(
        qa_path=qa_file,
        dialogue_path=dia_file,
        max_qa_samples=max_qa,
        max_dialogue_samples=max_dia
    )
    train_df, val_df, test_df = split_and_save_data(claims_df)
    return train_df, val_df, test_df


if __name__ == "__main__":
    run_preprocessing()
