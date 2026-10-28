"""
Semantic Feature Extraction Module
Extracts cosine similarities, semantic distances, and embeddings using all-MiniLM-L6-v2.
Includes in-memory caching to eliminate redundant encodings.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

from typing import List, Dict, Any, Union, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

_EMBEDDING_MODEL = None
_EMBEDDING_CACHE: Dict[str, np.ndarray] = {}
CACHE_MAX_SIZE = 50000


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2", device: str = "auto") -> SentenceTransformer:
    """
    Lazy loads the SentenceTransformer model on the appropriate device.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device

        _EMBEDDING_MODEL = SentenceTransformer(model_name, device=resolved_device)
    return _EMBEDDING_MODEL


def get_cached_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
    """
    Retrieves or calculates normalized embedding for a text string with caching.
    """
    global _EMBEDDING_CACHE
    if text in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[text]

    emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    # Evict if cache exceeds max size
    if len(_EMBEDDING_CACHE) >= CACHE_MAX_SIZE:
        # Clear half of cache
        keys_to_remove = list(_EMBEDDING_CACHE.keys())[:CACHE_MAX_SIZE // 2]
        for k in keys_to_remove:
            del _EMBEDDING_CACHE[k]

    _EMBEDDING_CACHE[text] = emb
    return emb


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculates cosine similarity between two normalized vectors.
    """
    if vec_a is None or vec_b is None:
        return 0.0
    # If already L2 normalized: dot product equals cosine similarity
    dot = float(np.dot(vec_a, vec_b))
    return max(-1.0, min(1.0, dot))


def extract_semantic_features_single(
    question: str,
    claim: str,
    full_answer: str = "",
    context: str = "",
    model: Optional[SentenceTransformer] = None
) -> Dict[str, float]:
    """
    Calculates semantic similarity metrics for a single claim.

    Features:
    - question_claim_similarity
    - question_claim_distance (1 - sim)
    - claim_answer_similarity
    - context_claim_similarity
    """
    if model is None:
        model = get_embedding_model()

    q_emb = get_cached_embedding(question.strip(), model) if question.strip() else None
    c_emb = get_cached_embedding(claim.strip(), model) if claim.strip() else None
    a_emb = get_cached_embedding(full_answer.strip(), model) if full_answer.strip() else None
    ctx_emb = get_cached_embedding(context.strip(), model) if context.strip() else None

    # Question - Claim
    if q_emb is not None and c_emb is not None:
        q_c_sim = compute_cosine_similarity(q_emb, c_emb)
        q_c_dist = float(1.0 - q_c_sim)
    else:
        q_c_sim = 0.0
        q_c_dist = 1.0

    # Claim - Full Answer
    if c_emb is not None and a_emb is not None:
        c_a_sim = compute_cosine_similarity(c_emb, a_emb)
    else:
        c_a_sim = 1.0 if (claim.strip() and claim.strip() == full_answer.strip()) else 0.0

    # Context - Claim (with sentence-level alignment and closed-book imputation)
    if ctx_emb is not None and c_emb is not None:
        ctx_c_sim = compute_cosine_similarity(ctx_emb, c_emb)
        # Check sentence-level max similarity if context has multiple sentences
        if "." in context or "\n" in context:
            ctx_sents = [s.strip() for s in context.replace("\n", ". ").split(".") if len(s.strip()) > 5]
            if len(ctx_sents) > 1:
                sent_sims = [compute_cosine_similarity(c_emb, get_cached_embedding(s, model)) for s in ctx_sents]
                if sent_sims:
                    ctx_c_sim = max(ctx_c_sim, float(max(sent_sims)))
    else:
        # Closed-book fallback: Question relevance acts as the reference
        ctx_c_sim = q_c_sim if q_c_sim > 0.0 else 0.50

    return {
        "question_claim_similarity": q_c_sim,
        "question_claim_distance": q_c_dist,
        "claim_answer_similarity": c_a_sim,
        "context_claim_similarity": ctx_c_sim
    }


def extract_semantic_features_batch(
    questions: List[str],
    claims: List[str],
    full_answers: List[str] = None,
    contexts: List[str] = None,
    batch_size: int = 64,
    model: Optional[SentenceTransformer] = None
) -> Dict[str, np.ndarray]:
    """
    Efficiently extracts semantic features in vectorized batches.
    """
    if model is None:
        model = get_embedding_model()

    n = len(claims)
    if full_answers is None:
        full_answers = [""] * n
    if contexts is None:
        contexts = [""] * n

    # Unique text encoding to avoid redundant computations
    all_unique_texts = list(set(questions + claims + [a for a in full_answers if a] + [c for c in contexts if c]))
    uncached = [t for t in all_unique_texts if t not in _EMBEDDING_CACHE and t.strip()]

    if uncached:
        embeddings = model.encode(
            uncached,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        for text, emb in zip(uncached, embeddings):
            _EMBEDDING_CACHE[text] = emb

    # Compute features for all pairs
    q_c_sims = np.zeros(n, dtype=np.float32)
    q_c_dists = np.zeros(n, dtype=np.float32)
    c_a_sims = np.zeros(n, dtype=np.float32)
    ctx_c_sims = np.zeros(n, dtype=np.float32)

    zero_vec = np.zeros(384, dtype=np.float32)

    for i in range(n):
        q_emb = _EMBEDDING_CACHE.get(questions[i].strip(), zero_vec) if questions[i].strip() else zero_vec
        c_emb = _EMBEDDING_CACHE.get(claims[i].strip(), zero_vec) if claims[i].strip() else zero_vec
        a_emb = _EMBEDDING_CACHE.get(full_answers[i].strip(), zero_vec) if full_answers[i].strip() else zero_vec
        ctx_emb = _EMBEDDING_CACHE.get(contexts[i].strip(), zero_vec) if contexts[i].strip() else zero_vec

        # Q-C
        sim = float(np.dot(q_emb, c_emb))
        q_c_sims[i] = sim
        q_c_dists[i] = 1.0 - sim

        # C-A
        c_a_sims[i] = float(np.dot(c_emb, a_emb))

        # C-CTX
        ctx_c_sims[i] = float(np.dot(ctx_emb, c_emb))

    return {
        "question_claim_similarity": q_c_sims,
        "question_claim_distance": q_c_dists,
        "claim_answer_similarity": c_a_sims,
        "context_claim_similarity": ctx_c_sims
    }


if __name__ == "__main__":
    q = "Who discovered penicillin?"
    c1 = "Alexander Fleming discovered penicillin in 1928."
    c2 = "Marie Curie discovered penicillin in 1928."
    
    f1 = extract_semantic_features_single(q, c1, full_answer=c1)
    f2 = extract_semantic_features_single(q, c2, full_answer=c2)
    print("Claim 1 (Fleming):", f1)
    print("Claim 2 (Curie):", f2)
