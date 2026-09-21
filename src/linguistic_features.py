"""
Linguistic Feature Extraction Module
Extracts surface, entity, numerical, POS, and certainty/uncertainty features using spaCy.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

from typing import List, Dict, Any, Optional
import re
import spacy
from src.claim_extraction import get_spacy_nlp

# Lexicons for Epistemic Certainty and Uncertainty
CERTAINTY_WORDS = {
    "definitely", "certainly", "clearly", "undeniably", "absolutely",
    "truly", "proven", "indisputable", "obvious", "always", "conclusively",
    "unquestionably", "decidedly", "evident", "plainly"
}

UNCERTAINTY_WORDS = {
    "probably", "possibly", "maybe", "likely", "reportedly", "approximately",
    "alleged", "allegedly", "perhaps", "seem", "seems", "seemed", "seeming",
    "appear", "appears", "appeared", "suggest", "suggests", "suggested",
    "might", "could", "tentative", "supposedly", "estimated", "roughly", "about"
}

YEAR_PATTERN = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
PERCENT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%\b|\bpercent\b", re.IGNORECASE)
MONEY_PATTERN = re.compile(r"[\$\€\£\¥]\s*\d+|\b\d+\s*(?:dollars|euros|pounds|usd)\b", re.IGNORECASE)

ENTITY_TYPES = ["PERSON", "ORG", "GPE", "LOC", "DATE", "CARDINAL", "MONEY", "PERCENT"]


def extract_linguistic_features_from_doc(doc: spacy.tokens.Doc) -> Dict[str, float]:
    """
    Extracts numerical feature dictionary from a processed spaCy Doc.
    """
    text = doc.text
    tokens = [t for t in doc if not t.is_space]
    num_tokens = len(tokens)
    words = [t for t in tokens if not t.is_punct]
    num_words = len(words)
    char_count = len(text)

    avg_word_length = (sum(len(w.text) for w in words) / num_words) if num_words > 0 else 0.0

    # Named Entity Counts
    ent_counts = {f"entity_{ent.lower()}_count": 0 for ent in ENTITY_TYPES}
    total_ents = 0
    for ent in doc.ents:
        label = ent.label_
        if label in ENTITY_TYPES:
            key = f"entity_{label.lower()}_count"
            ent_counts[key] = ent_counts.get(key, 0) + 1
            total_ents += 1

    entity_density = (total_ents / num_words) if num_words > 0 else 0.0

    # Numerical Features
    year_matches = len(YEAR_PATTERN.findall(text))
    num_matches = len(NUMBER_PATTERN.findall(text))
    percent_matches = len(PERCENT_PATTERN.findall(text))
    money_matches = len(MONEY_PATTERN.findall(text))

    has_year = 1.0 if year_matches > 0 else 0.0
    has_numeric = 1.0 if num_matches > 0 else 0.0

    # Part of Speech Counts & Ratios
    noun_count = 0
    verb_count = 0
    adj_count = 0
    adv_count = 0
    pron_count = 0

    for t in tokens:
        pos = t.pos_
        if pos in ("NOUN", "PROPN"):
            noun_count += 1
        elif pos == "VERB":
            verb_count += 1
        elif pos == "ADJ":
            adj_count += 1
        elif pos == "ADV":
            adv_count += 1
        elif pos == "PRON":
            pron_count += 1

    denom = float(num_tokens) if num_tokens > 0 else 1.0
    noun_ratio = noun_count / denom
    verb_ratio = verb_count / denom
    adj_ratio = adj_count / denom
    adv_ratio = adv_count / denom
    pron_ratio = pron_count / denom

    # Certainty / Uncertainty Markers
    lower_words = [t.lower_ for t in words]
    certainty_hits = sum(1 for w in lower_words if w in CERTAINTY_WORDS)
    uncertainty_hits = sum(1 for w in lower_words if w in UNCERTAINTY_WORDS)

    certainty_score = (certainty_hits / num_words) if num_words > 0 else 0.0
    uncertainty_score = (uncertainty_hits / num_words) if num_words > 0 else 0.0

    features = {
        # Surface features
        "char_count": float(char_count),
        "word_count": float(num_words),
        "token_count": float(num_tokens),
        "sentence_length": float(num_tokens),
        "avg_word_length": float(avg_word_length),

        # Entities
        **{k: float(v) for k, v in ent_counts.items()},
        "total_entity_count": float(total_ents),
        "entity_density": float(entity_density),

        # Numerical
        "year_count": float(year_matches),
        "number_count": float(num_matches),
        "percent_count": float(percent_matches),
        "money_count": float(money_matches),
        "has_year": float(has_year),
        "has_numeric": float(has_numeric),

        # POS
        "noun_count": float(noun_count),
        "verb_count": float(verb_count),
        "adj_count": float(adj_count),
        "adv_count": float(adv_count),
        "pron_count": float(pron_count),
        "noun_ratio": float(noun_ratio),
        "verb_ratio": float(verb_ratio),
        "adj_ratio": float(adj_ratio),
        "adv_ratio": float(adv_ratio),
        "pron_ratio": float(pron_ratio),

        # Epistemic stance
        "certainty_count": float(certainty_hits),
        "uncertainty_count": float(uncertainty_hits),
        "certainty_score": float(certainty_score),
        "uncertainty_score": float(uncertainty_score),
    }

    return features


def extract_linguistic_features_single(text: str, nlp=None) -> Dict[str, float]:
    """
    Extracts linguistic features for a single text claim string.
    """
    if nlp is None:
        nlp = get_spacy_nlp()
    doc = nlp(text)
    return extract_linguistic_features_from_doc(doc)


def extract_linguistic_features_batch(
    texts: List[str],
    batch_size: int = 256,
    nlp=None
) -> List[Dict[str, float]]:
    """
    Extracts linguistic features across a list of texts using spaCy pipe for high efficiency.
    """
    if nlp is None:
        nlp = get_spacy_nlp()

    results = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        results.append(extract_linguistic_features_from_doc(doc))
    return results


if __name__ == "__main__":
    sample = "Marie Curie definitely discovered penicillin in 1928, costing $500."
    feats = extract_linguistic_features_single(sample)
    for k, v in feats.items():
        print(f"{k}: {v}")
