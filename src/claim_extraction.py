"""
Claim Extraction Module
Splits LLM generated answers into individual claims using spaCy sentence segmentation.
"""

from typing import List, Dict, Any, Optional
import re
import spacy

_nlp_model = None

def get_spacy_nlp(model_name: str = "en_core_web_sm"):
    """
    Lazy load spaCy NLP model to conserve memory.
    """
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load(model_name)
        except Exception:
            # Fallback if model not found
            _nlp_model = spacy.blank("en")
            if "sentencizer" not in _nlp_model.pipe_names:
                _nlp_model.add_pipe("sentencizer")
    return _nlp_model


def sanitize_claim_text(text: str) -> str:
    """
    Cleans leading bullet points, numbers, and extra spaces from claim text.
    """
    # Remove leading list markers like "1.", "1)", "- ", "* ", "• "
    cleaned = re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s*", "", text)
    # Remove extra spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_claims(
    answer_text: str,
    response_id: Optional[str] = None,
    nlp=None
) -> List[Dict[str, Any]]:
    """
    Segments an answer string into a structured list of individual claims.

    Args:
        answer_text: The complete answer text to segment.
        response_id: Optional unique identifier for the response.
        nlp: Optional preloaded spaCy model.

    Returns:
        List of claim dictionaries with keys:
        - response_id
        - claim_id (1-indexed int)
        - claim_text
        - char_start
        - char_end
    """
    if not answer_text or not answer_text.strip():
        return []

    if nlp is None:
        nlp = get_spacy_nlp()

    doc = nlp(answer_text)
    claims: List[Dict[str, Any]] = []
    claim_counter = 1

    for sent in doc.sents:
        raw_text = sent.text.strip()
        cleaned_text = sanitize_claim_text(raw_text)

        # Filter out empty or trivial fragments (< 3 chars or punctuation only)
        if len(cleaned_text) < 3 or not any(c.isalnum() for c in cleaned_text):
            continue

        claims.append({
            "response_id": response_id or "resp_0",
            "claim_id": claim_counter,
            "claim_text": cleaned_text,
            "raw_claim_text": raw_text,
            "char_start": sent.start_char,
            "char_end": sent.end_char,
        })
        claim_counter += 1

    # Fallback if no sentences detected but text is present
    if not claims and answer_text.strip():
        cleaned_text = sanitize_claim_text(answer_text.strip())
        if len(cleaned_text) >= 3 and any(c.isalnum() for c in cleaned_text):
            claims.append({
                "response_id": response_id or "resp_0",
                "claim_id": 1,
                "claim_text": cleaned_text,
                "raw_claim_text": answer_text.strip(),
                "char_start": 0,
                "char_end": len(answer_text),
            })

    return claims


if __name__ == "__main__":
    sample = "Elon Musk founded SpaceX in 2002. The company is headquartered in Texas. SpaceX was originally called NASA Technologies."
    claims = extract_claims(sample, response_id="sample_1")
    for c in claims:
        print(f"Claim {c['claim_id']}: {c['claim_text']}")
