"""
text_detector.py
----------------
Core logic for AI vs Human text classification.
Loads pre-trained Naive Bayes model + TF-IDF vectorizer from /models directory.
All ML logic lives here — app.py just calls these functions.
"""

import pickle
import re
import os
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── NLTK downloads (silent) ──────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)

# ── Module-level globals (loaded once at import time) ────────────────────────
_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

# Resolve model paths relative to this file's location
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, "models")

_vectorizer = None
_model = None
_label_encoder = None


def _load_models():
    """Lazy-load models once and cache in module globals."""
    global _vectorizer, _model, _label_encoder

    if _vectorizer is None:
        with open(os.path.join(_MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            _vectorizer = pickle.load(f)

    if _model is None:
        with open(os.path.join(_MODELS_DIR, "nb_model.pkl"), "rb") as f:
            _model = pickle.load(f)

    if _label_encoder is None:
        with open(os.path.join(_MODELS_DIR, "label_encoder.pkl"), "rb") as f:
            _label_encoder = pickle.load(f)


# ── Text preprocessing (mirrors notebook exactly) ────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Clean and normalise raw text before vectorisation.
    Steps: lowercase → strip URLs → strip HTML → remove non-alpha →
           collapse whitespace → tokenise → remove stopwords → lemmatise.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(word)
        for word in tokens
        if word not in _stop_words and len(word) > 2
    ]

    return " ".join(tokens)


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(text: str) -> dict:
    """
    Run full inference pipeline on raw input text.

    Returns
    -------
    dict with keys:
        label        : "AI" or "Human"
        ai_prob      : float  (0–1)
        human_prob   : float  (0–1)
        confidence   : float  (probability of the predicted class, 0–1)
        cleaned_text : str    (preprocessed version, useful for debugging)
        top_features : list of (word, float) — top TF-IDF terms in this input
    """
    _load_models()

    cleaned = preprocess_text(text)

    if not cleaned.strip():
        return {
            "label": "Unknown",
            "ai_prob": 0.0,
            "human_prob": 0.0,
            "confidence": 0.0,
            "cleaned_text": cleaned,
            "top_features": [],
        }

    features = _vectorizer.transform([cleaned])
    pred_idx = _model.predict(features)[0]
    proba = _model.predict_proba(features)[0]

    # label_encoder: AI=0, Human=1  (confirmed from notebook output)
    ai_prob    = float(proba[0])
    human_prob = float(proba[1])
    label      = _label_encoder.inverse_transform([pred_idx])[0]  # "AI" or "Human"
    confidence = float(proba[pred_idx])

    top_features = _get_top_features(features, n=10)

    return {
        "label": label,
        "ai_prob": ai_prob,
        "human_prob": human_prob,
        "confidence": confidence,
        "cleaned_text": cleaned,
        "top_features": top_features,
    }


# ── Feature explanation ───────────────────────────────────────────────────────

def _get_top_features(tfidf_matrix, n: int = 10) -> list:
    """
    Return the top-n words from this document by TF-IDF weight.
    These are the words that most influenced the prediction.

    Returns list of (word, score) tuples sorted descending.
    """
    _load_models()

    feature_names = _vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]

    # Get indices of top-n non-zero scores
    top_indices = np.argsort(scores)[::-1][:n]
    top_indices = [i for i in top_indices if scores[i] > 0]

    return [(feature_names[i], float(scores[i])) for i in top_indices]


# ── Quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        "The proliferation of artificial intelligence technologies has precipitated a paradigm shift in computational methodologies.",
        "OMG I can't believe what happened at the mall yesterday, it was literally insane!!",
        "I woke up late this morning and rushed to get ready for work. The traffic was terrible as usual.",
    ]

    for s in samples:
        result = predict(s)
        print(f"\nText    : {s[:70]}...")
        print(f"Label   : {result['label']}")
        print(f"AI prob : {result['ai_prob']:.1%}")
        print(f"Human % : {result['human_prob']:.1%}")
        print(f"Confid. : {result['confidence']:.1%}")
        print(f"Top words: {[w for w, _ in result['top_features'][:5]]}")
