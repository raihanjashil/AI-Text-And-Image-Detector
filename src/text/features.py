import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import MAX_FEATURES, NGRAM_RANGE, MIN_DF, MAX_DF


def build_vectorizer():
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        stop_words='english',
        sublinear_tf=True,
        smooth_idf=True,
        dtype='float32',
    )


def save_vectorizer(vectorizer, path):
    joblib.dump(vectorizer, path)


def load_vectorizer(path):
    return joblib.load(path)
