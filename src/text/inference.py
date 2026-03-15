import joblib
from .config import MODELS_DIR
from .preprocess import preprocess_text
from .features import load_vectorizer


class TextDetector:
    def __init__(self, model_path=MODELS_DIR / 'nb_model.pkl'):
        self.vectorizer = load_vectorizer(MODELS_DIR / 'tfidf_vectorizer.pkl')
        self.label_encoder = joblib.load(MODELS_DIR / 'label_encoder.pkl')
        self.model = joblib.load(model_path)

    def predict(self, text: str):
        cleaned = preprocess_text(text)
        X = self.vectorizer.transform([cleaned])
        pred = self.model.predict(X)[0]
        label = self.label_encoder.inverse_transform([pred])[0]
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            confidence = float(self.model.predict_proba(X).max())
        return {'label': label, 'confidence': confidence}
