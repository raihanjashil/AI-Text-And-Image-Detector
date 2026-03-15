import joblib
from sklearn.naive_bayes import MultinomialNB
from .config import MODELS_DIR
from .data import load_dataset, split_data
from .features import build_vectorizer, save_vectorizer


def main():
    df = load_dataset()
    X_train_text, X_val_text, X_test_text, y_train, y_val, y_test, label_encoder = split_data(df)

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)

    save_vectorizer(vectorizer, MODELS_DIR / 'tfidf_vectorizer.pkl')
    joblib.dump(label_encoder, MODELS_DIR / 'label_encoder.pkl')
    joblib.dump(model, MODELS_DIR / 'nb_model.pkl')
    print('Saved NB model, vectorizer, and label encoder.')


if __name__ == '__main__':
    main()
