import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from .config import MODELS_DIR, RESULTS_DIR
from .data import load_dataset, split_data
from .features import load_vectorizer


def evaluate(model_name='nb_model.pkl'):
    df = load_dataset()
    X_train_text, X_val_text, X_test_text, y_train, y_val, y_test, label_encoder = split_data(df)

    vectorizer = load_vectorizer(MODELS_DIR / 'tfidf_vectorizer.pkl')
    model = joblib.load(MODELS_DIR / model_name)

    X_test = vectorizer.transform(X_test_text)
    preds = model.predict(X_test)

    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds)),
        'recall': float(recall_score(y_test, preds)),
        'f1': float(f1_score(y_test, preds)),
        'model': model_name,
    }
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.colorbar()
    plt.savefig(RESULTS_DIR / 'figures' / f'{model_name}_confusion_matrix.png', bbox_inches='tight')
    plt.close()

    with open(RESULTS_DIR / 'metrics' / f'{model_name}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == '__main__':
    evaluate()
