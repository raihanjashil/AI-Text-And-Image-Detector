import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .config import DATA_PATH, RANDOM_STATE, TEST_SIZE, VAL_SIZE
from .preprocess import preprocess_text


def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        if 'generated' in df.columns:
            df['label'] = df['generated'].apply(lambda x: 'AI' if x == 1.0 else 'Human')
        else:
            raise ValueError('Dataset must contain either label or generated column')
    df = df[['text', 'label']].dropna().copy()
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    return df


def split_data(df):
    X_text = df['cleaned_text']
    y = df['label']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_text, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )
    return X_train_text, X_val_text, X_test_text, y_train, y_val, y_test, label_encoder
