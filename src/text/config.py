from pathlib import Path
import os

IS_KAGGLE = os.path.exists('/kaggle/input')
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if IS_KAGGLE:
    DATA_PATH = Path('/kaggle/input/aidetect/AI_Human.csv')
else:
    DATA_PATH = PROJECT_ROOT / 'data' / 'AI_Human.csv'

MODELS_DIR = PROJECT_ROOT / 'models' / 'text'
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'figures').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'metrics').mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.125
MAX_FEATURES = 10000
NGRAM_RANGE = (1, 3)
MIN_DF = 5
MAX_DF = 0.9
