from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_DATA_PATH = DATA_DIR / "loan_approval_dataset.csv"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_loan_data.csv"
MODEL_PATH = MODELS_DIR / "loan_model.pkl"