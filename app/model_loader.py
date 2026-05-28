"""
Model weight loader.
If the weight file exists locally in models/, use it. Otherwise download it
from the Hugging Face Hub (for Streamlit Cloud deployment).
"""
from pathlib import Path

HF_REPO_ID = "Shafiya1234/medical-ai-dashboard-models"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def get_model_path(filename: str) -> str:
    local_path = MODELS_DIR / filename
    if local_path.exists():
        return str(local_path)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
