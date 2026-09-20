from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
OUTPUT_DIR = ROOT_DIR / "outputs"
UPLOAD_DIR = ROOT_DIR / "uploads"
WORKFLOW_DIR = ROOT_DIR / "workflows"

for _dir in (OUTPUT_DIR, UPLOAD_DIR, WORKFLOW_DIR):
    _dir.mkdir(exist_ok=True)
