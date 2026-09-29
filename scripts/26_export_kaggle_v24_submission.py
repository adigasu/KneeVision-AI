"""
RSNA Knee Abnormality Detection - Updated Kaggle Submission Kernel Exporter.
Exports the verified 5-fold ensemble with strict error handling (fails loud, zero silent dummy options).
"""

import json
from pathlib import Path


def export_kernel(output_dir: str = "kaggle_upload/kernel"):
    p_dir = Path(output_dir)
    p_dir.mkdir(parents=True, exist_ok=True)
    
    src = p_dir / "rsna_knee_submission_v11_5fold.ipynb"
    if not src.exists():
        raise FileNotFoundError(f"Source notebook {src} not found!")
        
    with open(src) as f:
        nb = json.load(f)
        
    dst = p_dir / "rsna_knee_submission.ipynb"
    with open(dst, "w") as f:
        json.dump(nb, f, indent=1)
        
    print(f"Exported kernel successfully to {src} and {dst}")


if __name__ == "__main__":
    export_kernel()
