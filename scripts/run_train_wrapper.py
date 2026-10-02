#""Runner wrapper for controlled ST-GCN training with exception capture."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from src.training.train_controlled_stgcn import main

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err_text = traceback.format_exc()
        print("CATCH EXCEPTION:", err_text)
        Path("train_error.log").write_text(err_text, encoding="utf-8")
        sys.exit(1)
