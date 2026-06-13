"""Cho phép `import app...` từ bất kỳ CWD nào + ép mock mode cho test (không load model)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # service/backend/
os.environ["MOCK_MODE"] = "1"                                 # ép mock TRƯỚC khi import app.main
