import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))            # lib chung (config/features/metrics/pool)
sys.path.insert(0, str(_ROOT / "data_prep"))      # build_train (test_no_leak)
