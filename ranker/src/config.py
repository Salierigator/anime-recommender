"""config.py — hằng số chung của ranker (path + knob build/train/eval).

Path tính từ vị trí file (CWD chạy = ranker/src, import flat). Firewall: ranker chỉ ĐỌC
`artifacts/` + `cleaned-data/{details,profiles}.csv` (file nhỏ) — KHÔNG ratings.csv,
KHÔNG retriever/train-data (mọi thứ cần đã được retriever/export.py đẩy sang artifacts).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # ranker/
ARTIFACTS = ROOT.parent / "artifacts"
CLEANED = ROOT.parent / "cleaned-data"
DATA = ROOT / "data"                                   # gitignored
POOLS = DATA / "pools"                                 # eval pools (build_eval.py)
DATASETS = DATA / "datasets"                           # train/valid (build_train.py)
MODELS = DATA / "models"                               # model tải về từ Colab / train local

SEED = 42
K_POOL = 200              # pool rerank lúc serve (two-stage); train candidates = top-K_POOL
POOL_DEPTH = 500          # eval pool lưu sâu 500 → ablation K∈{200,500} không cần re-encode
TARGET_FRAC = 0.2         # tỉ lệ positive giữ làm target khi build train (mirror EVAL_QUERY_FRAC)
N_TRAIN_USERS = 100_000   # số train user sample build dataset
MIN_POS_TRAIN = 5         # train user cần ≥ positives
HIST_FEAT_CAP = 256       # prefix history (top-by-score) dùng cho FEATURE (sims/affinity);
                          # encode U luôn dùng eval_history_cap từ user_tower.pt (1024)
HIST_TOP64 = 64           # prefix history lưu cho neural ranker (DIN attention)
CHUNK = 512               # số user mỗi chunk encode/score/feature

KS = [10, 50, 100, 200]   # metric K cho selection (500 chỉ ở ablation pool depth)
ALPHAS = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]   # blend sweep (0 = cosine baseline)

GRADE_MAP = "10→4, 9→3, 7-8→2, 0/5/6→1, non-target→0"


def grade(score):
    """Relevance grade theo score (vectorized OK). Phủ TẤT CẢ positive — khớp định nghĩa
    positive của retriever (status∈{completed,watching} & score∉[1,4])."""
    import numpy as np

    s = np.asarray(score)
    return np.select([s >= 10, s >= 9, s >= 7], [4, 3, 2], default=1).astype(np.int8)
