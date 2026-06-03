"""Config cho Two-Tower retrieval (xem plan.md / docs/TRAIN_DATA.md).

Mọi hyperparam để tune nằm trong TwoTowerConfig — chỉnh từ notebook rồi truyền vào
data/model/train. Feature vocab+dim KHÔNG hard-code ở đây: đọc từ feature_spec.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DATA = ROOT / "train-data"


def auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TwoTowerConfig:
    # --- model ---
    d: int = 128                      # dim không gian embedding chung (output 2 tower)
    genres_proj: int = 8             # Linear(22 -> genres_proj)
    themes_proj: int = 8             # Linear(53 -> themes_proj)
    mlp_hidden: List[int] = field(default_factory=lambda: [256])
    use_item_id: bool = False        # bật anime-id embedding trong ItemTower
    id_dim: int = 64                 # chiều id-embedding (bảng num_items × id_dim)
    id_dropout: float = 0.2          # prob mask id->OOV mỗi item lúc train (backoff content)

    # --- loss ---
    tau: float = 0.07                # temperature
    beta: float = 1.0                # trọng số nhánh hard-neg trong mẫu số

    # --- train ---
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 4096
    epochs: int = 1
    hist_dropout: float = 0.12       # prob bỏ toàn bộ history -> h_empty
    m_hardneg: int = 3               # số hard-neg sample mỗi anchor
    cache_refresh_steps: int = 300   # refresh item-vec cache mỗi N step
    log_every: int = 50

    # --- eval (cold-by-user) ---
    eval_ks: List[int] = field(default_factory=lambda: [10, 50, 100])
    eval_split: str = "val"
    eval_every_steps: int = 0        # eval val mỗi N step trong epoch (0 = chỉ cuối epoch)

    # --- infra ---
    device: str = field(default_factory=auto_device)
    num_workers: int = 0
    seed: int = 42
    subset: Optional[int] = None     # smoke: chỉ lấy N example train đầu (None = full)
    ckpt_dir: Path = field(default_factory=lambda: ROOT / "model" / "checkpoints")

    # --- paths (artifacts) ---
    train_data: Path = TRAIN_DATA
