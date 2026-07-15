"""Config cho Two-Tower retrieval (xem thesis-final:docs/TRAIN_DATA.md).

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
    score_pool: str = "none"         # weighted history pooling theo điểm: 'none'|'linear'|'learned'
    history_pool: str = "mean"       # gộp history: 'mean' (±score_pool) | 'attn' (learned-query attention, bỏ qua score_pool)
    history_source: str = "cache"    # nguồn vec phía history: 'cache' (item-vec detach) | 'embed' (bảng Embedding trainable riêng)

    # --- synopsis (content text-emb item-side; frozen artifact + projection trainable) ---
    use_synopsis: bool = False       # bật nhánh synopsis embedding trong ItemTower
    synopsis_dim: int = 48           # chiều sau khi chiếu, concat vào content path (~ngang khối genres/themes/studios)
    synopsis_proj_hidden: List[int] = field(default_factory=list)  # hidden MLP chiếu raw->dim; [] = Linear thuần, [128] nếu underfit
    synopsis_normalize: str = "none" # vec frozen trước chiếu: 'none' (artifact 07 đã L2) | 'l2' (re-norm khi swap artifact CHƯA norm). standardize CHƯA implement (model.py)
    synopsis_emb_file: str = "synopsis_emb.npy"            # artifact frozen [num_items, raw_dim] (swap bge/gte qua đây)
    synopsis_low_info_file: str = "synopsis_low_info.npy"  # bool [num_items]: NaN/<50 ký tự/placeholder -> dùng vec no_synopsis học được

    # --- loss ---
    tau: float = 0.07                # temperature
    beta: float = 1.0                # trọng số nhánh hard-neg trong mẫu số
    logq_alpha: float = 1.0          # hệ số logQ correction (1.0 = full, 0 = tắt)

    # --- train ---
    lr: float = 1e-3
    optimizer: str = "adam"          # 'adam' | 'adamw' (AdamW decouple weight_decay — chỉ khác khi weight_decay>0)
    cosine_lr: bool = False          # True = cosine-anneal LR lr->0 suốt train (thay LR hằng số)
    weight_decay: float = 0.0
    batch_size: int = 4096
    epochs: int = 1
    hist_dropout: float = 0.12       # prob bỏ toàn bộ history -> h_empty
    m_hardneg: int = 3               # số hard-neg sample mỗi anchor
    train_hist_len: int = 32         # số item history sample mỗi anchor (từ full list; augmentation)
    max_examples_per_user: Optional[int] = None  # cap example/user/epoch (resample mỗi epoch; None = off)
    cache_refresh_steps: int = 300   # refresh item-vec cache mỗi N step
    log_every: int = 50

    # --- eval (cold-by-user, protocol v2) ---
    eval_ks: List[int] = field(default_factory=lambda: [10, 50, 100, 200, 500])
    headline_k: int = 200            # metric chọn checkpoint = recall@headline_k (warm val)
    eval_history_cap: int = 1024     # prefix history (đã sort score desc) dùng lúc eval
    eval_split: str = "val"
    eval_every_steps: int = 0        # eval val mỗi N step trong epoch (0 = chỉ cuối epoch)
    eval_cold_in_loop: bool = False  # True: do_eval cũng chấm cold val (full-catalog) -> history (chỉ bật cho run final; KHÔNG ảnh hưởng chọn checkpoint)
    early_stop_patience: Optional[int] = None  # early-stop (None = TẮT)
    early_stop_min_delta: float = 0.0          # ngưỡng tối thiểu để reset patience (vd 0.001 cho recall@200)

    # --- infra ---
    device: str = field(default_factory=auto_device)
    num_workers: int = 0
    seed: int = 42
    subset: Optional[int] = None     # smoke: chỉ lấy N example train đầu (None = full)
    train_user_frac: Optional[float] = None  # HP-search: chỉ giữ ngẫu nhiên frac USER cho split=train (giữ full catalog+logQ+eval). None = full
    subset_seed: int = 12345         # seed lọc user của train_user_frac (tất định, ghi vào row log)
    ckpt_dir: Path = field(default_factory=lambda: ROOT / "checkpoints")

    # --- paths (artifacts) ---
    train_data: Path = TRAIN_DATA
