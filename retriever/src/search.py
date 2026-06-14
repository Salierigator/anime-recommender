"""Hyperparameter search driver cho retriever (random | grid) — importable, không phụ thuộc torch.

Triết lý (xem docs/EXPERIMENTS.md): random search > grid khi >3-4 chiều (Bergstra & Bengio).
Quy trình 2 tầng: COARSE trên subset (train_user_frac) để xếp hạng config rẻ -> CONFIRM
top-K trên full data. Mỗi config -> run_name TẤT ĐỊNH (encode knob khác default, sorted)
=> tự dedupe + RESUME khi Colab rớt session (tận dụng dedup (version,run_name) của leaderboard).

Notebook chỉ: định nghĩa `space`, gọi `run_search(run_experiment, iter_configs(space, ...))`.
run_experiment (cell 5) đã log best.pt/history/row/config lên Drive + rebuild runs.csv.
"""
from __future__ import annotations

import itertools
import random
from typing import Callable, Dict, Iterable, List, Tuple

# Viết tắt knob -> run_name gọn (fallback = tên knob nếu không có trong map).
_ABBR = {
    "d": "d", "use_item_id": "id", "id_dim": "iddim", "id_dropout": "iddrop",
    "history_source": "hist", "train_hist_len": "hl", "history_pool": "hpool",
    "score_pool": "spool", "tau": "tau", "logq_alpha": "alpha", "m_hardneg": "mhn",
    "lr": "lr", "weight_decay": "wd", "optimizer": "opt", "cosine_lr": "cos",
    "batch_size": "bs", "epochs": "ep", "hist_dropout": "hdrop",
    "max_examples_per_user": "cap", "train_user_frac": "uf",
    "use_synopsis": "syn", "synopsis_dim": "sd", "synopsis_proj_hidden": "sph",
    "synopsis_normalize": "snorm",
}

# Knob phụ thuộc: tắt cha -> bỏ con (tránh chạy lại model GIỐNG HỆT với tên khác nhau).
_DEPENDENT = {
    "use_synopsis": ["synopsis_dim", "synopsis_proj_hidden", "synopsis_normalize"],
    "use_item_id": ["id_dim", "id_dropout"],
}


def canonicalize(overrides: Dict) -> Dict:
    """Bỏ knob con khi cha tắt -> 2 config chỉ khác knob vô nghĩa sẽ trùng run_name + dedupe."""
    ov = dict(overrides)
    for parent, children in _DEPENDENT.items():
        if not ov.get(parent, False):
            for c in children:
                ov.pop(c, None)
    return ov


def _vstr(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:g}".replace(".", "").replace("-", "m")   # 0.5 -> 05, 1e-5 -> 1em05
    if isinstance(v, (list, tuple)):
        return "x".join(str(x) for x in v) if v else "0"
    return str(v).replace(".", "")


def deterministic_run_name(overrides: Dict, prefix: str = "v5") -> str:
    """Tên ổn định từ knob khác default (sorted theo key -> không phụ thuộc thứ tự dict)."""
    ov = canonicalize(overrides)
    parts = [f"{_ABBR.get(k, k)}{_vstr(v)}" for k, v in sorted(ov.items())]
    return f"{prefix}_" + "_".join(parts) if parts else f"{prefix}_base"


def iter_configs(space: Dict[str, List], method: str = "random", n: int = 20,
                 seed: int = 0, fixed: Dict = None, prefix: str = "v5"
                 ) -> Iterable[Tuple[str, Dict]]:
    """Sinh (run_name, overrides) từ `space` {knob: [values]}.
    `fixed`: knob áp cho MỌI config (vd {'train_user_frac':0.15,'epochs':2} cho tầng coarse).
    grid = itertools.product (vét cạn); random = sample n bộ (seeded). Dedupe theo run_name
    (đã canonicalize) -> không lặp model giống hệt."""
    fixed = fixed or {}
    seen = set()
    if method == "grid":
        keys = list(space)
        for combo in itertools.product(*(space[k] for k in keys)):
            ov = canonicalize({**fixed, **dict(zip(keys, combo))})
            name = deterministic_run_name(ov, prefix)
            if name in seen:
                continue
            seen.add(name)
            yield name, ov
    elif method == "random":
        rng = random.Random(seed)
        tries, cap = 0, max(n * 50, 100)
        while len(seen) < n and tries < cap:
            tries += 1
            ov = canonicalize({**fixed, **{k: rng.choice(v) for k, v in space.items()}})
            name = deterministic_run_name(ov, prefix)
            if name in seen:
                continue
            seen.add(name)
            yield name, ov
    else:
        raise ValueError(f"method '{method}' không hỗ trợ (grid|random)")


def run_search(run_experiment_cb: Callable, configs: Iterable[Tuple[str, Dict]],
               exists_fn: Callable[[str], bool] = None, dry_run: bool = False) -> List[Tuple]:
    """Lặp configs -> bỏ run đã có (exists_fn, resume) -> gọi run_experiment_cb(run_name, **overrides).
    dry_run: chỉ in tên (kiểm tra space local, không train). Trả [(run_name, overrides, out)]."""
    results = []
    for run_name, overrides in configs:
        if exists_fn is not None and exists_fn(run_name):
            print(f"  skip {run_name} (đã có trên Drive)")
            continue
        print(f"→ {run_name}  {overrides}")
        out = None if dry_run else run_experiment_cb(run_name, **overrides)
        results.append((run_name, overrides, out))
    return results


def default_space() -> Dict[str, List]:
    """Không gian search VÍ DỤ (sửa trong notebook). Là literal -> trích thẳng vào báo cáo.
    use_item_id giữ True làm control ở `fixed`; ở đây để các đòn bẩy nội dung + reg + synopsis."""
    return {
        "train_hist_len": [32, 64, 96],
        "history_source": ["cache", "embed"],
        "id_dropout": [0.1, 0.15, 0.2],
        "logq_alpha": [1.0, 0.75],
        "weight_decay": [0.0, 1e-5],
        "optimizer": ["adam", "adamw"],
        "use_synopsis": [False, True],
        "synopsis_dim": [48, 64],
        "synopsis_proj_hidden": [[], [128]],
    }
