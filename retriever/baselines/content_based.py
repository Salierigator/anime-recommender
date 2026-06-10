"""Content-based baseline (mean + IDF) cho stage Retrieval.

Mỗi item -> 1 content-vector từ feature đã encode (genres/themes multi-hot + one-hot
type/source/rating/demographics/start_year/episodes + multi-hot studios). Cột được IDF
down-weight (tag/feature phổ biến nhẹ đi), rồi L2-normalize từng item.

User (cold) -> profile = mean content-vector của history (support), L2-norm; score =
cosine(profile, mọi item). Mask non-candidate (logq) + item đã seen, top-K, đo
recall@K/ndcg@K y hệt protocol metrics.evaluate. Không train (chỉ tính IDF tất định).

So sánh: dùng CÙNG history như user-tower -> apples-to-apples ("Two-Tower có hơn pure
content-similarity không?"). Output -> model/baselines/content_based.txt.

Usage: venv/bin/python model/baselines/content_based.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))                 # model/ -> import flat config/data/metrics
import config as cfg_mod
import data as data_mod
from metrics import group_examples
import _eval

SPLIT = "test"


def build_content_matrix(cfg) -> np.ndarray:
    """[N, D] content-vector: IDF-weighted + L2-norm. PAD/OOV -> vector 0 (đã mask ở eval)."""
    spec = data_mod.load_feature_spec(cfg.train_data)
    items = data_mod.ItemTable(cfg.train_data)
    N = items.num_items
    fs = spec["item_features"]

    blocks = []
    # genres / themes: bỏ cột "present" (index cuối) — không phải content tag
    blocks.append(items.genres[:, : fs["genres"]["present_index"]].numpy())   # [N,21]
    blocks.append(items.themes[:, : fs["themes"]["present_index"]].numpy())   # [N,52]

    # one-hot cat/bucket: bỏ id 0 (OOV/null/none -> không là tín hiệu content)
    for col, key in [
        ("type_id", "type"), ("source_id", "source"), ("rating_id", "rating"),
        ("demographics_id", "demographics"), ("startyear_bucket", "start_year"),
        ("episodes_bucket", "episodes"),
    ]:
        v = items.cat[col].numpy()                                            # [N]
        vocab = fs[key]["vocab"]
        oh = np.zeros((N, vocab - 1), dtype=np.float32)                       # bỏ cột id 0
        m = v >= 1
        oh[m, v[m] - 1] = 1.0
        blocks.append(oh)

    # studios multi-value -> multi-hot, bỏ id 0 (empty); giữ 1 (OOV studio)
    studio_vocab = fs["studios"]["vocab"]                                     # 302
    stud = np.zeros((N, studio_vocab - 1), dtype=np.float32)
    sids = items.studios.numpy()                                             # [N,S] pad 0
    rows = np.repeat(np.arange(N), sids.shape[1])
    cols = sids.reshape(-1)
    keep = cols >= 1
    stud[rows[keep], cols[keep] - 1] = 1.0
    blocks.append(stud)

    C = np.concatenate(blocks, axis=1).astype(np.float32)                     # [N, D]

    # IDF trên item thật (idx>=2): hiếm -> nặng. df+1 smoothing, +1 để df=full vẫn >0.
    first_real = spec["special_idx"]["first_real"]
    real = C[first_real:]
    n_real = real.shape[0]
    df = (real > 0).sum(axis=0)                                              # [D]
    idf = np.log(n_real / (df + 1.0)) + 1.0
    C *= idf[None, :]

    norm = np.linalg.norm(C, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return C / norm                                                          # [N, D] L2-norm


def main():
    cfg = cfg_mod.TwoTowerConfig()
    spec = data_mod.load_feature_spec(cfg.train_data)
    logq = data_mod.load_logq(cfg.train_data).to(cfg.device)
    users = data_mod.UserTable(cfg.train_data, spec["k_history"], spec["hard_neg_cap"])

    Cn = torch.from_numpy(build_content_matrix(cfg)).to(cfg.device)          # [N, D]

    def score_fn(u, hist):
        v = Cn[hist]                                                         # [E, Hk, D]
        mask = (hist != 0).unsqueeze(-1).float()                            # [E, Hk, 1]
        prof = (v * mask).sum(1) / mask.sum(1).clamp(min=1.0)                # [E, D] mean history
        prof = torch.nn.functional.normalize(prof, dim=1)
        return prof @ Cn.t()                                                # [E, N] cosine

    smoke = "--smoke" in sys.argv
    ds = data_mod.ExamplesDataset(cfg.train_data, SPLIT, subset=4000 if smoke else None)
    queries = group_examples(ds.user_idx, ds.anime_idx)
    out, n, n_cand = _eval.rank_eval(cfg, users, queries, logq, score_fn, cfg.eval_ks)

    header = [
        f"# Content-based baseline (mean + IDF) — split={SPLIT}{' [SMOKE]' if smoke else ''}",
        f"# generated {datetime.now().isoformat(timespec='seconds')}  device={cfg.device}",
        f"# users evaluated: {n:,}   candidates (finite logq): {n_cand:,}   content_dim: {Cn.shape[1]}",
    ]
    out_name = "content_based_smoke.txt" if smoke else "content_based.txt"
    _eval.write_result(HERE / out_name, header, out, cfg.eval_ks)


if __name__ == "__main__":
    main()
