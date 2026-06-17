"""Content-based baseline (mean + IDF) cho stage Retrieval.

Mỗi item -> 1 content-vector từ feature đã encode (genres/themes multi-hot + one-hot
type/source/rating/demographics/start_year/episodes + multi-hot studios). Cột được IDF
down-weight (tag/feature phổ biến nhẹ đi), rồi L2-normalize từng item.

User (cold) -> profile = mean content-vector của history (support), L2-norm; score =
cosine(profile, mọi item). Protocol v2: mask seen−query, history prefix cap. Không
train (chỉ IDF tất định) -> cold-capable (content vector của H tồn tại) — đây là
comparator chính của model trên cold slice.

So sánh: dùng CÙNG history như user-tower -> apples-to-apples ("Two-Tower có hơn pure
content-similarity không?"). Output -> retriever/baselines/content_based.txt.

Usage: venv/bin/python retriever/baselines/content_based.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))                 # import flat config/data/metrics
import data as data_mod
import _eval

SPLIT = "test"
BATCH = 64            # gather [E, W<=1024, 423] — batch nhỏ để không phình memory


def build_content_matrix(cfg, use_idf: bool = True) -> np.ndarray:
    """[N, D] content-vector: (IDF-weighted nếu use_idf) + L2-norm. PAD/OOV -> vector 0 (đã mask ở eval)."""
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
    if use_idf:
        first_real = spec["special_idx"]["first_real"]
        real = C[first_real:]
        n_real = real.shape[0]
        df = (real > 0).sum(axis=0)                                          # [D]
        idf = np.log(n_real / (df + 1.0)) + 1.0
        C *= idf[None, :]

    norm = np.linalg.norm(C, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return C / norm                                                          # [N, D] L2-norm


def make_score_fn(Cn):
    def score_fn(u, hist):
        v = Cn[hist]                                                         # [E, W, D]
        mask = (hist != 0).unsqueeze(-1).float()                            # [E, W, 1]
        prof = (v * mask).sum(1) / mask.sum(1).clamp(min=1.0)                # [E, D] mean history
        prof = torch.nn.functional.normalize(prof, dim=1)
        return prof @ Cn.t()                                                # [E, N] cosine
    return score_fn


def main():
    # --- ablation IDF on/off trên VAL (chọn theo recall@headline_k) ---
    cfg, _, logq, users, qv, mv, _, _, _, _ = _eval.setup("val")
    HK = cfg.headline_k
    sweep = {}
    mats = {}
    for use_idf in (True, False):
        mats[use_idf] = torch.from_numpy(build_content_matrix(cfg, use_idf)).to(cfg.device)
        out_v, _, _ = _eval.rank_eval(cfg, users, qv, logq, make_score_fn(mats[use_idf]),
                                      cfg.eval_ks, mv, batch=BATCH)
        sweep[use_idf] = out_v[f"recall@{HK}"]
    best_idf = max(sweep, key=sweep.get)

    # --- report TEST với cấu hình thắng (+ liked); logq split-independent, dùng lại ---
    _, _, _, usersT, q_warm, m_warm, q_cold, m_cold, qs_warm, qs_cold = _eval.setup(SPLIT)
    Cn = mats[best_idf]
    score_fn = make_score_fn(Cn)
    out_w, n_w, n_cand = _eval.rank_eval(cfg, usersT, q_warm, logq, score_fn, cfg.eval_ks,
                                         m_warm, batch=BATCH, query_scores=qs_warm)
    out_c, n_c, _ = _eval.rank_eval(cfg, usersT, q_cold, logq, score_fn, cfg.eval_ks,
                                    m_cold, batch=BATCH, pooled=True, query_scores=qs_cold)

    lines = _eval.header(f"Content-based baseline (mean{', +IDF' if best_idf else ', no-IDF'})",
                         cfg, SPLIT, n_cand,
                         extra=f"content_dim={Cn.shape[1]}, use_idf={best_idf} (selected on val)")
    lines += [f"## val sweep (recall@{HK}): "
              + "  ".join(f"idf={ui}: {sweep[ui]:.6f}" for ui in (True, False)), ""]
    lines += _eval.section("warm (test)", out_w, cfg.eval_ks, n_w)
    lines += _eval.section("cold (test_cold, full-catalog)", out_c, cfg.eval_ks, n_c, pooled=True)
    _eval.write_result(HERE / "content_based.txt", lines)


if __name__ == "__main__":
    main()
