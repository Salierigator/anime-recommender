"""encode_user.py — đặt 1 user lên bản đồ có sẵn (CLI). "You are here" + top-K gợi ý.

Stack ARTIFACTS (firewall-clean): tái dùng encoder của ranker (load_user_encoder + encode_users)
— KHÔNG load best.pt. User-tower không có user-id nên username lạ vẫn encode được, miễn có history.

  # user trong dataset (history từ artifacts/users_history.parquet qua username):
  python map/encode_user.py <username>
  # user tổng hợp từ list mal_id (mỗi dòng 1 id):
  python map/encode_user.py --mal-ids ids.txt --name me

Ghi outputs/overlay_user_<name>.parquet (kind∈{user,neighbor}) -> viz.py --overlay.

Import order BẮT BUỘC (xem service/backend/app/ml/recommender.py): torch trước; pool (kéo
features) TRƯỚC user_encode (user_encode chèn retriever/src vào sys.path → 'config' đổi nghĩa).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _common as C

import torch  # noqa: E402  (torch TRƯỚC mọi thứ — segfault OpenMP trên mac)

sys.path.insert(0, str(C.ROOT / "ranker" / "src"))
from pool import UsersHistory, encode_users        # noqa: E402  (pool kéo features — TRƯỚC user_encode)
from user_encode import load_user_encoder          # noqa: E402


def history_from_dataset(username: str, uh: UsersHistory):
    split = pd.read_parquet(C.ARTIFACTS / "user_split.parquet")
    row = split[split["username"] == username]
    if row.empty:
        raise SystemExit(f"username '{username}' không có trong dataset. Dùng --mal-ids cho user lạ.")
    u = int(row["user_idx"].iloc[0])
    ids, scores = uh.history(u)
    return (ids.astype(np.int64), scores.astype(np.int64),
            int(uh.gender_id[u]), int(uh.joined_bucket[u]))


def history_from_mal_ids(path: Path, meta: dict):
    """List mal_id -> anime_idx (qua item_index), score=0, gender=0, joined=cohort mới nhất."""
    wanted = [int(x) for x in Path(path).read_text().split() if x.strip()]
    idx = pd.read_parquet(C.ARTIFACTS / "item_index.parquet")
    mal2idx = dict(zip(idx["mal_id"].to_numpy(), idx["anime_idx"].to_numpy()))
    ids = np.array([mal2idx[m] for m in wanted if m in mal2idx], dtype=np.int64)
    if len(ids) == 0:
        raise SystemExit("Không map được mal_id nào sang anime_idx (sai id?).")
    print(f"map {len(ids)}/{len(wanted)} mal_id -> anime_idx")
    newest = len(meta["user_features"]["joined"]["bins"]) - 2
    return ids, np.zeros(len(ids), np.int64), 0, int(newest)


def main() -> None:
    ap = argparse.ArgumentParser(description="Đặt user point lên map")
    ap.add_argument("username", nargs="?", help="username trong dataset")
    ap.add_argument("--mal-ids", type=Path, help="file list mal_id (user lạ)")
    ap.add_argument("--name", help="tên file overlay (mặc định = username / 'malids')")
    ap.add_argument("--method", default="pumap2d")
    ap.add_argument("--top-k", type=int, default=15, help="số anime gợi ý gần U nhất để highlight")
    args = ap.parse_args()

    enc, meta = load_user_encoder("cpu")
    cap = meta.get("eval_history_cap", 1024)

    if args.mal_ids:
        ids, scores, gender, joined = history_from_mal_ids(args.mal_ids, meta)
        name = args.name or "malids"
    elif args.username:
        ids, scores, gender, joined = history_from_dataset(args.username, UsersHistory())
        name = args.name or args.username
    else:
        raise SystemExit("Cần <username> hoặc --mal-ids.")

    U = encode_users(enc, [ids], [scores],
                     np.array([gender], np.int64), np.array([joined], np.int64), cap)  # [1,d]
    Un = U.numpy()

    coords = C.load_coords(args.method).set_index("anime_idx")
    base = pd.read_parquet(C.OUTPUTS / "base.parquet").set_index("anime_idx")

    # top-K gợi ý: cosine full catalog NHƯNG chỉ giữ item CÓ trên map (coords = base SFW real, đã
    # loại hentai + PAD/OOV) → neighbor nhất quán với map train-không-hentai. Khớp serving: U mã hoá
    # từ FULL history, chỉ lọc nsfw ở ĐẦU RA. Loại thêm item đã xem (seen).
    cos = (U @ enc.item_cache.t()).numpy()[0]
    on_map = np.zeros(len(cos), dtype=bool)
    on_map[coords.index.to_numpy()] = True
    cos[~on_map] = -np.inf
    cos[ids] = -np.inf
    nbr = np.argsort(-cos)[: args.top_k].astype(np.int64)

    reducer = C.load_reducer(args.method)
    user_xy = C.transform_to_coords(args.method, reducer, Un)[0]

    rows = [{"kind": "user", "label": name, "anime_idx": -1,
             "x": float(user_xy[0]), "y": float(user_xy[1])}]
    for a in nbr:
        c = coords.loc[a]
        title = base.loc[a, "title"] if a in base.index else str(a)
        rows.append({"kind": "neighbor", "label": str(title), "anime_idx": int(a),
                     "x": float(c["x"]), "y": float(c["y"])})

    out = C.OUTPUTS / f"overlay_user_{name}.parquet"
    pd.DataFrame(rows).to_parquet(out)
    print(f"-> {out}  (user point + {len(rows) - 1} neighbor)")
    print(f"history dùng: {len(ids)} item (cap {cap}); top-{args.top_k} gợi ý anime_idx: {nbr.tolist()}")


if __name__ == "__main__":
    main()
