"""recommend.py — CLI inference end-to-end: username → 2 section gợi ý (firewall-clean, no UI).

    venv/bin/python service/backend/recommend.py <username> [--top-k 20] [--cold-k 10]
                                                 [--live] [--no-cache] [--dump]
    venv/bin/python service/backend/recommend.py --mal-ids service/backend/fixtures/dummy_mal_ids.txt

Đây là CLI entry "mỏng" — logic serve nằm ở app/ml/recommender.py (class Recommender), client MAL
ở app/clients/mal_api.py. App API (FastAPI) cũng tái dùng Recommender qua app/services/real_service.py.

Nguồn history (theo thứ tự):
  1. username có trong dataset → artifacts/users_history.parquet (offline, không cần API;
     user val/test: history = support — gợi ý có thể trúng query held-out, đó là điểm tốt).
  2. --live hoặc username ngoài dataset → MAL v2 animelist + Jikan profile (app/clients/mal_api.py,
     cần MAL_CLIENT_ID trong service/.env); cache JSON ở backend/cache/.
  3. --mal-ids <file>: list mal_id (1 id/dòng) giả làm list completed chưa chấm điểm.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# CWD chạy = repo root: thêm backend/ vào path để `import app...` (script dir = backend cũng đủ).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.ml.recommender import Recommender                # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"


def fetch_live(username: str, no_cache: bool):
    """MAL v2 animelist + Jikan profile, cache JSON ở backend/cache/."""
    from app.clients import mal_api
    CACHE.mkdir(exist_ok=True)
    al_p, pr_p = CACHE / f"{username}_animelist.json", CACHE / f"{username}_profile.json"
    if not no_cache and al_p.exists() and pr_p.exists():
        print(f"[cache] {al_p.name} + {pr_p.name}")
        return json.loads(al_p.read_text()), json.loads(pr_p.read_text())
    print(f"[api] fetch animelist + profile '{username}' ...")
    animelist = mal_api.get_user_anime_list(username)
    profile = mal_api.get_user_profile(username)
    al_p.write_text(json.dumps(animelist, ensure_ascii=False))
    pr_p.write_text(json.dumps(profile or {}, ensure_ascii=False))
    return animelist, profile


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n── {title} " + "─" * max(0, 76 - len(title)))
    print(f"{'#':>3}  {'title':<46} {'year':>5} {'type':<7} {'MAL':>5}"
          f"{'pred':>8} {'cos':>7}")
    for i, r in enumerate(rows, 1):
        print(f"{i:>3}  {(r['title'] or '?')[:46]:<46} {r['year'] or '-':>5} "
              f"{r['type']:<7} {r['mal_score'] if r['mal_score'] is not None else '-':>5}"
              f"{r.get('pred', ''):>8} {r.get('cos', ''):>7}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Gợi ý anime cho 1 user (CLI).")
    ap.add_argument("username", nargs="?", help="MAL username (dataset hoặc live)")
    ap.add_argument("--mal-ids", type=Path, help="file mal_id (1 id/dòng) thay cho username")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--cold-k", type=int, default=10)
    ap.add_argument("--anchor", type=int, help="mal_id: tìm anime giống X (giữ cá nhân hoá user)")
    ap.add_argument("--live", action="store_true", help="ép fetch API dù username có trong dataset")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--dump", action="store_true", help="dump JSON ra backend/cache/")
    args = ap.parse_args()
    if not args.username and not args.mal_ids:
        ap.error("cần username hoặc --mal-ids")

    t0 = time.time()
    rec = Recommender()
    print(f"[load] Recommender sẵn sàng ({time.time() - t0:.1f}s)")

    if args.mal_ids:
        ids = [int(l) for l in args.mal_ids.read_text().split() if l.strip()]
        user, name = rec.user_from_mal_ids(ids), args.mal_ids.stem
        print(f"[user] {len(ids)} mal_id → {len(user['hist_idx'])} item trong corpus")
    else:
        name = args.username
        user = None if args.live else rec.user_from_dataset(name)
        if user is None:
            animelist, profile = fetch_live(name, args.no_cache)
            if not animelist:
                print("[!] animelist rỗng (list private / user không tồn tại / thiếu "
                      "MAL_CLIENT_ID trong service/.env)")
            user = rec.user_from_animelist(animelist or [], profile, name)

    n_hist = len(user["hist_idx"])
    print(f"[user] '{name}' source={user['source']} split={user['split']} "
          f"history={n_hist} seen(masked)={len(user['seen'])}"
          + ("   [!] COLD START → gợi ý thiên phổ biến" if n_hist == 0 else ""))
    top = [rec._row(int(a)) | {"score": int(s)}
           for a, s in zip(user["hist_idx"][:5], user["hist_score"][:5])]
    for r in top:
        print(f"        hist: {r['title'][:40]:<40} (score {r['score']})")

    t1 = time.time()
    try:
        out = rec.recommend(user, top_k=args.top_k, cold_k=args.cold_k,
                            anchor_mal_id=args.anchor)
    except KeyError:
        ap.error(f"--anchor {args.anchor} không có trong corpus")
    print(f"[infer] {time.time() - t1:.2f}s  (α={rec.alpha}, K={rec.k_retrieve})"
          + (f"  [anchor mal_id={args.anchor}]" if args.anchor else ""))
    main_title = (f"Anime giống mal_id {args.anchor} (rerank LightGBM, top {args.top_k})"
                  if args.anchor else f"Gợi ý cho bạn (rerank LightGBM, top {args.top_k})")
    print_table(main_title, out["main"])
    print_table(f"Anime mới cho bạn (cold theo retriever, top {args.cold_k})", out["cold"])

    if args.dump:
        CACHE.mkdir(exist_ok=True)
        p = CACHE / f"{name}_recs.json"
        p.write_text(json.dumps({"user": name, **out}, ensure_ascii=False, indent=2,
                                default=str))
        print(f"\n[dump] {p}")


if __name__ == "__main__":
    main()
