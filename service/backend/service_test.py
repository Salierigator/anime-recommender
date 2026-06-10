"""service_test.py — CLI test full luồng: nhập username → gợi ý + log để kiểm tra.

  python service/backend/service_test.py <username> [--top-k 20] [--no-cache]

Fetch animelist (MAL v2) + profile (Jikan) live, CACHE JSON ra service_test_output/ (rerun nhanh +
soi raw response). Chạy Recommender → in bảng top-K + dump <username>_recs.json (history, user-feats,
recs kèm feature, timing). Cold/list rỗng vẫn ra gợi ý (h_empty), chỉ cảnh báo.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mal_api
from recommend import Recommender

OUT = Path(__file__).resolve().parent / "service_test_output"


def fetch_cached(username: str, no_cache: bool):
    OUT.mkdir(exist_ok=True)
    al_p, pr_p = OUT / f"{username}_animelist.json", OUT / f"{username}_profile.json"
    if not no_cache and al_p.exists() and pr_p.exists():
        print(f"[cache] đọc {al_p.name} + {pr_p.name}")
        return json.loads(al_p.read_text()), json.loads(pr_p.read_text())
    print(f"[api] fetch animelist + profile '{username}' ...")
    animelist = mal_api.get_user_anime_list(username)
    profile = mal_api.get_user_profile(username)
    al_p.write_text(json.dumps(animelist, ensure_ascii=False))
    pr_p.write_text(json.dumps(profile, ensure_ascii=False))
    return animelist, profile


def main() -> None:
    ap = argparse.ArgumentParser(description="Test full luồng recommender cho 1 MAL username.")
    ap.add_argument("username")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--no-cache", action="store_true", help="bỏ qua cache, fetch lại từ API")
    args = ap.parse_args()

    t0 = time.time()
    rec = Recommender()
    print(f"[load] Recommender sẵn sàng ({time.time() - t0:.1f}s)")

    animelist, profile = fetch_cached(args.username, args.no_cache)
    if not animelist:
        print(f"[!] animelist rỗng/không lấy được cho '{args.username}' "
              f"(list private, user không tồn tại, hoặc MAL_CLIENT_ID sai).")

    t1 = time.time()
    recs, info = rec.recommend(animelist or [], profile, top_k=args.top_k, username=args.username)
    infer_s = time.time() - t1

    print(f"\n=== user '{args.username}' ===")
    print(f"entries={info['n_entries']}  seen(masked)={info['n_seen']}  "
          f"positive={info['n_positive']}  history={info['n_history']}"
          + ("   [!] COLD/empty history → gợi ý thiên phổ biến" if info["cold"] else ""))
    print(f"gender_id={info['gender_id']}  joined_bucket={info['joined_bucket']}  "
          f"u_n_rated={info['u_n_rated']:.0f}  u_mean={info['u_mean_score']:.2f}  "
          f"u_std={info['u_std_score']:.2f}  u_age={info['u_account_age']:.1f}")
    print(f"blend α={info['alpha']}  k_retrieve={info['k_retrieve']}  infer={infer_s:.2f}s\n")

    print(f"{'#':>3}  {'title':<45} {'mal_id':>8} {'score':>6} {'blend':>7} {'cos':>7} {'rank_pred':>10}")
    for r in recs:
        sc = f"{r['score']:.2f}" if r["score"] is not None else "-"
        print(f"{r['rank']:>3}  {(r['title'] or '?')[:45]:<45} {r['mal_id']:>8} {sc:>6} "
              f"{r['blend_score']:>7.4f} {r['cos_uv']:>7.4f} {r['ranker_pred']:>10.3f}")

    dump = OUT / f"{args.username}_recs.json"
    dump.write_text(json.dumps({"username": args.username, "info": info, "recs": recs},
                               ensure_ascii=False, indent=2))
    print(f"\n[dump] {dump}  (tổng {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
