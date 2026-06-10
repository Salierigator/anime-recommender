"""
Test flow for the MAL data fetchers (no model / UI yet).

  Anime metadata: runs over service/backend/dummy_mal_ids.txt
  User data:      runs only if a username is given
                    python test_api.py --user <MAL_username>
                    (or set env MAL_TEST_USER=<name>)

Full JSON responses are dumped under service/backend/api_test_output/ for inspection;
a compact summary is printed to the console.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mal_api

HERE = Path(__file__).resolve().parent
IDS_FILE = HERE / "dummy_mal_ids.txt"
OUT_DIR = HERE / "api_test_output"


def _dump(name, obj):
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def _read_ids():
    return [
        int(line)
        for line in IDS_FILE.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_anime():
    ids = _read_ids()
    print(f"\n=== ANIME METADATA — {len(ids)} dummy ids ===")
    ok = 0
    for aid in ids:
        meta = mal_api.get_anime_metadata(aid)
        if not meta:
            continue
        ok += 1
        _dump(f"anime_{aid}.json", meta)
        genres = ", ".join(g["name"] for g in meta.get("genres", [])) or "-"
        s = meta.get("start_season") or {}
        season = f"{s.get('season', '')} {s.get('year', '')}".strip() or "-"
        print(
            f"  [{aid}] {meta.get('title', '?')}\n"
            f"        type={meta.get('media_type')} eps={meta.get('num_episodes')} "
            f"mean={meta.get('mean')} rank={meta.get('rank')} pop={meta.get('popularity')}\n"
            f"        season={season} | genres: {genres}"
        )
    print(f"--- anime ok: {ok}/{len(ids)} (json in {OUT_DIR.name}/) ---")


def test_user(username):
    print(f"\n=== USER '{username}' ===")

    profile = mal_api.get_user_profile(username)
    if profile:
        _dump(f"user_{username}_profile.json", profile)
        favs = (profile.get("favorites") or {}).get("anime", [])
        stats = (profile.get("statistics") or {}).get("anime", {})
        print(
            f"  demographics: gender={profile.get('gender')} "
            f"birthday={profile.get('birthday')} location={profile.get('location')} "
            f"joined={profile.get('joined')}"
        )
        print(
            f"  stats: mean_score={stats.get('mean_score')} "
            f"completed={stats.get('completed')} watching={stats.get('watching')} "
            f"total_entries={stats.get('total_entries')}"
        )
        print(f"  favorite anime: {len(favs)}")

    history = mal_api.get_user_anime_list(username)
    _dump(f"user_{username}_animelist.json", history)
    scored = sum(1 for e in history if (e.get("list_status") or {}).get("score", 0) > 0)
    print(f"  animelist entries: {len(history)} (scored: {scored})")
    print(f"--- user done (json in {OUT_DIR.name}/) ---")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=os.environ.get("MAL_TEST_USER"))
    ap.add_argument("--skip-anime", action="store_true")
    args = ap.parse_args()

    if not args.skip_anime:
        test_anime()

    if args.user:
        test_user(args.user)
    else:
        print("\n[i] No username given — skipping user test.")
        print("    Run later:  python test_api.py --user <MAL_username>")


if __name__ == "__main__":
    main()
