"""Crawl user animelists via MAL API v2 (official, 1000 entries/page, ~3 req/s).

Picks pending usernames from the state DB (fed by collect_usernames.py) and
appends flat rows to data/raw/ratings.csv. Timestamps (updated_at, start/finish
date) are kept when MAL returns them — often missing on old entries, cleaning
decides what to do with them.

Private list -> HTTP 403, deleted account -> 404: both are recorded and skipped
forever. Exhausted retries -> status 'error' (re-queue with --retry-errors).
Crash between CSV append and the status mark re-crawls that one user on resume
-> duplicate rows for them, deduped in cleaning.

    venv/bin/python data/crawler/crawl_ratings.py               # all pending users
    venv/bin/python data/crawler/crawl_ratings.py --limit 5     # smoke test
    venv/bin/python data/crawler/crawl_ratings.py --retry-errors
"""
from __future__ import annotations

import argparse
import csv

import common

OUT = common.RAW / "ratings.csv"
FIELDS = ("list_status{status,score,num_episodes_watched,is_rewatching,"
          "updated_at,start_date,finish_date}")
COLS = ["username", "anime_id", "status", "score", "num_episodes_watched",
        "is_rewatching", "updated_at", "start_date", "finish_date"]


def fetch_list(client, username):
    """All pages for one user -> (rows, status_str)."""
    url = f"{common.MAL_BASE}/users/{username}/animelist"
    params = {"limit": 1000, "fields": FIELDS, "nsfw": "true"}
    rows = []
    while url:
        r = client.get(url, params=params)
        if r is None:
            return None, "error"
        if r.status_code in (403, 404):
            return None, f"http_{r.status_code}"
        if r.status_code != 200:
            return None, "error"
        d = r.json()
        for e in d["data"]:
            ls = e.get("list_status", {})
            rows.append([
                username, e["node"]["id"], ls.get("status", ""), ls.get("score", 0),
                ls.get("num_episodes_watched", 0), int(bool(ls.get("is_rewatching"))),
                ls.get("updated_at", ""), ls.get("start_date", ""), ls.get("finish_date", ""),
            ])
        url = d.get("paging", {}).get("next")
        params = None  # paging.next already carries the query string
    return rows, ("ok" if rows else "empty")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="crawl at most N users this run")
    ap.add_argument("--rps", type=float, default=3.0, help="max requests/second (measured safe: 4.5)")
    ap.add_argument("--retry-errors", action="store_true", help="re-queue users marked 'error'")
    args = ap.parse_args()

    common.install_signals()
    conn = common.db()
    client = common.Client(min_interval=1.0 / args.rps,
                           headers={"X-MAL-CLIENT-ID": common.load_mal_client_id()})

    if args.retry_errors:
        n = conn.execute("UPDATE users SET ratings_status=NULL WHERE ratings_status='error'").rowcount
        conn.commit()
        print(f"[i] re-queued {n} error users")

    q = "SELECT username FROM users WHERE ratings_status IS NULL ORDER BY discovered_at"
    if args.limit:
        q += f" LIMIT {args.limit}"
    pending = [r[0] for r in conn.execute(q).fetchall()]
    print(f"[i] {len(pending)} users to crawl -> {OUT}")

    new_file = not OUT.exists()
    n_users = n_rows = 0
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(COLS)
        for username in pending:
            if common.STOP:
                break
            rows, status = fetch_list(client, username)
            if rows:
                w.writerows(rows)
                f.flush()
                n_rows += len(rows)
            conn.execute(
                "UPDATE users SET ratings_status=?, ratings_at=?, ratings_n=? WHERE username=?",
                (status, common.now_iso(), len(rows) if rows else 0, username))
            conn.commit()
            n_users += 1
            tag = f"{len(rows)} entries" if rows else status
            print(f"[+] {n_users}/{len(pending)} {username}: {tag}")

    print(f"[done] {n_users} users, {n_rows} rows appended this run")
    for st, n in conn.execute(
            "SELECT ratings_status, COUNT(*) FROM users GROUP BY ratings_status"):
        print(f"    {st or 'pending'}: {n}")
    conn.close()


if __name__ == "__main__":
    main()
