"""Collect active usernames by polling MAL's "Recently Online Users" page.

https://myanimelist.net/users.php shows ~20 recently-online users and turns over
completely within seconds (measured: 240 unique/min at a 5s poll, zero dupes).
Usernames go into the state DB (INSERT OR IGNORE = dedup), where the ratings and
profile crawlers pick them up. Safe to stop/restart any time; run it in the
background for days.

    venv/bin/python data/crawler/collect_usernames.py                 # until Ctrl-C
    venv/bin/python data/crawler/collect_usernames.py --minutes 60
    venv/bin/python data/crawler/collect_usernames.py --iterations 3  # smoke test
"""
from __future__ import annotations

import argparse
import re
import time
from urllib.parse import unquote

import common

USERS_URL = "https://myanimelist.net/users.php"
PROFILE_RE = re.compile(r'href="/profile/([^"/]+)"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between polls")
    ap.add_argument("--minutes", type=float, help="stop after this many minutes")
    ap.add_argument("--iterations", type=int, help="stop after this many polls")
    args = ap.parse_args()

    common.install_signals()
    conn = common.db()
    client = common.Client(min_interval=args.interval, headers={"User-Agent": common.BROWSER_UA})

    t_end = time.time() + args.minutes * 60 if args.minutes else None
    total_new = 0
    polls = 0
    while not common.STOP:
        if t_end and time.time() > t_end:
            break
        if args.iterations is not None and polls >= args.iterations:
            break
        r = client.get(USERS_URL)
        polls += 1
        if r is None or r.status_code != 200:
            print(f"[-] poll {polls}: HTTP {r.status_code if r else 'fail'}")
            continue
        names = {unquote(m) for m in PROFILE_RE.findall(r.text)}
        now = common.now_iso()
        cur = conn.executemany(
            "INSERT OR IGNORE INTO users(username, discovered_at) VALUES(?,?)",
            [(n, now) for n in names],
        )
        conn.commit()
        total_new += cur.rowcount
        if polls % 12 == 0 or args.iterations:
            db_total = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            print(f"[+] poll {polls}: +{cur.rowcount} new (run total {total_new}, db total {db_total})")

    db_total = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    print(f"[done] {polls} polls, {total_new} new usernames this run, {db_total} in db")
    conn.close()


if __name__ == "__main__":
    main()
