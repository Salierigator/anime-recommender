"""Crawl the full anime catalog via Jikan /anime pagination (~1215 pages x 25).

Jikan is used (not MAL v2) because its schema matches details.csv exactly:
genres/themes/demographics/studios come pre-split and `favorites` exists (absent
from MAL v2, needed by the ranker). Catalog pages are served from Jikan's cache,
so this works even during Jikan<->MAL outages. Full sweep ~30-40 min at 1 req/s.

Output: data/raw/details.jsonl.gz — one raw Jikan anime object per line (keep
everything, flatten to CSV during cleaning; dedup by mal_id keep-last there).
Resume: last finished page in the state DB; safe to stop/restart.

    venv/bin/python data/crawler/crawl_details.py              # full sweep / resume
    venv/bin/python data/crawler/crawl_details.py --pages 2    # smoke test
    venv/bin/python data/crawler/crawl_details.py --restart    # new sweep from page 1
"""
from __future__ import annotations

import argparse
import gzip
import json

import common

OUT = common.RAW / "details.jsonl.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, help="crawl at most N pages this run")
    ap.add_argument("--restart", action="store_true",
                    help="start a fresh sweep from page 1 (rotates the old output file)")
    args = ap.parse_args()

    common.install_signals()
    conn = common.db()
    client = common.Client(min_interval=1.1, tries=8)

    if args.restart:
        common.kv_set(conn, "details_next_page", 1)
        conn.commit()
        if OUT.exists():
            bak = OUT.with_suffix(f".{common.now_iso()[:10]}.bak.gz")
            OUT.rename(bak)
            print(f"[i] rotated old output -> {bak.name}")

    page = int(common.kv_get(conn, "details_next_page", 1))
    last_page = None
    done_this_run = 0
    n_anime = 0

    with gzip.open(OUT, "at", encoding="utf-8") as f:
        while not common.STOP:
            if args.pages is not None and done_this_run >= args.pages:
                break
            if last_page is not None and page > last_page:
                break
            # no extra params: Jikan returns NSFW entries by default (sfw filter is
            # opt-in), and the bare URL is what stays hot in Jikan's cache
            r = client.get(f"{common.JIKAN_BASE}/anime", params={"page": page})
            if r is None or r.status_code != 200:
                print(f"[-] page {page}: HTTP {r.status_code if r else 'fail'} — stopping (resume later)")
                break
            d = r.json()
            last_page = d["pagination"]["last_visible_page"]
            for row in d["data"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n_anime += len(d["data"])
            page += 1
            done_this_run += 1
            common.kv_set(conn, "details_next_page", page)
            conn.commit()
            if page % 50 == 0 or args.pages:
                print(f"[+] page {page - 1}/{last_page} done ({n_anime} anime this run)")

    if last_page is not None and page > last_page:
        common.kv_set(conn, "details_completed_at", common.now_iso())
        conn.commit()
        print(f"[done] full sweep complete: {last_page} pages -> {OUT}")
    else:
        print(f"[paused] next page = {page}{f'/{last_page}' if last_page else ''} -> rerun to resume")
    conn.close()


if __name__ == "__main__":
    main()
