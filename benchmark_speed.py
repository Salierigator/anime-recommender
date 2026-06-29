"""benchmark_speed.py — so tốc độ recommend giữa baselines / two-tower / two-stage.

Đo CÔNG BẰNG "tốc độ recommend" = online latency mỗi request: history 1 user -> list
top-K anime_idx. Loại khỏi đồng hồ: load artifact + train offline (ALS/sim/popularity)
+ lookup title. Tính vào đồng hồ: fetch history + score + mask seen + topk (+ rerank
LightGBM cho two-stage). CPU-only, pin THREADS thread đồng đều mọi method.

2 chỉ số: per-request latency (E=1: p50/mean/p90/p95 ms) + throughput (batch 64, rec/s).

Vì baselines (retriever/src) và two-tower/two-stage (ranker/src qua Recommender) đụng tên
module flat (config/metrics) -> KHÔNG chạy chung 1 process. Script tự spawn 2 worker
subprocess cô lập, rồi gộp kết quả ra benchmark_speed.txt.

    venv/bin/python benchmark_speed.py [N_SAMPLE]      # mặc định 500 user warm-test
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
THREADS = 4                  # pin đồng đều mọi method (torch + BLAS + implicit)
TOPK = 20                    # cỡ list cuối cùng trả user
THROUGHPUT_BATCH = 64
WARMUP = 5
SEED = 123

METHOD_ORDER = ["random", "popular", "meta_popular", "content", "itemknn", "mf",
                "two-tower", "two-stage"]


# ───────────────────────── helpers chung ─────────────────────────
def summarize(times: list[float]) -> dict:
    a = np.asarray(times, dtype=np.float64) * 1000.0     # -> ms
    return {"p50": float(np.percentile(a, 50)), "mean": float(a.mean()),
            "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)),
            "n": int(a.size)}


def bench_latency(call_one, sample) -> dict:
    """E=1: thời gian sinh top-K cho từng user (sau warmup)."""
    for uid in sample[:WARMUP]:
        call_one(uid)
    times = []
    for uid in sample:
        t = time.perf_counter()
        call_one(uid)
        times.append(time.perf_counter() - t)
    return summarize(times)


def bench_throughput(call_chunk, sample) -> float:
    """rec/s khi xử lý theo batch (server thực phục vụ nhiều request)."""
    call_chunk(sample[:min(THROUGHPUT_BATCH, len(sample))])      # warmup
    t = time.perf_counter()
    for s in range(0, len(sample), THROUGHPUT_BATCH):
        call_chunk(sample[s:s + THROUGHPUT_BATCH])
    el = time.perf_counter() - t
    return len(sample) / el


def host_info() -> tuple[str, str]:
    """(host, stack) để stamp vào header — số tuyệt đối chỉ có nghĩa khi biết máy."""
    import platform
    if platform.system() == "Darwin":
        def sc(k):
            return subprocess.run(["sysctl", "-n", k], capture_output=True,
                                  text=True).stdout.strip()
        cpu, cores, memb = sc("machdep.cpu.brand_string"), sc("hw.logicalcpu"), sc("hw.memsize")
        ram = f"{int(memb)//1024**3} GB" if memb.isdigit() else "?"
        host = f"{cpu} · {cores} logical cores · {ram} · macOS {platform.mac_ver()[0]}"
    else:
        host = f"{platform.processor() or '?'} · {os.cpu_count()} logical cores · {platform.platform()}"
    import torch
    import lightgbm
    stack = (f"python {platform.python_version()} · torch {torch.__version__} · "
             f"lightgbm {lightgbm.__version__} · numpy {np.__version__}")
    return host, stack


def pick_sample(uids: list[int], n: int) -> list[int]:
    uids = sorted(int(u) for u in uids)
    rng = np.random.default_rng(SEED)
    n = min(n, len(uids))
    return sorted(int(x) for x in rng.choice(uids, n, replace=False))


# ═══════════════════════ WORKER: baselines ═══════════════════════
def run_baselines(out_path: Path, n_sample: int):
    import torch
    torch.set_num_threads(THREADS)
    base = ROOT / "retriever" / "baselines"
    sys.path.insert(0, str(base))
    sys.path.insert(0, str(ROOT / "retriever" / "src"))
    import data as data_mod                              # noqa
    import metrics                                       # noqa
    import _eval                                         # noqa
    import content_based as content_mod                  # noqa
    import itemknn as knn_mod                            # noqa
    import meta_popular as meta_mod                      # noqa
    import mf as mf_mod                                  # noqa

    cfg, spec, logq, users, q_warm, m_warm, _, _, _, _ = _eval.setup("test")
    cfg.device = "cpu"
    logq = logq.cpu()
    N = logq.shape[0]
    candidate_mask = torch.isfinite(logq)
    cap = cfg.eval_history_cap
    sample = pick_sample(list(q_warm.keys()), n_sample)
    neg_inf = float("-inf")

    def make_callables(score_fn):
        def rank(uid_list):
            u = np.asarray(uid_list, dtype=np.int64)
            ids, _, _ = users.eval_history_batch(u, cap)
            hist = torch.from_numpy(ids).long()
            scores = score_fn(u, hist)                            # [E, N]
            scores[:, ~candidate_mask] = neg_inf
            mpad = torch.from_numpy(metrics._pad_mask_ids(m_warm, list(uid_list)))
            scores.scatter_(1, mpad, neg_inf)
            return torch.topk(scores, TOPK, dim=1).indices
        return (lambda uid: rank([uid])), rank

    # --- dựng score_fn từng baseline (phần OFFLINE, không tính giờ) ---
    score_fns = {}

    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    score_fns["random"] = lambda u, h: torch.rand(len(u), N, generator=g)

    pop = torch.from_numpy(
        np.bincount(data_mod.ExamplesDataset(cfg.train_data, "train").anime_idx,
                    minlength=N).astype(np.float32))
    score_fns["popular"] = lambda u, h: pop.unsqueeze(0).expand(len(u), N).clone()

    meta = meta_mod.members_scores(cfg, N)
    score_fns["meta_popular"] = lambda u, h: meta.unsqueeze(0).expand(len(u), N).clone()

    Cn = torch.from_numpy(content_mod.build_content_matrix(cfg, use_idf=True))
    score_fns["content"] = content_mod.make_score_fn(Cn)

    import scipy.sparse as sp  # noqa
    ui = knn_mod.build_user_items(cfg, spec["num_users"], N)
    S = knn_mod.fit_sim(ui, K=50)                                 # config chốt (BASELINES.md)
    score_fns["itemknn"] = knn_mod.make_score_fn(cfg, S, N)

    # MF: online latency (fold-in + matmul) phụ thuộc factors=128, KHÔNG phụ thuộc #train-user
    # -> fit f128/α1 (ndcg-opt) trên subset cho nhanh; latency vẫn trung thực (note ở header).
    u_all, a_all = mf_mod.load_train_arrays(cfg)
    ui_sub = mf_mod.subset_users_matrix(u_all, a_all, spec["num_users"], N, 15_000, cfg.seed)
    mf_model = mf_mod.fit_als(ui_sub, factors=128, reg=0.05, alpha=1.0, iters=15,
                              seed=cfg.seed, num_threads=THREADS)
    score_fns["mf"] = mf_mod.make_score_fn(cfg, mf_model, N)

    results = {}
    for name in ["random", "popular", "meta_popular", "content", "itemknn", "mf"]:
        call_one, call_chunk = make_callables(score_fns[name])
        lat = bench_latency(call_one, sample)
        rps = bench_throughput(call_chunk, sample)
        results[name] = {"latency": lat, "throughput_rps": rps}
        print(f"[baselines] {name:14s} p50={lat['p50']:.2f}ms  {rps:.0f} rec/s", flush=True)

    out_path.write_text(json.dumps({"methods": results, "n_cand": int(candidate_mask.sum()),
                                    "n_sample": len(sample)}))


# ═══════════════════════ WORKER: model (two-tower + two-stage) ═══════════════════════
def run_model(out_path: Path, n_sample: int):
    import torch
    torch.set_num_threads(THREADS)
    sys.path.insert(0, str(ROOT / "service" / "backend"))
    from app.ml.recommender import Recommender         # noqa  (kích hoạt thứ tự import an toàn)
    import pool                                         # noqa  (đã cache đúng config ranker)
    from features import build_frame                    # noqa

    rec = Recommender(device="cpu")
    item_cache = rec.enc.item_cache
    uh = pool.UsersHistory()
    seen = pool.load_eval_seen()
    queries, _ = pool.load_queries("test")
    sample = pick_sample(list(queries.keys()), n_sample)

    def inputs(uid_list):
        hist_ids, hist_sc, gender, joined, seen_lists = [], [], [], [], []
        for u in uid_list:
            ids, sc = uh.history(u)
            hist_ids.append(ids.astype(np.int64))
            hist_sc.append(sc.astype(np.int64))
            gender.append(uh.gender_id[u])
            joined.append(uh.joined_bucket[u])
            seen_lists.append(seen.get(u, ids).astype(np.int64))
        return (hist_ids, hist_sc, np.asarray(gender, np.int64),
                np.asarray(joined, np.int64), seen_lists)

    def encode(inp):
        hist_ids, hist_sc, gender, joined, _ = inp
        return pool.encode_users(rec.enc, hist_ids, hist_sc, gender, joined, rec.cap)

    # ---------- two-tower: full-catalog cosine top-K ----------
    def tt_chunk(uid_list):
        inp = inputs(uid_list)
        U = encode(inp)
        cand, _ = pool.topk_pool(U, item_cache, inp[4], TOPK)
        return cand

    # ---------- two-stage: cosine top-k_retrieve (warm-only) -> rerank LightGBM ----------
    def ts_parts(uid_list):
        """Trả (t_retrieval, t_rerank). Tách 2 stage để báo cáo chi phí rerank."""
        inp = inputs(uid_list)
        hist_ids, hist_sc = inp[0], inp[1]
        t0 = time.perf_counter()
        U = encode(inp)
        cand, cos = pool.topk_pool(U, item_cache, inp[4], rec.k_retrieve,
                                   cold_idx=rec.cold_idx)
        t1 = time.perf_counter()
        ages = np.full(len(uid_list), np.nan, dtype=np.float64)
        stats = pool.user_stats_from_support(hist_sc, ages)
        cross = pool.cross_features(rec.V, rec.itemfeat, cand, cos, hist_ids, stats)
        X = build_frame(rec.itemfeat, cand.ravel(), cross)
        pred = rec.booster.predict(X).reshape(cand.shape)
        for row in pred:                                         # top-K mỗi user
            np.argsort(-row)[:TOPK]
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1

    def ts_chunk(uid_list):
        ts_parts(uid_list)

    results = {}

    # two-tower
    tt_lat = bench_latency(lambda uid: tt_chunk([uid]), sample)
    tt_rps = bench_throughput(tt_chunk, sample)
    results["two-tower"] = {"latency": tt_lat, "throughput_rps": tt_rps}
    print(f"[model] two-tower  p50={tt_lat['p50']:.2f}ms  {tt_rps:.0f} rec/s", flush=True)

    # two-stage (+ tách retrieval / rerank)
    for uid in sample[:WARMUP]:
        ts_parts([uid])
    tot, ret, rer = [], [], []
    for uid in sample:
        a, b = ts_parts([uid])
        ret.append(a); rer.append(b); tot.append(a + b)
    ts_rps = bench_throughput(ts_chunk, sample)
    results["two-stage"] = {"latency": summarize(tot), "throughput_rps": ts_rps,
                            "latency_retrieval": summarize(ret),
                            "latency_rerank": summarize(rer)}
    print(f"[model] two-stage  p50={summarize(tot)['p50']:.2f}ms  {ts_rps:.0f} rec/s", flush=True)

    out_path.write_text(json.dumps({"methods": results, "k_retrieve": rec.k_retrieve,
                                    "n_sample": len(sample)}))


# ═══════════════════════ orchestrator ═══════════════════════
def spawn(worker: str, out_path: Path, n_sample: int):
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[v] = str(THREADS)
    env["CUDA_VISIBLE_DEVICES"] = ""
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--worker", worker, "--out", str(out_path), "--n", str(n_sample)]
    print(f"\n=== spawn worker '{worker}' (threads={THREADS}) ===", flush=True)
    subprocess.run(cmd, env=env, check=True)


def render(baselines: dict, model: dict, n_sample: int) -> str:
    methods = {**baselines["methods"], **model["methods"]}
    L = []
    host, stack = host_info()
    L.append("# benchmark_speed — tốc độ recommend (online latency mỗi request)")
    L.append(f"# generated {time.strftime('%Y-%m-%dT%H:%M:%S')}  device=cpu  threads={THREADS}")
    L.append(f"# host  = {host}")
    L.append(f"# stack = {stack}")
    L.append(f"# sample = {n_sample} user warm-test ngẫu nhiên (seed={SEED})  top_k={TOPK}"
             f"  k_retrieve(two-stage)={model.get('k_retrieve', '?')}")
    L.append(f"# candidates (finite logq) = {baselines.get('n_cand', '?'):,}")
    L.append("# vùng đo = fetch history -> score -> mask seen -> topk (+ rerank LightGBM cho")
    L.append("#   two-stage). LOẠI: load artifact, train offline (ALS/sim/popularity), lookup title.")
    L.append("# MF: model fit subset 15k-user (f128/α1) — online latency phụ thuộc factors, KHÔNG")
    L.append("#   phụ thuộc #train-user nên trung thực; itemknn fit FULL train (K=50).")
    L.append("")
    L.append("## per-request latency (E=1, ms) + throughput (batch "
             f"{THROUGHPUT_BATCH}, rec/s)")
    L.append(f"{'method':<14} {'p50':>8} {'mean':>8} {'p90':>8} {'p95':>8} {'rec/s':>10}")
    L.append("-" * 62)
    for m in METHOD_ORDER:
        r = methods[m]
        lat = r["latency"]
        L.append(f"{m:<14} {lat['p50']:>8.2f} {lat['mean']:>8.2f} {lat['p90']:>8.2f} "
                 f"{lat['p95']:>8.2f} {r['throughput_rps']:>10.0f}")
    L.append("")
    ts = methods["two-stage"]
    L.append("## two-stage — tách chi phí 2 stage (per-request p50/mean ms)")
    L.append(f"  retrieval (cosine top-{model.get('k_retrieve', '?')})  "
             f"p50={ts['latency_retrieval']['p50']:.2f}  mean={ts['latency_retrieval']['mean']:.2f}")
    L.append(f"  rerank (29 feat + LightGBM)   "
             f"p50={ts['latency_rerank']['p50']:.2f}  mean={ts['latency_rerank']['mean']:.2f}")
    L.append(f"  tổng                          "
             f"p50={ts['latency']['p50']:.2f}  mean={ts['latency']['mean']:.2f}")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", choices=["baselines", "model"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("n_sample", nargs="?", type=int, default=500)
    args = ap.parse_args()

    if args.worker == "baselines":
        run_baselines(args.out, args.n); return
    if args.worker == "model":
        run_model(args.out, args.n); return

    n_sample = args.n_sample
    scratch = Path(os.environ.get("TMPDIR", "/tmp"))
    b_out, m_out = scratch / "bench_baselines.json", scratch / "bench_model.json"
    spawn("baselines", b_out, n_sample)
    spawn("model", m_out, n_sample)
    baselines = json.loads(b_out.read_text())
    model = json.loads(m_out.read_text())
    text = render(baselines, model, n_sample)
    (ROOT / "benchmark_speed.txt").write_text(text)
    print("\n" + text)
    print(f"saved {ROOT / 'benchmark_speed.txt'}")


if __name__ == "__main__":
    main()
