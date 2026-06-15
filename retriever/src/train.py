"""Training loop Two-Tower retrieval (xem plan.md, docs/TRAIN_DATA.md §6).

fit(cfg): build artifact + model, train InfoNCE+logQ+hard-neg, eval cold-by-user, checkpoint.
CLI: `python model/train.py --smoke` chạy nhanh trên subset để verify pipeline.
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg_mod
import data as data_mod
import metrics as metrics_mod
from loss import info_nce_logq
from model import TwoTower


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build(cfg: cfg_mod.TwoTowerConfig):
    spec = data_mod.load_feature_spec(cfg.train_data)
    logq = data_mod.load_logq(cfg.train_data).to(cfg.device)
    item_table = data_mod.ItemTable(cfg.train_data, cfg.synopsis_emb_file,
                                    cfg.synopsis_low_info_file).to(cfg.device)
    users = data_mod.UserTable(cfg.train_data, spec["hard_neg_cap"])
    model = TwoTower(spec, cfg, item_table).to(cfg.device)
    return spec, logq, item_table, users, model


def fit(cfg: cfg_mod.TwoTowerConfig):
    set_seed(cfg.seed)
    spec, logq, item_table, users, model = build(cfg)

    train_ds = data_mod.ExamplesDataset(cfg.train_data, "train", subset=cfg.subset,
                                        max_per_user=cfg.max_examples_per_user, seed=cfg.seed,
                                        user_frac=cfg.train_user_frac, user_frac_seed=cfg.subset_seed)
    collate = data_mod.Collate(users, cfg.hist_dropout, cfg.m_hardneg, cfg.train_hist_len)
    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=cfg.num_workers, drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    print(f"[fit] device={cfg.device} · train {len(train_ds):,} ex · "
          f"{len(loader):,} steps/epoch · {cfg.epochs} epochs")

    # eval warm protocol v2: queries + mask (seen − query). Cold slice KHÔNG đo trong
    # loop (final-exam discipline) — dùng metrics.run_cold_eval từ notebook/script.
    eval_ds = data_mod.ExamplesDataset(cfg.train_data, cfg.eval_split)
    queries = metrics_mod.group_examples(eval_ds.user_idx, eval_ds.anime_idx)
    mask_ids = metrics_mod.build_masks(data_mod.load_eval_seen(cfg.train_data), queries)

    opt_cls = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
    opt = opt_cls(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(loader) * cfg.epochs)
             if cfg.cosine_lr else None)

    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    headline = f"recall@{cfg.headline_k}"
    history = {"loss_steps": [], "loss_vals": [], "eval_steps": [], "eval_metrics": []}
    # loss_sum/loss_cnt: gom loss (đã log mỗi log_every) để in TB mỗi lần eval rồi reset
    state = {"best": -1.0, "loss_sum": 0.0, "loss_cnt": 0, "saved_once": False,
             "since_best": 0, "stop": False}

    def do_eval(step, epoch):
        model.refresh_item_cache()                         # cache mới nhất (warm) trước eval
        m = metrics_mod.evaluate(model, users, queries, logq, cfg.eval_ks, mask_ids,
                                 eval_history_cap=cfg.eval_history_cap)
        history["eval_steps"].append(step)
        history["eval_metrics"].append(m)
        if state["loss_cnt"]:
            loss_str = f"loss {state['loss_sum'] / state['loss_cnt']:.3f}"
            state["loss_sum"], state["loss_cnt"] = 0.0, 0
        else:
            loss_str = "loss   —  "                         # baseline random-init: chưa có loss
        rec = "  ".join(f"@{k} {m[f'recall@{k}']:.3f}" for k in cfg.eval_ks)
        improved = m[headline] > state["best"] + cfg.early_stop_min_delta   # tính TRƯỚC khi cập nhật best
        mark = ""
        if m[headline] > state["best"]:                    # vẫn lưu MỌI cải thiện strict -> best.pt = true best
            state["best"] = m[headline]
            ckpt = cfg.ckpt_dir / "best.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg, "metrics": m,
                        "epoch": epoch, "step": step}, ckpt)
            mark = "  ★"
            if not state["saved_once"]:                    # in path checkpoint đúng 1 lần
                print(f"  (best.pt -> {ckpt})")
                state["saved_once"] = True
        if improved:                                       # early-stop: reset/đếm theo cải thiện ĐÁNG KỂ (min_delta)
            state["since_best"] = 0
        else:
            state["since_best"] += 1
            if cfg.early_stop_patience is not None and state["since_best"] >= cfg.early_stop_patience:
                state["stop"] = True
                mark += "  ⏹"
        print(f"  step {step:>7,} │ {loss_str} │ recall {rec} │ "
              f"ndcg@{cfg.eval_ks[-1]} {m[f'ndcg@{cfg.eval_ks[-1]}']:.3f} │ n={int(m['n_users']):,}{mark}")

    step = 0
    do_eval(0, 0)                                          # baseline random-init (neo đường cong)
    for epoch in range(cfg.epochs):
        if cfg.max_examples_per_user is not None and epoch > 0:
            train_ds.resample(epoch)                       # rút lại mẫu per-user mỗi epoch
        print(f"── epoch {epoch} ({len(train_ds):,} ex) " + "─" * 24)
        model.refresh_item_cache()                         # cache đầu epoch
        t0 = time.time()
        for batch in loader:
            if step % cfg.cache_refresh_steps == 0 and step > 0:
                model.refresh_item_cache()
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            U, V_pos, V_hn = model(batch)
            loss = info_nce_logq(
                U, V_pos, V_hn, batch["hardneg_mask"], batch["pos"],
                logq, cfg.tau, cfg.beta, cfg.logq_alpha,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            if step % cfg.log_every == 0:                  # log loss cho curve + gom để in TB
                lv = loss.item()
                history["loss_steps"].append(step)
                history["loss_vals"].append(lv)
                state["loss_sum"] += lv
                state["loss_cnt"] += 1
            if cfg.eval_every_steps > 0 and step > 0 and step % cfg.eval_every_steps == 0:
                do_eval(step, epoch)
                if state["stop"]:
                    break                                  # early-stop giữa epoch
            step += 1

        if state["stop"]:                                  # đã dừng giữa epoch -> bỏ eval cuối epoch (tránh dup)
            print(f"   └ epoch {epoch} dừng sớm (early-stop) trong {time.time() - t0:.1f}s · best={state['best']:.4f}")
            break
        do_eval(step, epoch)                               # eval cuối epoch
        print(f"   └ epoch {epoch} xong trong {time.time() - t0:.1f}s")
        if state["stop"]:                                  # plateau phát hiện ở eval cuối epoch
            break
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="subset nhỏ, verify pipeline")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--synopsis", action="store_true", help="bật use_synopsis (cần artifact synopsis_emb.npy)")
    ap.add_argument("--user_frac", type=float, default=None, help="train_user_frac — subset % user cho HP-search")
    ap.add_argument("--optimizer", type=str, default=None, help="adam|adamw")
    ap.add_argument("--early_stop_patience", type=int, default=None, help="dừng nếu headline không cải thiện sau N eval")
    args = ap.parse_args()

    cfg = cfg_mod.TwoTowerConfig()
    if args.smoke:
        cfg.subset = 50_000
        cfg.batch_size = 512
        cfg.epochs = 1
        cfg.cache_refresh_steps = 20
        cfg.log_every = 10
        cfg.num_workers = 0
        cfg.ckpt_dir = cfg.ckpt_dir / "smoke"   # đừng đè best.pt thật (production) bằng model smoke
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    if args.synopsis:
        cfg.use_synopsis = True
    if args.user_frac is not None:
        cfg.train_user_frac = args.user_frac
    if args.optimizer is not None:
        cfg.optimizer = args.optimizer
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    fit(cfg)


if __name__ == "__main__":
    main()
