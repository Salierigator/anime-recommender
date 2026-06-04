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
    item_table = data_mod.ItemTable(cfg.train_data).to(cfg.device)
    users = data_mod.UserTable(cfg.train_data, spec["k_history"], spec["hard_neg_cap"])
    model = TwoTower(spec, cfg, item_table).to(cfg.device)
    return spec, logq, item_table, users, model


def fit(cfg: cfg_mod.TwoTowerConfig):
    set_seed(cfg.seed)
    spec, logq, item_table, users, model = build(cfg)

    train_ds = data_mod.ExamplesDataset(cfg.train_data, "train", subset=cfg.subset)
    collate = data_mod.Collate(users, cfg.hist_dropout, cfg.m_hardneg)
    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate, num_workers=cfg.num_workers, drop_last=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    print(f"[fit] device={cfg.device} · train {len(train_ds):,} ex · "
          f"{len(loader):,} steps/epoch · {cfg.epochs} epochs")

    eval_ds = data_mod.ExamplesDataset(cfg.train_data, cfg.eval_split)
    queries = metrics_mod.group_examples(eval_ds.user_idx, eval_ds.anime_idx)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(loader) * cfg.epochs)
             if cfg.cosine_lr else None)

    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    headline = f"recall@{cfg.eval_ks[-1]}"
    history = {"loss_steps": [], "loss_vals": [], "eval_steps": [], "eval_metrics": []}
    # loss_sum/loss_cnt: gom loss (đã log mỗi log_every) để in TB mỗi lần eval rồi reset
    state = {"best": -1.0, "loss_sum": 0.0, "loss_cnt": 0, "saved_once": False}

    def do_eval(step, epoch):
        model.refresh_item_cache()                         # cache mới nhất trước eval
        m = metrics_mod.evaluate(model, users, queries, logq, cfg.eval_ks)
        history["eval_steps"].append(step)
        history["eval_metrics"].append(m)
        if state["loss_cnt"]:
            loss_str = f"loss {state['loss_sum'] / state['loss_cnt']:.3f}"
            state["loss_sum"], state["loss_cnt"] = 0.0, 0
        else:
            loss_str = "loss   —  "                         # baseline random-init: chưa có loss
        rec = "  ".join(f"@{k} {m[f'recall@{k}']:.3f}" for k in cfg.eval_ks)
        mark = ""
        if m[headline] > state["best"]:
            state["best"] = m[headline]
            ckpt = cfg.ckpt_dir / "best.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg, "metrics": m,
                        "epoch": epoch, "step": step}, ckpt)
            mark = "  ★"
            if not state["saved_once"]:                    # in path checkpoint đúng 1 lần
                print(f"  (best.pt -> {ckpt})")
                state["saved_once"] = True
        print(f"  step {step:>7,} │ {loss_str} │ recall {rec} │ "
              f"ndcg@{cfg.eval_ks[-1]} {m[f'ndcg@{cfg.eval_ks[-1]}']:.3f} │ n={int(m['n_users']):,}{mark}")

    step = 0
    do_eval(0, 0)                                          # baseline random-init (neo đường cong)
    for epoch in range(cfg.epochs):
        print(f"── epoch {epoch} " + "─" * 38)
        model.refresh_item_cache()                         # cache đầu epoch
        t0 = time.time()
        for batch in loader:
            if step % cfg.cache_refresh_steps == 0 and step > 0:
                model.refresh_item_cache()
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            U, V_pos, V_hn = model(batch)
            loss = info_nce_logq(
                U, V_pos, V_hn, batch["hardneg_mask"], batch["pos"],
                logq, cfg.tau, cfg.beta,
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
            step += 1

        do_eval(step, epoch)                               # eval cuối epoch
        print(f"   └ epoch {epoch} xong trong {time.time() - t0:.1f}s")
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="subset nhỏ, verify pipeline")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = cfg_mod.TwoTowerConfig()
    if args.smoke:
        cfg.subset = 50_000
        cfg.batch_size = 512
        cfg.epochs = 1
        cfg.cache_refresh_steps = 20
        cfg.log_every = 10
        cfg.num_workers = 0
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    fit(cfg)


if __name__ == "__main__":
    main()
