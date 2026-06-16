# EXPERIMENTS — thử nghiệm chọn retriever final (synopsis · subset HP-search · config linh hoạt)

> Nguồn viết đồ án cho phần **thử nghiệm chọn model retriever cuối cùng**. Bổ sung cho
> `TWO_TOWER_MODEL.md` (kiến trúc + protocol eval) và `DATA_SPLIT.md` (split). Số liệu từng run
> nằm ở Drive `recommender_train_colab/runs.csv` + `runs/v5/<run_name>/{row,config,history}.json`;
> tổng hợp số chốt: `RESULTS.md`. Code: `retriever/src/{config,data,model,train,search}.py`,
> `retriever/data_prep/07_synopsis_emb.py`, `retriever/train.ipynb`.
>
> ⚠️ **Config final hiện tại = `v_final`** (2026-06-16): `history_source=embed`, `train_hist_len=128`,
> 10 epoch, **synopsis ON dim 64** (`final_syn`/best.pt). Số `v5_hist64_ep2` trong `RESULTS.md`/
> `PROGRESS.md` đã STALE — chờ re-measure + retrain ranker (`docs/RANKER.md §9`).

## 0. Bối cảnh

Retriever Two-Tower đã chạy được + chốt tạm `v5_hist64_ep2` (warm test recall@200 .6608). Mục tiêu
giai đoạn này: **thử nghiệm có hệ thống để chọn cấu hình final** — thêm 1 nguồn tín hiệu mới
(synopsis), và **search hyperparameter** rẻ + có bằng chứng tái lập cho báo cáo. Ràng buộc: train
chỉ trên Colab (`train.ipynb`); local chỉ test/smoke; mọi run log lên Drive.

Ba trục thử nghiệm, tất cả bật/tắt qua `TwoTowerConfig` (không hardcode):

| Trục | Knob (default) | Artifact cần |
|---|---|---|
| Synopsis content | `use_synopsis=False`, `synopsis_dim=48`, `synopsis_proj_hidden=[]`, `synopsis_normalize='none'` | `synopsis_emb.npy`, `synopsis_low_info.npy` |
| Subset HP-search | `train_user_frac=None`, `subset_seed=12345` | — (dùng full train-data) |
| Regularizer | `optimizer='adam'`, `weight_decay=0.0` | — |

---

## 1. Synopsis embedding (content text-emb, item-side)

**Vì sao**: model hiện bỏ HOÀN TOÀN cột `synopsis` (94% anime có, ~100% tiếng Anh). Synopsis mang
tín hiệu nội dung mà 9 feature categorical/multi-hot không có — kỳ vọng cải thiện nhất ở **cold item**
(anime mới chỉ có content, id→OOV) và head-precision warm.

**Pipeline (frozen embedding + projection trainable):**

1. **Offline 1 lần** — `data_prep/07_synopsis_emb.py` (chạy local CPU, SAU `01`):
   - Đọc `cleaned-data/details.csv::synopsis` + `train-data/anime_id_map.parquet` (join mal_id→anime_idx).
   - Tiền xử lý: **strip `(Source: ...)`** ở đuôi (~32.9% dính, boilerplate gây nhiễu);
     cờ `low_info = NaN | <50 ký tự | placeholder "No synopsis…"` (đo thực tế **15.1%**, khớp ước lượng
     6% NaN + 9.4% <50 ký tự).
   - Encode `all-MiniLM-L6-v2` (384 dim), **L2-normalize trong script** → `synopsis_emb.npy [num_items,384]`
     (row 0,1 = PAD/OOV = zeros) + `synopsis_low_info.npy [num_items] bool` (row 0,1 = True) + `synopsis_meta.json`.
   - **Frozen, swappable**: đổi `bge-small-en-v1.5`/`gte-small` (cùng 384) qua `cfg.synopsis_emb_file`,
     không train, TÁCH khỏi vòng re-export retriever.
2. **Trong model** (`ItemTower`, gate `cfg.use_synopsis`):
   - Chiếu raw 384 → `synopsis_dim` (mặc định 48 ≈ ngang khối genres/themes/studios; không lấn 60 dim
     content còn lại) bằng `_mlp(384, synopsis_proj_hidden, synopsis_dim)` — `[]` = Linear thuần,
     `[128]` = 1 hidden nếu underfit. Concat vào content path SAU studios, trước id → `in_dim` tự cộng.
   - **Low-info → vec học được**: row low_info (gồm PAD/OOV) thay projection bằng `no_synopsis`
     (`nn.Parameter`, học được) → tower phớt lờ synopsis rác mà không hỏng phần synopsis tốt.
     Thay SAU projection (post-proj) nên né NaN do normalize vec ~0.
3. **Export/serve**: `item_vectors.npy` build qua `refresh_item_cache` chạy ItemTower → synopsis vào
   vector TỰ ĐỘNG, `export.py`/`test_export.py` không cần sửa (output vẫn `d=128`; synopsis nằm trong
   tower trước MLP cuối). Chỉ cần `synopsis_emb.npy` có mặt trong train-data lúc export.

**Ladder thử**: (1) baseline `use_synopsis=True, dim=48, hidden=[]` (MiniLM) vs control OFF — đo warm
recall@200 + cold; (2) nếu cải thiện → `dim=64`, `hidden=[128]`, rồi swap `bge-small`. Mỗi lần lật 1 đòn bẩy.

**Kết quả (2026-06-16)**: ablation `final` (syn OFF) vs `final_syn` (syn ON) — **cùng config v_final**,
chỉ khác `use_synopsis` — xác nhận synopsis cải thiện **warm** (test ndcg@10 +.025, r@200 +.010), đã
chốt `synopsis_dim=64` vào best.pt. Bảng số + caveat cold (chưa có ablation): **`docs/SYNOPSIS_EMB.md`**.

---

## 2. Subset cho hyperparameter-search (runtime, lọc % user)

**Vấn đề**: mỗi run Colab tốn GPU-phút/giờ; full train ~67M example/epoch. Cần subset để xếp hạng
config RẺ, nhưng **không bóp méo phân phối / không kết luận sai vì data nhỏ**.

**Thiết kế** (`ExamplesDataset._keep_user_frac`, bật bằng `cfg.train_user_frac`):
- **Lọc theo USER ngẫu nhiên** (random distinct user, seeded `subset_seed`), KHÔNG phải first-N
  (first-N = lát user_idx liền kề → lệch phân phối). Chỉ áp **split=train**.
- **GIỮ NGUYÊN**: item catalog đầy đủ (22.8k → độ khó retrieval không đổi, synopsis áp mọi item),
  `logq.npy` (xem §2.1), và **eval val/test FULL** (variance metric thấp → xếp hạng config đáng tin).
- Vì sao subset USER (không subset ITEM): subset item đổi catalog size + logQ + tập cold H + độ khó
  "retrieve từ 22.8k" → phá semantics; subset user chỉ giảm số example train, giữ mọi thứ khác.

### 2.1. Vì sao KHÔNG cần tính lại logQ

logQ vào loss (`loss.py`): `s_in[:,j] -= alpha * logq[pos_j]` — trừ theo CỘT positive rồi vào softmax
(`F.cross_entropy`). Sample ngẫu nhiên frac user → count train mỗi item co ~×frac → `log(count)` dịch
~`log(frac)` **toàn cục**; một hằng số cộng GIỐNG NHAU cho mọi cột **triệt tiêu trong softmax**.
- Sai số per-item ở **tail** (count nhỏ, floor `max(count,1)` ở `06`) là THẬT (lệch khỏi `log(frac)`
  cỡ `O(1/√(count·frac))`), nhưng là **common-mode giữa các config** đem so → không đổi thứ hạng.
- Bảo hiểm: bước **confirm top-K trên full data** (logQ đúng) chốt lại. ⇒ Diễn đạt báo cáo:
  "subset training dùng logQ full-corpus như popularity prior cố định; thành phần toàn cục triệt tiêu
  trong mẫu số InfoNCE, phần tail common-mode + confirm trên full data" — KHÔNG nói "invariance tuyệt đối".

### 2.2. Phương pháp chống "kết luận sai vì data nhỏ"

1. Subset **15–20%** user (đừng <10%: tail-logQ lệch hơn + pool hard-neg/user thưa, tín hiệu train nhiễu).
2. Eval trên **val FULL** (≈14k user) → variance metric thấp (chênh <0.004 r@100 = noise, `DATA_SPLIT.md §8`).
3. Subset chỉ để **XẾP HẠNG** config (coarse) → **CONFIRM top-K trên full data** → mới chốt + final-exam.
4. Mỗi run log `train_user_frac` + `subset_seed` (lọc `runs.csv` để tách coarse vs confirm).

---

## 3. Config linh hoạt + cơ chế search

**Mọi hyperparam ở `TwoTowerConfig`** (đọc feature vocab/dim từ `feature_spec.json`, không hardcode
trong model). Đòn bẩy mới thêm so với bản trước: synopsis (§1), `train_user_frac`/`subset_seed` (§2),
`optimizer` (`adam|adamw` — AdamW decouple weight_decay, đòn bẩy regularization khi search).

**Search driver** `src/search.py` (importable, unit-test ở `tests/test_search.py`, notebook mỏng):
- `iter_configs(space, method='random'|'grid', n, seed, fixed)` → `(run_name, overrides)`.
  Random > grid khi >3-4 chiều (Bergstra & Bengio 2012); grid = `itertools.product` cho không gian nhỏ.
  `fixed` = knob áp mọi config (vd `{use_item_id:True, epochs:2, train_user_frac:0.15}` cho coarse).
- `deterministic_run_name(overrides)` = encode knob KHÁC default (sorted) → tên ổn định + `canonicalize`
  bỏ knob con khi cha tắt (vd `use_synopsis=False` → bỏ `synopsis_dim`) ⇒ **không chạy lại model giống hệt**.
- **Resume khi Colab rớt session**: `run_search(run_experiment, configs, exists_fn)` bỏ run đã có trên
  Drive (`(RUNS_DIR/run_name).exists()`) — tận dụng dedup `(version,run_name)` của `rebuild_leaderboard`.

**Logging (bằng chứng đồ án)** — mỗi run, `run_experiment` (notebook cell 5) ghi vào Drive
`runs/v5/<run_name>/`: `best.pt` + `history.json` (loss/eval curve) + `row.json` (1 dòng leaderboard) +
**`config.json` = `dataclasses.asdict(cfg)` đầy đủ** (provenance, future-proof). `runs.csv` = gom mọi run,
sort `test_recall@200`, có cột mới `use_synopsis/synopsis_dim/synopsis_normalize/train_user_frac/subset_seed/optimizer`.

---

## 4. Quy trình end-to-end

```bash
# (local, 1 lần) sinh synopsis artifact rồi đẩy 2 .npy lên Drive train-data/
venv/bin/python retriever/data_prep/07_synopsis_emb.py --device cpu
# (local) test trước khi train
venv/bin/python -m pytest retriever/tests -q
cd retriever/src && ../../venv/bin/python train.py --smoke --synopsis --user_frac 0.5 --optimizer adamw
```

Trên Colab (`train.ipynb`): cell 3 base_cfg (knob mới có sẵn) → cell 6b **SEARCH coarse** (random,
subset 15%) → xem `runs.csv` (lọc `train_user_frac=0.15`) chọn top-K → **CONFIRM** chạy lại FULL
(bỏ `train_user_frac`) → chốt model final → re-export `artifacts/` + retrain ranker (`RANKER.md §9`)
→ final-exam (test + test_cold) chấm 1 lần.

## 5. Đọc số khi viết báo cáo

- **Bảng so config**: `runs.csv` (cột knob + `{val,test}_{recall,ndcg}@K`). Tách coarse (subset, có
  `train_user_frac`) vs confirm (full) — chỉ kết luận final từ **confirm trên full data**.
- **Đường cong**: `runs/v5/<run>/history.json` (loss + val metric theo step) → notebook cell 7-8.
- **Cold (anime mới)**: notebook cell 10 (`split='val'` khi tune; `test_cold` = final exam, 1 lần).
- **Provenance đầy đủ 1 run**: `runs/v5/<run>/config.json`. Bar baselines: `RESULTS.md` / `BASELINES.md`.
