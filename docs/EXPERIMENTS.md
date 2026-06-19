# EXPERIMENTS — thử nghiệm chọn retriever final (synopsis · subset HP-search · config linh hoạt · loss ablation)

> Nguồn viết đồ án cho phần **thử nghiệm chọn model retriever cuối cùng**. Bổ sung cho
> `TWO_TOWER_MODEL.md` (kiến trúc + protocol eval) và `DATA_SPLIT.md` (split). Số liệu từng run
> nằm ở Drive `recommender_train_colab/runs.csv` + `runs/v5/<run_name>/{row,config,history}.json`
> (bản local snapshot warm+cold: `retriever/runs/{runs,cold_runs}.csv`);
> tổng hợp số chốt: `RESULTS.md`. Code: `retriever/src/{config,data,model,train,search}.py`,
> `retriever/data_prep/07_synopsis_emb.py`, `retriever/train.ipynb`.
>
> ⚠️ **Config final = `final`** (2026-06-17): `history_source=embed`, `train_hist_len=128`, 10 epoch, d128,
> τ.07, logQ α=1, **synopsis OFF**. Synopsis (`final_syn`) đã test on/off và **bị bác** — cải thiện warm
> nhưng regress cold, mà retriever ưu tiên cold (`docs/SYNOPSIS_EMB.md`). ✅ **Re-export DONE (2026-06-17)**:
> `best.pt`/`artifacts/` giờ là `final` (`CONTRACT.md` step 31500), serve-path official trong `docs/RESULTS.md §3b`;
> ranker đã retrain trên pool `final` (2026-06-18, `lrank_t20_gainLin` — `docs/RANKER.md`) → pipeline đồng bộ.

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

## 1. Synopsis embedding (content text-emb, item-side) — ĐÃ TEST, **BỊ BÁC**

**Vì sao thử**: model bỏ HOÀN TOÀN cột `synopsis` (94% anime có, ~100% tiếng Anh) — tín hiệu nội dung mà
9 feature categorical/multi-hot không có. **Kỳ vọng ban đầu**: giúp nhất ở **cold item** (anime mới chỉ có
content, id→OOV) + head-precision warm.

**Cách làm (tóm tắt — thiết kế + code chi tiết ở `docs/SYNOPSIS_EMB.md`, không lặp ở đây)**: frozen
`all-MiniLM-L6-v2` (384 dim, L2-norm, swappable; sinh offline bằng `data_prep/07_synopsis_emb.py`) →
projection trainable `synopsis_dim` concat vào content path `ItemTower` (gate `cfg.use_synopsis`); row
low-info (~15%) thay bằng `no_synopsis` học được. Export/serve không cần sửa (synopsis chảy qua
`refresh_item_cache`).

**Kết quả (2026-06-17) = REJECTED**: ablation `final` (OFF) vs `final_syn` (ON) — cùng config v_final, chỉ
khác `use_synopsis`:
- **Warm cải thiện** (test ndcg@10 +.064, r@200 +.010) — nhưng dồn vào head-precision, là việc của ranker.
- **Cold REGRESS mạnh** (val_cold r@200 −.115, liked_recall@200 −.148, honly −.066) — ngược động cơ ban đầu.

→ retriever ưu tiên cold (cold serve = cosine trực tiếp), nên **chốt `final` synopsis OFF**. Bảng số đầy
đủ + cơ chế warm↑/cold↓ (co-adaptation với id; cạnh tranh capacity với feature cấu trúc; MiniLM frozen ít
discriminative cho cold): **`docs/SYNOPSIS_EMB.md`**. Code synopsis giữ nguyên (chỉ `use_synopsis=False`).

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

## 4. Loss ablation — logQ / τ / β / m_hardneg

Quét 4 tham số hàm loss `info_nce_logq` (subset 15% user, ep2, checkpoint-path test — coarse, đủ xếp hạng
lever; §2 giải thích vì sao subset tin được). **Mục đích: xác định lever nào THẬT sự đổi kết quả.** Công
thức loss + kiến trúc model GIỮ NGUYÊN — đây chỉ là kết luận empirical về độ nhạy tham số.

### 4.1. logQ α — lever quan trọng nhất (giữ α=1)

| logq_alpha | test r@200 | test ndcg@10 |
|---|---|---|
| 0 (tắt)   | .3416 | .1319 |
| 0.5       | .5738 | .3520 |
| **1.0** ★ | **.6211** | **.5047** |

Tắt logQ (α=0) làm model **sụp** (ndcg@10 .13, r@200 .34): popularity-debiasing là linh hồn — metric recall
thưởng item phổ biến, không trừ logQ thì model gom hết về head và hỏng cá nhân hoá. α=1 (trừ đủ) tốt nhất.

### 4.2. τ (temperature) — lever nhỏ (giữ .07)

| tau | test r@200 | test ndcg@10 |
|---|---|---|
| 0.05 | .6200 | .4920 |
| **0.07** ★ | **.6209** | **.5071** |
| 0.10 | .6181 | .4967 |

Có ảnh hưởng nhưng nhỏ; .07 nhỉnh nhất ở ndcg@10, chênh r@200 trong vùng noise (<.004) → chọn .07.

### 4.3. β và m_hardneg — **VÔ NGHĨA (không phải lever)**

Kết quả phản trực giác đáng ghi: hàm loss thiết kế CÓ nhánh hard-negative (sample từ `hard_neg_ids` của
chính user, nhân hệ số β) — nghe rất hợp lý — nhưng đo ra **không đổi kết quả**.

**β (bs16384, mhn=5):** .5 → r@200 .6204 / ndcg@10 .5057 · 1.0 → .6211 / .5047 · 2.0 → .6210 / .5054.
**m_hardneg (bs16384, β=1):** 0 → .6208 / .5010 · 5 → .6211 / .5047 · 10 → .6210 / .5062.

Mọi chênh lệch <.004 (noise floor `DATA_SPLIT.md §8`); m_hardneg=0 (tắt hẳn hard-neg) ngang bằng.

**Kiểm soát giả thuyết "batch lớn nuốt negative"** (nghi β/m_hardneg trơ vì bs16384 quá lớn, ≈16k in-batch
negative áp đảo 5-20 hard-neg). Lặp ở batch nhỏ hơn (test r@200):

| batch_size | mhn=0 | mhn=5 | mhn=10 | mhn=20 |
|---|---|---|---|---|
| 2048 | **.6420** | .6413 | .6404 | .6401 |
| 512  | **.6516** | .6498 | .6494 | .6484 |

→ **mhn=0 vẫn thắng/hoà ở MỌI batch size** ⇒ giả thuyết bị bác. Hard-neg không phải lever ở mọi quy mô đã thử.

**Vì sao β & m_hardneg trơ — 2 cơ chế:**

1. **Hard-neg là item SEEN → bị mask ở eval.** `hard_neg_ids` = `dropped ∪ (score 1-4)` = interaction user
   ĐÃ xem. Protocol eval mask toàn bộ `seen(user) − query` khỏi candidate (`metrics.py`). Nên đúng những
   item model học để dìm xuống lại **không bao giờ là candidate lúc chấm** → công học không chuyển thành
   recall/ndcg warm. (Hard-neg dạy "phân biệt đã-thích vs đã-bỏ" — nhưng metric retrieval không đo việc đó.)
2. **β là no-op tuyệt đối khi m_hardneg=0.** Trong `loss.py`, logit hard-neg = `s_hn + log(β)`; khi mhn=0
   nhánh hard-neg bị mask hết về −inf → `log(β) + (−inf) = −inf` bất kể β (kiểm chứng: cặp run bs512 mhn=0
   cho kết quả byte-identical dù đổi β). Khi mhn>0, β chỉ dịch một hằng số nhỏ trên vài logit gần như không
   đóng góp vào mẫu số InfoNCE (bị ~B in-batch negative áp đảo) → tác động chìm dưới noise.

**Kết luận**: β và m_hardneg **không phải đòn bẩy** cho bài toán + protocol này. **GIỮ NGUYÊN công thức loss
+ kiến trúc** (chỉ ghi nhận empirical, không đề xuất gỡ nhánh hard-neg); final dùng m_hardneg=5 / β=1 như
xuyên suốt — vì trơ nên giá trị cụ thể không ảnh hưởng.

## 5. Quy trình end-to-end

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

## 6. Đọc số khi viết báo cáo

- **Bảng so config**: `runs.csv` (cột knob + `{val,test}_{recall,ndcg}@K`). Tách coarse (subset, có
  `train_user_frac`) vs confirm (full) — chỉ kết luận final từ **confirm trên full data**.
- **Đường cong**: `runs/v5/<run>/history.json` (loss + val metric theo step) → notebook cell 7-8.
- **Cold (anime mới)**: notebook cell 10 (`split='val'` khi tune; `test_cold` = final exam, 1 lần).
- **Provenance đầy đủ 1 run**: `runs/v5/<run>/config.json`. Bar baselines: `RESULTS.md` / `BASELINES.md`
  (baselines đã chốt 2026-06-17 — itemknn K=50, content IDF, MF per-axis f128, +liked-metric).
