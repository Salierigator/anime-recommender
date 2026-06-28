# Model (Two-Tower Retrieval)

Doc tổng hợp cho `retriever/src/` — kiến trúc + training + **protocol eval**. Ăn artifacts `retriever/train-data/` (xem `docs/TRAIN_DATA.md`; split + support/query: `docs/DATA_SPLIT.md`). Baselines so sánh: `docs/BASELINES.md`.

> Config đã **chốt `final` (no synopsis, 2026-06-17)** và **re-export xong** — `best.pt`/`artifacts/` =
> `final` (serve-path official [RESULTS.md §3b](RESULTS.md)); ranker cũng đã retrain trên pool `final`
> (2026-06-18) → pipeline đồng bộ. Giá trị config chốt: §6 (khối "Config `final`"). Số mới nhất + tổng hợp + nguồn từng số: [RESULTS.md](RESULTS.md). Kiến trúc/protocol ổn định.

---

## 1. Overview

**Mục tiêu**: học 2 tower (user/item) sao cho `cosine(U, V⁺)` cao cho cặp positive; serve = brute-force top-K trên ~22.8k item. Headline tuning = **warm recall@200** (val); final exam = **cold slice** (test_cold — gợi ý anime mới chỉ bằng content, id→OOV).

**Smoke** (50k ex, 1 epoch, MPS): loss 6.80; warm val r@200 0.011 (random-init) → 0.239 sau 97 step; cold-eval máy móc chạy (8.4k user / 150k pairs). Epoch full 67.5M @ bs8192 ≈ 8.2k step.

## 2. Files — `retriever/src/` + `tests/`

```
src/
├── config.py     # TwoTowerConfig — TOÀN BỘ knob (bảng §6)
├── data.py       # ItemTable + UserTable (history RAGGED full) + eval_seen/cold_mask
│                 # + ExamplesDataset (resample per-user cap) + Collate (sample history)
├── model.py      # ItemTower / UserTower / TwoTower (+cold_oov cache, history_source)
├── loss.py       # info_nce_logq (+logq_alpha)
├── metrics.py    # protocol eval: build_masks / load_eval_protocol / evaluate / run_cold_eval
└── train.py      # fit(cfg): loop + eval warm + checkpoint best theo recall@headline_k
tests/            # pytest invariants — KHÔNG cần train-data (fixtures synthetic)
```

Convention: import flat, CWD = `retriever/src/`. Tests: `venv/bin/python -m pytest retriever/tests -q`.

## 3. Kiến trúc

2 tower → `d=128`, L2-norm → score = cosine. MLP mỗi tower: `Linear(in→256) → ReLU → Linear(256→128)`. Vocab/dim đọc từ `feature_spec.json`.

- **ItemTower**: 6 cat/bucket emb (28) ⊕ genres Linear(22→8) ⊕ themes Linear(53→8) ⊕ studios Emb(302,16) masked-mean (16) [⊕ anime-id Emb(N,id_dim) nếu `use_item_id`] → MLP → V[128]. **id-dropout**: train, prob `id_dropout`, real idx≥2 → OOV (dạy backoff cold + giữ content path sống).
- **UserTower**: pooled-history (128) ⊕ gender Emb(4,4) ⊕ joined Emb(5,4) → MLP → U[128]. `h_empty` learned thay pooling khi history rỗng. Không user-id (cold-by-user).
- **Pooling history** (`history_pool`): `mean` (± `score_pool` linear/learned) | `attn` (learned-query attention). Kết quả đo: score_pool trung tính, attn không cải thiện warm và **regress cold mạnh** (cold ndcg@10 ~.066 vs mean ~.12) khi vec history bị detach (xem `history_source`) — số đầy đủ warm+cold: `docs/EXPERIMENTS.md §5.1`.
- **`history_source`**: `cache` = lookup `item_cache` **detach** (user side không có grad về biểu diễn history); `embed` = bảng `nn.Embedding(num_items, d, padding_idx=0)` **trainable** riêng cho phía history → gradient chảy qua đường history. Nếu chốt `embed`: export phải đóng gói thêm `hist_emb` + service đổi cách pool.
- **Item-vec cache**: refresh đầu epoch + mỗi `cache_refresh_steps`; eval dùng cache. **`refresh_item_cache(cold_mask=...)`**: row thuộc H encode id→OOV (content thật) — bắt buộc cho cold eval.

## 4. Batch (collate)

Mỗi example `(user_idx, pos)`. Batch B:
- **History**: sample `train_hist_len` (config `final`: 128) vị trí từ **full list** của user (with-replacement khi list > L — dup nhẹ chấp nhận để vectorize; ngắn hơn → lấy hết + pad); gỡ anchor bằng mask; `hist_dropout` (12%) bỏ trọn history → h_empty. Mỗi step thấy tổ hợp history khác nhau (augmentation), item đuôi của heavy user cũng được vào history.
- **Hard-neg**: sample m phân biệt từ `hard_neg_ids` (−H) của chính user; thiếu → PAD + mask; 0 → loss thuần in-batch.
- **`max_examples_per_user`** (None=off): mỗi epoch resample ≤C example/user (lexsort vectorized, tất định theo epoch) — align trọng số train (∝n_pos, mean ~270) với eval (đều per user), epoch ngắn ~4×.

## 5. Loss

InfoNCE + logQ + τ + **`logq_alpha`**: `s_in − α·logq[pos]` (α=1 full, 0 tắt — knob cân popularity vì metric thưởng popularity còn logQ chống). Hard-neg không logQ, nhân β. Mask false-negative + mask pad → −inf.

> Ablation các tham số loss (`docs/EXPERIMENTS.md §4`): **logQ α là lever quan trọng nhất** (α=0 → sụp); τ tác dụng nhỏ (.07 best); **β và m_hardneg đo ra TRƠ** (không phải lever — hard-neg là item seen nên bị mask ở eval; β no-op khi m_hardneg=0). Đây chỉ là kết luận empirical — **công thức + nhánh hard-neg giữ nguyên**.

## 6. Config surface (`TwoTowerConfig` — đổi từ notebook, không hardcode)

| Nhóm | Field (default) |
|---|---|
| Model | `d`(128), `mlp_hidden`([256]), `use_item_id`(F), `id_dim`(64), `id_dropout`(.2), `history_pool`(mean), `score_pool`(none), **`history_source`(cache\|embed)** |
| Loss | `tau`(.07), `beta`(1.0), **`logq_alpha`(1.0)** |
| Train | `lr`(1e-3), `cosine_lr`(F — bật = cosine-anneal lr về 0 theo tổng step, đáng bật khi train nhiều epoch), `weight_decay`(0), `batch_size`(4096), `epochs`, `hist_dropout`(.12), `m_hardneg`(3), **`train_hist_len`(32)**, **`max_examples_per_user`(None)**, `cache_refresh_steps`(300) |
| Eval | **`eval_ks`([10,50,100,200,500])**, **`headline_k`(200)**, **`eval_history_cap`(1024)**, `eval_split`(val), `eval_every_steps`(0) |

(Số trong ngoặc = **default của `TwoTowerConfig`**, KHÔNG phải config chốt.)

**Config `final` (chốt 2026-06-17 — đè default ở trên):** `d=128`, `mlp_hidden=[256]`,
`use_item_id=True`, `id_dim=128`, `id_dropout=0.15`, `history_pool=mean`, `score_pool=none`,
**`history_source=embed`**, `train_hist_len=128`, `tau=0.07`, `beta=1.0`, `logq_alpha=1.0`,
`optimizer=adam`, `lr=1e-3`, `weight_decay=0`, `batch_size=16384`, `epochs=10`, `hist_dropout=0.12`,
`m_hardneg=5`, `eval_history_cap=1024`, `headline_k=200`, **`use_synopsis=False`**. Best step =
**31500** (epoch 7). Nguồn: `retriever/runs/runs.csv` row `final` + `artifacts/CONTRACT.md`. Vì sao
các lựa chọn này (logQ α=1, τ.07, β/m_hardneg trơ, synopsis OFF): [EXPERIMENTS.md](EXPERIMENTS.md)
+ [SYNOPSIS_EMB.md](SYNOPSIS_EMB.md).

## 7. Protocol eval (`metrics.py`) — phần quan trọng nhất

> Định nghĩa chuẩn support/query/seen-mask/trần recall@K: [DATA_SPLIT.md §4–8](DATA_SPLIT.md) (owner).
> Phần dưới mô tả cách `metrics.py` hiện thực protocol đó cho retriever.

1. **Seen-mask**: mask = `seen(user) − query_đang_chấm` (`build_masks`; seen từ `eval_seen.parquet`, MỌI status kể cả PTW — khớp serving filter cả list). Query ⊆ seen nên tuyệt đối không mask thẳng seen.
2. **2 slice**:
   - **Warm** (`examples val/test`): tuning + chọn checkpoint (`recall@headline_k` trên val). Cache warm (id thật).
   - **Cold** (`examples {val,test}_cold`): `run_cold_eval` — refresh cache `cold_mask` (H → id OOV, content thật) rồi rank **full-catalog**; thêm chế độ `h_only` (candidate chỉ H — diagnostic content thuần, tách khỏi cạnh tranh warm-vs-cold). **test_cold = final exam, chấm 1 lần lúc cuối**; val_cold để debug. KHÔNG nằm trong train loop.
3. **Metrics**: recall@K/ndcg@K mean-per-user; cold thêm **pooled hitrate@K** (= tổng hit / tổng pairs — slice mỏng nên per-user noisy) + `n_users`/`n_pairs`. History eval = prefix `eval_history_cap` (list đã sort score desc). Noise floor: chênh <~0.004 r@100 trên ~14k user = không kết luận. **liked-metric** (report-only): khi truyền `query_scores`, thêm `liked_recall@K`/`liked_ndcg@K` trên cùng ranking, chỉ tính query "liked" (score≥1 & > mean support của user) — định nghĩa + lý do [METRIC.md](METRIC.md).
4. Baselines (`retriever/baselines/`) dùng đúng harness này (`_eval.py` mirror `metrics.evaluate`): random/popular/content/itemknn/mf (warm) + random/content/**meta_popular** (cold); popular/mf/itemknn cold = N/A by construction (không score được item ngoài train — ghi rõ trong .txt, không bịa số).

## 8. Invariants padding/masking/OOV (test ở `retriever/tests/`)

| Invariant | Test |
|---|---|
| PAD=0 (padding_idx, zeros) / OOV=1 (HỌC được) / real ≥2; examples ≥2 | prep `99` |
| History pad=0 + mask; gỡ anchor; rỗng → h_empty (mọi pool); attn toàn-pad NaN-safe | `test_collate`, `test_model` |
| Sample history nằm đúng slice user (không tràn offsets) | `test_collate` |
| Studios: id 0 = empty-token học được khi row toàn 0; 0 lẻ trong row non-empty = pad bị mask | `test_model` |
| id-dropout chỉ train + idx≥2 + candidate path; eval/cache dùng id thật | `test_model` |
| `cold_oov` cache: row H == encode OOV ≠ warm; row warm không đổi | `test_model` |
| `cache` không grad về item tower qua history; `embed` có grad (padding_idx=0 không grad) | `test_model` |
| Loss: false-neg mask, pad hard-neg −inf, `logq_alpha` đúng nghĩa, không NaN | `test_loss` |
| Eval: seen bị mask, query KHÔNG bị mask, pooled hitrate đếm đúng, candidate_mask H-only | `test_metrics` |

## 9. Chạy

```bash
venv/bin/python -m pytest retriever/tests -q          # invariants
cd retriever/src && ../../venv/bin/python train.py --smoke   # end-to-end nhanh
# train thật: train.ipynb trên Colab — train-data trên Drive phải khớp bản local
venv/bin/python retriever/export.py && venv/bin/python retriever/test_export.py  # → artifacts/
```

## 10. Export → `artifacts/` (firewall)

`retriever/export.py` biến `checkpoints/best.pt` + `train-data/` thành bộ file ổn định trong `artifacts/` cho ranker + service (re-run mỗi khi best.pt đổi; schema chi tiết: `artifacts/CONTRACT.md` tự sinh):

- **`item_vectors.npy`** `[N,128]` — chạy item-tower qua serve-path cache: **row item cold (H) encode id→OOV** (content thật, khớp đúng cách đã đo cold eval); row warm id thật.
- **`item_index.parquet`** (anime_idx → mal_id + `is_cold`), **`user_split.parquet`**.
- **`user_tower.pt`** — chỉ user-side state_dict (lọc bỏ `item_tower.*`) + đủ cfg/spec để service dựng lại `UserTower` không cần ItemTable; nếu `history_source=embed` thì đóng gói kèm bảng `hist_emb.*`.
- **Eval protocol cho ranker**: `eval_queries_{val,test,val_cold}.parquet` + `eval_seen.parquet` + `users_history.parquet` (MỌI user, history FULL — ranker không stream ratings.csv). `eval_queries_test_cold` (final exam) CHỈ export khi `--final-exam`, default xoá file cũ để harness ranker không chấm nhầm.
- `reconcile_spec_to_ckpt()`: nếu spec local đã regenerate khác lúc train (vd vocab joined đổi) → ép vocab theo shape checkpoint, in cảnh báo (không nuốt im).

`retriever/test_export.py` validate firewall-faithful: dựng user-encoder **thuần từ artifacts** (user_tower.pt + item_vectors.npy — KHÔNG load best.pt để encode, đúng như service sẽ làm), assert invariants các file ranker (test_cold vắng mặt, users_history phủ đúng user_split, query∩history=∅, query⊆seen, cold query đều `is_cold`, history sort desc), rồi chấm lại đủ protocol warm val/test + val_cold và ghi **`eval_reference.json`** — mốc cho sanity gate của ranker (`ranker/eval.py --baseline-only`).

### 10.1 Hai bộ số của CÙNG model `final` — đọc cho đúng (quan trọng)

Cùng một retriever `final` được đo bằng **hai đường khác nhau**, cho hai bộ số — đừng trộn lẫn:

| | Checkpoint-path (đo lúc train) | Serve-path (qua artifacts) — **OFFICIAL** |
|---|---|---|
| Nguồn | `retriever/runs/runs.csv` (đo trong lúc train) | `artifacts/eval_reference.json` (`test_export.py` re-encode thuần từ artifacts) |
| Encode | UserTower in-training | user-encoder dựng lại từ `user_tower.pt`+`item_vectors.npy` (item cold = id→OOV), đúng đường service chạy |
| test_warm recall@200 | .6852 | **.6758** |
| test_warm ndcg@10 | .4242 | **.5323** |

- **Số OFFICIAL = serve-path** (`eval_reference.json`): đây là số hệ thống thật tạo ra lúc serve, và cũng
  là **baseline cosine + trần recall@200 = .6758** mà ranker kế thừa (xem [RESULTS.md §3b/§6](RESULTS.md)).
- recall@200 hai đường gần khớp (.6852 vs .6758). **ndcg@10 chênh ngược chiều, lớn hơn** (.4242 <
  .5323) — cơ chế (xem [RESULTS.md §2](RESULTS.md)): `final` train `history_source=embed`, lúc eval-train
  (checkpoint-path) 1.142 row cold-H encode bằng **id thật chưa train = vector noise** → một số lọt lên
  **đầu** ranking làm distractor → dìm ndcg@10; serve-path encode H bằng **id→OOV (content)** bớt distractor
  đầu bảng → ndcg@10 phục hồi (.5323), còn recall@200 hơi giảm (content-H là distractor "hợp lý").
- **Hệ quả thực tế: so sánh model dùng số SERVE-PATH / POOL** — chúng cùng một thang đo ở **K≤200**
  (baseline cosine trùng tới ~1e-15 giữa serve-path full-catalog `eval_reference.json` và pool
  `ranker_meta.json::baseline_test`; by construction top-K ⊆ pool — [RESULTS.md §1](RESULTS.md)). Vì vậy
  baselines retriever ([BASELINES.md](BASELINES.md)), retriever serve-path (§3b / `eval_reference.json`) và
  two-stage + cosine pool ([RESULTS.md §6](RESULTS.md)) **so trực tiếp được** ở head/mid. Số
  **checkpoint-path** (`runs.csv`, vd ndcg@10 .4242) chỉ để so run-vs-run lúc train — **KHÔNG** đem so với
  baseline/pool. So two-stage vs MF hợp lệ ở K≤200; khác biệt *cơ sở* chỉ ở **recall@200+** (MF retrieve
  full catalog vs two-stage kẹt trần pool .6758 = candidate-generation, việc của retriever).
