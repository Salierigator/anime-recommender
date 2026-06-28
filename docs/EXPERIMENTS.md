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

## 5. Ablation kiến trúc & regularization (checkpoint-path test — coarse)

Các knob kiến trúc/regularization còn lại (pooling, id_dropout, MLP width, `d`, optimizer, epochs/history_source)
được thăm dò chủ yếu bằng **random-search** (cell 6b, subset 15% user, ep2 trừ khi ghi khác). Số WARM = cột
`test_*` của `runs.csv` (**checkpoint-path**, cùng đường đo §4 — KHÔNG trộn serve-path `RESULTS.md §3b`); số
COLD = `cold_runs.csv` (val_cold, 8.388 user). Cột bảng: warm `r@100 / r@200 / ndcg@10 / liked_ndcg@10`
‖ cold `r@200 / honly_r@200 / ndcg@10`.

> ⚠️ **Đọc đúng mức**: phần lớn run là điểm random-search, **nhiều knob đồng biến** (đổi optimizer thường đổi
> kèm width/α/wd) → KHÔNG phải A/B một-yếu-tố. Bảng nào **cô lập sạch** được đánh dấu *(sạch)*; bảng confounded
> đánh dấu *(thăm dò)* — chỉ đọc xu hướng, không kết luận đơn yếu tố. Anchor để định cỡ: `final`
> (warm checkpoint-path ndcg@10 **.4242** / r@200 **.6852** — ndcg thấp là đặc thù H-noise của `history_source=embed`,
> serve-path thật .5323, xem `RESULTS.md §2/§3b`); `v5_hist64_ep2` (.5135 / .6608).

### 5.1. Pooling history (`attn` vs `mean`) + `score_pool` — *(cụm gần nhau warm; cold tách rõ)*

Mọi run hl256/embed/uf015/ep2 (τ/m_hardneg đồng biến nhẹ giữa các dòng):

| history_pool · score_pool | warm r@100 | r@200 | ndcg@10 | liked_ndcg@10 | cold r@200 | honly_r@200 | cold ndcg@10 |
|---|---|---|---|---|---|---|---|
| **attn** (mhn5 τ.005, best attn) | .4950 | .6293 | .5075 | .3644 | .2884 | .7936 | **.0662** |
| **attn** (mhn10 τ.005) | .4950 | .6293 | .5036 | .3632 | .3031 | .7968 | **.0702** |
| **mean · none** (mhn10 τ.007) ★ | .4981 | .6313 | .5036 | .3627 | .4109 | .8304 | .1236 |
| mean · linear (mhn5 τ.005) | .4914 | .6243 | .5060 | .3800 | .4047 | .8240 | .1100 |
| mean · learned (mhn5 τ.005) | .4977 | .6310 | .4994 | .3613 | .4071 | .8251 | .1141 |

**Đọc**: warm gần như đồng đều (ndcg@10 .49–.51 mọi cấu hình) → pooling **không phải lever warm**. Nhưng **cold
tách rõ**: `attn` sụp (cold ndcg@10 **.066–.070**, r@200 .29–.30) so với `mean` (.11–.12, r@200 ~.41) — learned-query
attention **co-adapt với history warm, hỏng backoff cold** (đúng quan sát "attn không cải thiện khi vec history bị
detach"). `score_pool` none/linear/learned tương đương cả warm lẫn cold → **trung tính**. → chốt **`history_pool=mean`,
`score_pool=none`** (đơn giản nhất, không thua warm, thắng cold). Đây là nguồn số cho `TWO_TOWER_MODEL.md`.

### 5.2. `id_dropout` (0.15 / 0.30 / 0.50) — *(sạch)*

Cô lập sạch: embed/hl128/uf015/ep2/bs16384, chỉ đổi `id_dropout`.

| id_dropout | warm r@100 | r@200 | ndcg@10 | liked_ndcg@10 | cold r@200 | honly_r@200 | cold ndcg@10 |
|---|---|---|---|---|---|---|---|
| **0.15** ★ | .4887 | **.6211** | **.5047** | **.3698** | **.3747** | .8130 | .1114 |
| 0.30 | .4799 | .6131 | .4807 | .3520 | .3511 | .8062 | .1225 |
| 0.50 | .4763 | .6076 | .4738 | .3498 | .3413 | .8139 | .1310 |

**Đọc**: warm **giảm đơn điệu** theo dropout (ndcg@10 .5047→.4738, r@200 .6211→.6076) — dropout id cao làm
**underfit id-path** warm. Cold **ngược chiều nhẹ**: cold ndcg@10 tăng (.1114→.1310) nhưng cold r@200 lại giảm
(.3747→.3413) — ép dùng content nhiều hơn giúp head cold chút nhưng tổng recall cold vẫn tụt. Net **0.15 thắng** (ưu
tiên warm + cold recall; final dùng 0.15).

### 5.3. MLP width (`hl`) + embedding dim (`d`) — *(width thăm dò; d cặp sạch)*

| run | warm r@100 | r@200 | ndcg@10 | liked_ndcg@10 | cold r@200 | honly_r@200 | cold ndcg@10 |
|---|---|---|---|---|---|---|---|
| hl32 (embed ep2) | .4656 | .5971 | .4838 | .3503 | .3085 | .7843 | .0425 |
| hl256 (embed ep2) | .4977 | .6309 | .5007 | .3605 | .4143 | .8333 | .1283 |
| **d128** hl256 ep8 ★ | .5267 | .6667 | .4882 | .3551 | .3836 | .8170 | **.1373** |
| d256 hl256 ep8 | .5274 | .6663 | .5211 | .3789 | .3934 | .8251 | .1086 |

**Đọc**: width hl32→hl256 nâng cả warm (ndcg@10 .4838→.5007) lẫn cold mạnh (cold ndcg@10 .0425→.1283) — "rộng hơn
tốt hơn" rõ, nhưng run hl256 đổi kèm `weight_decay` nên *(thăm dò)*, không đơn yếu tố. `d`: **cặp khớp** (ep8/hl256/embed/lr.0015)
→ d256 nhỉnh warm ndcg@10 (.5211 vs .4882), recall@200 ~ngang (.6663 vs .6667), **nhưng cold kém** (cold ndcg@10 .1086
vs **.1373**). → chốt **`d=128`**: cold tốt hơn + nhẹ/nhanh serve, warm recall ngang (head-precision là việc của ranker).

### 5.4. Optimizer (`adam` vs `adamw`) + `weight_decay` — *(thăm dò, confounded)*

Không có cặp khớp hoàn toàn — `optimizer` luôn đổi kèm width/α/wd, nên chỉ đọc xu hướng:

| run | warm r@100 | r@200 | ndcg@10 | liked_ndcg@10 | cold r@200 | cold ndcg@10 |
|---|---|---|---|---|---|---|
| cache · adam · α1 · hl256 · wd1e-5 | .4641 | .5998 | .4287 | .3077 | .4146 | .1323 |
| cache · adamw · α1 · hl96 · wd1e-5 | .4774 | .6112 | .4664 | .3373 | .3482 | .0982 |
| cache · adamw · α.75 · hl96 · wd1e-5 | .4691 | .6054 | .4429 | .3070 | .4124 | .1153 |
| embed · adam · α1 · hl32 · wd1e-5 | .4656 | .5971 | .4838 | .3503 | .3085 | .0425 |
| embed · adamw · α.75 · hl32 · wd0 | .4735 | .6083 | .4555 | .3173 | .3807 | .1172 |

**Đọc**: hướng **không nhất quán** — `adamw` nhỉnh ở nhóm cache hl96 (.4664 > cache adam hl256 .4287, nhưng width
khác), còn ở embed hl32 thì `adam` α1 (.4838) > `adamw` α.75 (.4555, nhưng α/wd khác). → optimizer/wd **KHÔNG phải
lever quyết định**; lực chi phối thật là `history_source` + epoch (§5.5). final dùng **`adam`** (mặc định) xuyên suốt.

### 5.5. Confirm runs — epochs + `history_source` (dẫn tới `final`) — *(thăm dò → chốt)*

Chuỗi confirm trên full data (bỏ subset), tăng dần epoch về `final` (ep10):

| run | warm r@100 | r@200 | ndcg@10 | liked_ndcg@10 | cold r@200 | honly_r@200 | cold ndcg@10 |
|---|---|---|---|---|---|---|---|
| embed ep2 (hist32) | .5125 | .6465 | .4639 | .3436 | .3914 | .8433 | .1498 |
| embed ep4 hl128 | .5411 | .6777 | .4552 | .3352 | .4359 | .8490 | .1694 |
| cache ep6 hl128 | .5355 | .6736 | .4462 | .3261 | .4153 | .8221 | .1697 |
| embed ep6 hl256 | .5284 | .6693 | .5042 | .3703 | .4264 | .8200 | .1120 |
| embed ep8 hl256 | .5267 | .6667 | .4882 | .3551 | .3836 | .8170 | .1373 |
| **`final` embed ep10** ★ | .5462 | **.6852** | .4242 | .3145 | **.4664** | .8234 | .1398 |

**Đọc**: warm r@200 (candidate-generation feeding ranker) **leo dần tới `final` ep10** (.6852, cao nhất) và cold r@200
cũng đạt đỉnh ở `final` (.4664). `embed` ≥ `cache` ở recall sâu. ndcg@10 checkpoint-path của `final` thấp (.4242) là
**đặc thù H-noise** của `history_source=embed` lúc eval-train — serve-path thật .5323 (`RESULTS.md §2/§3b`). → xác nhận
hướng chốt: **`history_source=embed`, train tới ep10**.

## 6. Quy trình end-to-end

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

## 7. Đọc số khi viết báo cáo

- **Bảng so config**: `runs.csv` (cột knob + `{val,test}_{recall,ndcg}@K`). Tách coarse (subset, có
  `train_user_frac`) vs confirm (full) — chỉ kết luận final từ **confirm trên full data**.
- **Đường cong**: `runs/v5/<run>/history.json` (loss + val metric theo step) → notebook cell 7-8.
- **Cold (anime mới)**: notebook cell 10 (`split='val'` khi tune; `test_cold` = final exam, 1 lần).
- **Provenance đầy đủ 1 run**: `runs/v5/<run>/config.json`. Bar baselines: `RESULTS.md` / `BASELINES.md`
  (baselines đã chốt 2026-06-17 — itemknn K=50, content IDF, MF per-axis f128, +liked-metric).
