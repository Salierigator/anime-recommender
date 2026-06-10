# Model (Two-Tower Retrieval)

Doc tổng hợp cho `retriever/src/` — giải thích **build cái gì, set up thế nào, vì sao**. Dựng kiến trúc Two-Tower (PyTorch) + training pipeline ăn artifacts `retriever/train-data/` (xem `docs/TRAIN_DATA.md`).

---

## 1. Overview

**Mục tiêu**: stage Retrieval — học 2 tower (user / item) sao cho `cosine(U, V⁺)` cao cho cặp positive, dùng để ANN top-K từ ~22.8k item. Headline metric = **cold-by-user** (hold-out trọn user).

**Philosophy** (khớp `TRAIN_DATA.md §2`): data artifact (cố định, frozen ở `retriever/train-data/`) vs model parameter (học, nằm trong checkpoint). Vocab/dim/special-idx đọc từ `retriever/train-data/feature_spec.json` — **không hard-code** trong model. Tham số học: ma trận embedding + 2 Linear chiếu genres/themes + h_empty + MLP 2 tower. **τ (0.07) và β (1.0) là hyperparam cố định** trong `config.py`, không phải `nn.Parameter`.

**Kết quả smoke** (50k example, 1 epoch, MPS): loss 8.12 → 6.58; eval cold-by-user chạy, checkpoint OK.

**Throughput** (Mac MPS, bs=4096): ~66k ex/s → **~16′/epoch** (65.3M example); cache refresh ~7ms; eval <1s. Colab T4 ~10–18′, L4/A100 ~4–8′. Model compute-light → cổ chai dễ ở collate/transfer (đặt `num_workers=2–4`), không phải GPU.

---

## 2. Files — `retriever/src/`

```
retriever/
├── src/
│   ├── config.py     # TwoTowerConfig: mọi hyperparam (d, tau, beta, lr, batch, hist_dropout, m_hardneg…)
│   ├── data.py       # load artifact 1 lần vào RAM + ExamplesDataset + collate (history/hard-neg/mask)
│   ├── model.py      # ItemTower, UserTower, TwoTower (+ item-vec cache cho nhánh history)
│   ├── loss.py       # info_nce_logq: InfoNCE + logQ + temperature + hard-neg + 2 loại mask
│   ├── metrics.py    # evaluate cold-by-user: recall@K, ndcg@K
│   └── train.py      # fit(cfg) / build(cfg) + loop + eval + checkpoint; CLI --smoke
└── train.ipynb       # notebook điều khiển: import → config → build → train (tune hyperparam)
```

Convention path: `.py` trong `src/` dùng `ROOT = Path(__file__).resolve().parent.parent` (= `retriever/`); artifacts ở `ROOT/"train-data"`. Import flat → **chạy với CWD = `retriever/src/`** (`import config, data, model as M, loss, metrics, train`); notebook (`retriever/train.ipynb`) tự thêm `sys.path` tới `src/`.

---

## 3. Kiến trúc

Cả 2 tower → vector `d=128`, **L2-normalize** (→ score = cosine). MLP mỗi tower (`_mlp`, hidden=`[256]`) = `Linear(in→256) → ReLU → Linear(256→128)` — **1 hidden layer 256, ReLU, không BatchNorm/dropout**. Vocab/dim mỗi feature đọc từ `feature_spec.json`, không hard-code.

### 3.1 Tổng thể

```
            ITEM TOWER                                    USER TOWER
  anime_idx ─► gather 9 feature              history_ids ─► item_cache lookup (detach, no-grad)
   ├ 6 cat/bucket emb (pad_idx=0) ── 28        └► masked-mean ──────────── 128   (rỗng → h_empty)
   ├ genres  Linear(22→8) ──┐                 gender_id ─► Embedding(4,4) ── 4
   ├ themes  Linear(53→8) ──┼── 16            joined    ─► Embedding(5,4) ── 4
   ├ studios Emb(302,16)→mean ── 16
   └ anime_id Emb(N,64,pad0) ── 64*           *use_item_id (mặc định off);
        concat ────────── 60 (+64 id)          train: id-dropout 0.2 → OOV
            │                                            │
  Linear(60|124→256)→ReLU→Linear(256→128)     Linear(136→256)→ReLU→Linear(256→128)
            │                                            │
        L2-normalize                                 L2-normalize
            ▼                                            ▼
          V [128] ───────────►  s = (U · V) / τ  ◄─────────── U [128]
                              (U,V đã norm → dot = cosine)
```

### 3.2 ItemTower (`model.py`)

`anime_idx → gather 9 feature từ ItemTable (+ id-emb optional) → concat 60|124 → MLP → L2-norm → V[128]`.

| Nhánh feature | Layer | Out dim |
|---|---|---|
| type / source / rating / demographics / start_year / episodes | 6×`nn.Embedding(vocab, dim, padding_idx=0)` — dim **4 / 8 / 4 / 4 / 4 / 4** | **28** |
| genres (multi-hot, width 22) | `nn.Linear(22 → 8)` | 8 |
| themes (multi-hot, width 53) | `nn.Linear(53 → 8)` | 8 |
| studios (multi-value) | `nn.Embedding(302, 16)` (**không** padding_idx) → **masked-mean**; id 0 = `empty_id` (studio rỗng, ~28% anime — **học được**, không zeros), pad cấu trúc khử bằng "row toàn 0" rồi mask | 16 |
| **anime_id** (`use_item_id`, mặc định off) | `nn.Embedding(num_items, id_dim=64, padding_idx=0)` — PAD=0 zeros (no grad), OOV(1) & real(2..) **học**. Bắt collaborative residual content không diễn tả nổi (2 anime trùng feature ≈ vector khác nhau theo audience) | 64 |
| **concat** | | **60** (off) / **124** (on) |
| MLP | `Linear(60\|124 → 256) → ReLU → Linear(256 → 128)` | 128 |
| output | `F.normalize` (L2) | **V [128]** |

- **id-dropout** (`id_dropout=0.2`): lúc train, mỗi real item (`idx≥2`) có prob 0.2 bị mask id→OOV(1) trong nhánh id — **content vẫn gather bằng id thật**. Mục đích kép: (1) dạy vector OOV làm backoff cho anime mới, (2) regularize ép nhánh content luôn tiên đoán được (không thì model dồn hết qua id → content teo → serve anime mới ra rác). Song song `hist_dropout`. Chỉ áp ở candidate path (`encode_items`, có grad); `item_cache` (eval + history pooling) **luôn dùng id thật** (warm).
- **Vì sao item-id GIỮ còn user-id DROP**: cold-by-user hold-out trọn **user** nhưng KHÔNG hold-out **item** → mọi item lúc eval đều warm → id-emb dùng được, metric thưởng. (User-id lạ lúc eval → vô dụng + ăn gian, nên drop — xem §3.4.)

### 3.3 UserTower (`model.py`)

`pooled_history ⊕ gender ⊕ joined → concat 136 → MLP → L2-norm → U[128]`.

| Nhánh feature | Layer | Out dim |
|---|---|---|
| history | pool cached item-vec (detach) theo `history_ids`, masked; cách gộp = `history_pool` (**mean** mặc định, ±`score_pool`; hoặc **attn** = learned-query attention, xem dưới); row rỗng → `h_empty` | 128 |
| gender | `nn.Embedding(4, 4, padding_idx=0)` | 4 |
| joined | `nn.Embedding(5, 4)` (**không** padding_idx; NULL gộp vào cohort mới nhất 2022+ ở 04 nên cả 5 bucket đều học được, khác `gender` vẫn giữ OOV(0)) | 4 |
| **concat** | | **136** |
| MLP | `Linear(136 → 256) → ReLU → Linear(256 → 128)` | 128 |
| output | `F.normalize` (L2) | **U [128]** |

- `h_empty` = `nn.Parameter[128]`, init `N(0, 0.02²)`, **learned** (KHÔNG zeros) — là output nhánh history khi rỗng (thay pooling chia-cho-0), rồi concat với gender/joined như bình thường.
- **Không có user-id embedding** — drop ở v1: cold-by-user hold-out trọn user → id user lạ lúc eval vô dụng; tệ hơn, id cho model "ăn gian" memorize user → giảm áp lực lên nhánh history → hỏng đúng metric.
- **Weighted history pooling** (`score_pool`, mặc định `none`): thay masked-mean bằng trung bình **có trọng số theo điểm** user chấm cho item trong history (`history_scores`, có sẵn trong artifact, align `history_ids`). Mean coi fan 10/10 = phim bỏ dở 6/10 như nhau → weighting sharpen user-vec về đúng gu. 3 mode:
  - `none` — masked-mean (như cũ).
  - `linear` — fix cứng `w = score.clamp(min=1)` (chưa-chấm 0→1, 10→10).
  - `learned` — `w = softplus(Embedding(11,1)[score])`, trọng số dương **học** per mức điểm 0..10 (11 bucket; pad bị mask nên không cần padding_idx).

  Weighted-mean = `Σ(mask·w·v) / Σ(mask·w)`; softplus/clamp ⇒ w>0 ⇒ không chia 0; row rỗng vẫn → `h_empty`. **Vì sao learned hợp hơn**: điểm lệch mạnh (77.6% là 9–10, 9.36% chưa chấm) → `linear` gần vô tác dụng ở khối 9–10 (10 vs 9 = 1.11×); `learned` để data tự quyết. Trọng số per-bucket **chung mọi user** (per-user normalization để sau). Áp cho cả train (collate) lẫn eval (`metrics.py`).

  **Kết quả đo (v3)**: `score_pool` trung tính — ep6 `learned` test r@100 0.3964 ≈ `none` 0.3965; ep1 `linear` 0.3735 < `learned` 0.3769 ≈ `none`. Tức cho tự do thì model dẹt trọng số về ≈ mean: **độ lớn điểm (giữa các phim đã-thích) không thêm tín hiệu retrieval** ngoài membership. → chuyển sang reweight theo **nội dung** (`history_pool='attn'`).
- **Attention pooling** (`history_pool`, mặc định `mean`): `attn` thay masked-mean bằng **learned-query attention** trên history — 1 query học `q[128]` + key-proj `Linear(128,128,bias=False)`, `attn = softmax((W·v)·q / √128)` (mask pad = −inf; row rỗng fill logits 0 cho NaN-safe), `pooled = Σ attn·v` (**value = chính item-vec** → vẫn tổ hợp lồi cùng không gian ⇒ so trực tiếp với mean = uniform). Query init **unit-scale** (`/√d` ⇒ logits ~unit-var, attend thật từ step 0, không khởi-tạo-bằng-mean). `item_cache` vẫn **detach** (query/key học, item-vec đóng băng). **Bỏ qua `score_pool`** (attention chính là cách gộp). Động cơ: cold-by-user nghẽn ở user-vec = 1 centroid mean; `score_pool` đã chứng minh reweight theo **điểm** vô tác dụng → `attn` reweight theo **nội dung** (item nào đáng tóm tắt), trục khác hẳn.

### 3.4 Item-vec cache

`TwoTower.refresh_item_cache`: nhánh history pool từ bảng item-vec **cache** (`item_cache [num_items, 128]`, detach, no grad), refresh dày (đầu mỗi epoch + mỗi `cache_refresh_steps=300`). 22.8k item recompute ~7ms → staleness ~0. Candidate path (positive + hard-neg) tính **fresh, có grad** mỗi step.

---

## 4. Cơ chế batch (`data.py` collate)

Mỗi example = `(user_idx, pos_item)`. Batch B → B user, B positive.
- **History**: `history = stored_history_ids − {pos_item}` (gỡ anchor) rồi pool; rỗng → `h_empty`.
- **History dropout ~12%**: random bỏ toàn bộ history (dùng `h_empty`) cho một phần example/step → supervise `h_empty` + regularize + đẩy thẳng cold-by-user.
- **In-batch negative** = B−1 positive của anchor khác (free, từ `U @ V_posᵀ`, ma trận vuông không cần pad).
- **Hard-neg per-anchor**: sample m≈3 item **phân biệt** (without-replacement) từ `hard_neg_ids` của *chính user đó*. Dựng `[B, m]` + **mask**:
  - lens < m → phần dư là **PAD** (id 0) + mask False (vd 1 dropped → `[Y1,PAD,PAD]`, mask `[1,0,0]`).
  - user 0 dropped → toàn mask → **loss thuần in-batch** (đừng bịa hard-neg).
  - **KHÔNG** gộp hard-neg thành pool chung: "negative khó riêng của user A" mất nghĩa nếu áp cho B.

---

## 5. Luồng data qua model (forward → cosine)

1 batch B example chạy qua `TwoTower.forward(batch)` → ra 2 ma trận score. Input tensor (collate §4 dựng):

- `pos [B]`, `hardneg_ids [B,m]`, `hardneg_mask [B,m]`
- `history_ids [B,30]`, `history_mask [B,30]` (đã gỡ anchor + bỏ pad + áp dropout), `history_scores [B,30]` (align `history_ids`, cho weighted pooling `score_pool`; `attn` không dùng)
- `gender_id [B]`, `joined_bucket [B]`

**Item side** (fresh, **có grad** — đây là path được supervise):
1. `V_pos = encode_items(pos)`: mỗi `anime_idx` gather 9 feature từ `ItemTable` → ItemTower → `[B, 128]`.
2. `V_hn = encode_items(hardneg_ids)`: flatten `[B·m]` → ItemTower → reshape `[B, m, 128]`.

**User side**:
3. `pool_history(history_ids, history_mask, history_scores)`: lookup `item_cache[history_ids]` (**detach**, không grad qua history) → `[B, 30, 128]` → gộp theo `history_pool`: **masked-mean** (±**weighted theo điểm** nếu `score_pool≠none`) hoặc **attention** (`attn`), §3.3 → `[B, 128]`; row nào mask rỗng → thay bằng `h_empty`.
4. `user_tower(pooled, gender_id, joined_bucket)`: concat `[B, 136]` → MLP → L2-norm → `U [B, 128]`.

**Tính cosine để so độ khớp** (U, V đã L2-norm → dot product = cosine), chia temperature τ≈0.07:
5. **In-batch**: `s_in = (U @ V_posᵀ) / τ → [B, B]`. Entry `(i,j)` = `cos(Uᵢ, V_pos_j)/τ`; **đường chéo** = positive của chính anchor i, **off-diagonal** = in-batch negative (positive của anchor khác — free, không cần encode thêm).
6. **Hard-neg per-anchor**: `s_hn = (U.unsqueeze(1) · V_hn).sum(-1) / τ → [B, m]`. Entry `(i,k)` = `cos(Uᵢ, hard-neg_k của chính user i)/τ`.

→ `s_in`, `s_hn` là đầu vào trực tiếp của Loss (§6): ghép thành `[B, B+m]`, áp logQ + 2 loại mask rồi cross-entropy với target = đường chéo.

---

## 6. Loss (`loss.py` — info_nce_logq)

InfoNCE + **logQ correction** + temperature. `s(i,x) = (U_i·V_x)/τ`, U,V đã norm, τ≈0.07.

$$\mathcal{L}_i = -\log \frac{\exp(s(i,i^{+}) - \log Q(i^{+}))}{\sum_{j}\exp(s(i,j) - \log Q(j)) + \beta\sum_{k}\exp(s(i,k))}$$

`logits = cat([s_in (B cột), s_hn (m cột)]) → [B, B+m]`, target = đường chéo. Bốn điểm bắt buộc:
- **logQ** (`retriever/train-data/logq.npy`, popularity item-as-positive) trừ vào in-batch (gồm positive). Thiếu → retriever đè item phổ biến bệnh lý.
- Hard-neg **KHÔNG** logQ (sample từ phân phối cố ý giữ "khó"); nhân β (cộng `log β`), β=1 start.
- **Mask false-negative**: 2 anchor trùng `pos_item` → off-diagonal đó set −∞ (đừng tự phạt positive của mình).
- **Mask pad** hard-neg → −∞ (`exp(−∞)=0`, ô PAD biến mất khỏi mẫu số). Không NaN: mẫu số luôn còn positive + in-batch.

---

## 7. Eval — cold-by-user (`metrics.py`)

User eval: build `U` từ history (support; query items KHÔNG nằm trong history → không leak, đã assert ở pipeline) → score vs toàn `item_cache` → **mask non-candidate** (`logq=-inf`) + **mask item đã seen** → top-K → **recall@K / ndcg@K** (K=10/50/100) so với query items. Full val (13.6k user) ~0.7s.
