# DATA_SPLIT.md — Split & định nghĩa support/query

> Mô tả cách chia data + protocol eval của retriever. Pipeline build chi tiết: `docs/TRAIN_DATA.md`. Model + metrics: `docs/TWO_TOWER_MODEL.md`. Số liệu thực tế: `docs/TRAIN_DATA.md §1` (snapshot prep) + `docs/RESULTS.md` (tổng hợp).

> ⚠️ Các con số (counts/trần recall) = snapshot prep **2026-06-10** — tất định theo seed 42, chỉ đổi nếu sửa `prep_config.py` và re-run prep. Protocol thì cố định.

## 1. Bức tranh tổng — 2 trục overlay

Split có **2 trục độc lập chồng lên nhau** (overlay — không tách thêm nhóm user riêng cho cold-item):

```
                                  TRỤC ITEM
                     warm (~95% anime)        cold H (5% anime mới nhất)
           ┌──────────────────────────────┬─────────────────────────────┐
TRỤC USER  │                              │                             │
train 90%  │ train examples + history     │ VỨT HẲN (không example,     │
           │ + hard_neg                   │ không history, không neg)   │
           ├──────────────────────────────┼─────────────────────────────┤
val   5%   │ 80% → support (history)      │ 100% → query                │
           │ 20% → query  = examples/val  │ = examples/val_cold         │
           ├──────────────────────────────┼─────────────────────────────┤
test  5%   │ 80% → support (history)      │ 100% → query                │
           │ 20% → query  = examples/test │ = examples/test_cold        │
           └──────────────────────────────┴─────────────────────────────┘
```

- **Trục user (cold-user)**: hold trọn user. Trả lời "user CHƯA TỪNG thấy lúc train, encode từ history + features thì gợi ý tốt không?"
- **Trục item (cold-item H)**: cách ly trọn item khỏi mọi đường training. Trả lời "anime MỚI ra mắt, model chưa từng thấy id, gợi ý bằng content được không?"
- Cùng một tập eval user phục vụ cả 2 trục (cold pairs rất thưa, tách user riêng cho cold sẽ không đủ mẫu).

## 2. Trục user — cold-user 90/5/5

- `hash(username, seed=42) → train 90% / val 5% / test 5%` (stable hash, `03_split.py`). User nguyên vẹn nằm trong đúng 1 split — không có interaction nào của eval user lọt vào train.
- Điều kiện vào val/test: `n_pos ≥ EVAL_MIN_POS = 11` (user quá ít positive thì không đủ để vừa có support vừa có query). User dưới ngưỡng → train.
- Số hiện tại: 291.001 user = train 262.676 / val 14.058 / test 14.267.

## 3. Trục item — cold H

- **H = `ceil(5% · N)` anime mới nhất theo `start_date`** (parse full date, sort desc, tie-break ổn định theo mal_id). Hiện tại: 1.142 anime, cutoff **2024-09-30**. List: `train-data/cold_items.parquet`.
- **Item null `start_date` (291 anime): chỉ bị loại khỏi VÒNG TUYỂN H** — không thể khẳng định nó "mới nhất" khi không biết ngày. Chúng KHÔNG bị vứt khỏi data: vẫn là item warm bình thường (được train, được nằm trong history, được làm candidate, được làm warm query).
- Không có "bucket time" nào khác — split không phải temporal split toàn cục. Chỉ có 2 nhãn item: `∈ H` (cold) và `∉ H` (warm, bất kể ngày, kể cả null-date).
- **H bị cách ly ở 4 chỗ** (99_verify assert cả 4): ① train examples, ② history của MỌI user, ③ hard_neg pools, ④ support của eval user. Positive-H của train user vứt hẳn; positive-H của eval user → toàn bộ thành cold query.

## 4. Định nghĩa thuật ngữ

| Thuật ngữ | Định nghĩa | Nguồn file |
|---|---|---|
| **positive** | `status ∈ {completed, watching}` & `score ∉ [1,4]` | `prep_config.is_pos_expr()` |
| **hard-neg** | `dropped` ∪ `score ∈ [1,4]` (mọi status) | `prep_config.is_hardneg_expr()` |
| **seen** | MỌI interaction MỌI status của eval user (kể cả PTW, on_hold, score thấp) | `eval_seen.parquet` |
| **support** | Phần positive-warm của eval user mà model ĐƯỢC NHÌN làm input — chính là history khi encode user tower | `users.parquet: history_ids` |
| **query** | Phần positive bị GIẤU đi làm đáp án — model phải rank chúng cao trong catalog mà không được nhìn | `examples/split={val,test}[,_cold]` |
| **history** | = support (với eval user). Với train user: toàn bộ positive-warm (anchor được gỡ tại runtime khỏi history của chính batch đó) | `users.parquet` |

Quan hệ tập hợp (per eval user): `query ⊔ support = positive-warm` (chia 20/80 bằng tie-hash reproducible, seed 42); `cold query = positive ∩ H`; `seen ⊇ support ∪ query ∪ mọi thứ khác user từng chạm`.

## 5. Mỗi loại user — data dùng thế nào

**Train user (262.676):**
- examples = toàn bộ positive-warm (67.46M pairs) — mỗi pair là 1 anchor InfoNCE.
- history = cũng chính list đó (full, sort score desc); lúc train sample `train_hist_len=128`/anchor (config `final`), và anchor đang chấm bị gỡ khỏi history để khỏi tự nhìn đáp án.
- hard_neg ≤ 64/user, đã trừ H.

**Eval user (val/test):**
- Encode user tower bằng: support history (prefix `eval_history_cap=1024` của list full sort score desc) + gender + joined.
- Chấm trên 2 slice:
  - **Warm slice** (`examples/val|test`): query là item warm — đo chất lượng retrieval điều kiện bình thường. Dùng để **tuning** (headline `recall@200`).
  - **Cold slice** (`examples/val_cold|test_cold`): query là item H — item cache phải refresh với **id→OOV cho mọi row H** (mô phỏng anime ngoài vocab lúc serve; encode bằng id thật chưa train = đo noise). Đo khả năng gợi ý content-only.

## 6. Năm tập examples — dùng khi nào

| Tập | Pairs | Users | Vai trò |
|---|---|---|---|
| `train` | 67.456.284 | 262.676 | anchor InfoNCE |
| `val` | 751.026 | 14.029¹ | **vòng train Colab eval tập này** — chọn epoch/early-stop, headline `val recall@200` |
| `test` | 748.751 | 14.250¹ | **bảng leaderboard sort theo `test_recall@200`** — so sánh run/baseline warm |
| `val_cold` | 150.335 | 8.388 | debug/tuning cold (chạy tay, cell 10 notebook) |
| `test_cold` | 145.691 | 8.510 | **FINAL EXAM — ✅ đã chấm 1 lần (2026-06-18)**: full-catalog ndcg@10 .1397 / r@200 .4710; honly (chỉ rank giữa H) ndcg@10 .2368 / r@100 .6755 / r@200 .8261 (khớp val_cold → generalize) — `docs/RESULTS.md §7` |

¹ users có ≥1 warm query (user `n_warm < 2` không có warm query nhưng vẫn có thể có cold query).

Lưu ý quan hệ user giữa các tập: KHÔNG có nhóm user riêng cho cold — 8.388 users của `val_cold` là **subset** của 14.058 val users (những user có ≥1 positive ∈ H), tương tự 8.510 ⊂ 14.267 test users. Một user có thể xuất hiện ở cả slice warm lẫn cold (với 2 tập query khác nhau).

Kỷ luật: không tune trên `test`/`test_cold`. Vòng lặp quyết định chạy trên `val` (warm) + `val_cold` (cold); 2 tập test chỉ để báo cáo.

## 7. Seen-mask

Khi chấm, model rank **toàn catalog**. Mọi item user đã chạm (seen) — trừ chính query đang chấm — bị set `−inf`:

```
mask(user) = seen(user) − query_đang_chấm
```

- **Vì sao mask seen**: support history là thứ model vừa được nhìn làm input và (với MF/KNN) được fit trực tiếp — chúng tất yếu chiếm top-K. Lúc serve, service cũng filter toàn bộ list MAL của user trước khi trả kết quả. Không mask (hoặc chỉ mask một phần history) → top-K bị chiếm chỗ bởi item "đã xem" không bao giờ được recommend thật → recall bị đè thấp giả tạo và lệch khỏi serving.
- **Vì sao seen gồm cả PTW/on_hold/score thấp** (không chỉ positive): serving filter theo TOÀN BỘ list MAL của user — item đã nằm trong list (kể cả plan_to_watch) không bao giờ được recommend lại. Eval mask phải mirror đúng hành vi đó, nếu không số eval lệch khỏi serving.
- **Vì sao trừ query ra**: query ⊆ seen by construction — mask cả query thì đáp án biến mất, recall = 0.
- Warm và cold dùng 2 mask khác nhau (khác tập query bị trừ ra).

## 8. Trần lý thuyết recall@K (đọc số cho đúng)

Metric là mean-per-user của `hits@K / R` với R = TOÀN BỘ query của user. Khi `R > K` thì user đó tối đa đạt `K/R < 1` → trần trung bình < 1:

| Slice | q/user p50 | trần r@10 | trần r@50 | trần r@100 | trần r@200 | trần r@500 |
|---|---|---|---|---|---|---|
| warm val | 36 | .402 | .851 | .958 | .993 | 1.000 |
| warm test | 35 | .408 | .852 | .959 | .993 | 1.000 |
| cold val | 9 | .744 | .975 | .996 | 1.000 | 1.000 |
| cold test | 9 | .756 | .976 | .997 | 1.000 | 1.000 |

- **Headline `recall@200` trần ≈ 0.993 — coi như không kẹt trần.** `recall@10` warm trần 0.41: MF đạt .195 nghĩa là ~48% của mức khả thi, không phải 19.5% của 1.0.
- Trần áp dụng NHƯ NHAU cho mọi model/baseline (cùng tập query) → so sánh tương đối vẫn công bằng; chỉ số tuyệt đối ở K nhỏ trông thấp một phần vì trần.
- `ndcg@K` KHÔNG dính trần kiểu này: IDCG chuẩn hóa theo `min(R, K)` nên user nhiều query vẫn đạt được 1.0 lý thuyết.

## 9. Invariants được verify (99_verify + pytest)

- leak eval-example ∩ history = 0 (cả warm lẫn cold).
- H-isolation 4 chỗ = 0 vi phạm; cold examples ⊆ H đúng split.
- seen ⊇ history ∪ query của mọi eval user.
- History sort score desc (tie hash asc) → prefix cap = top-by-score.
- logQ: H có count train = 0 → floor `max(count,1)` giữ H finite (vẫn là candidate khi rank full catalog).

## 10. Audit phân phối eval (2026-06-14) — val/test có đại diện / bị bias không?

Đo phân phối val/test (warm + cold) so với toàn catalog + train-positive (script aggregate, không dump row). **Kết luận: SẠCH — `val ≈ test` ở MỌI chiều; warm query khớp phân phối train-positive; mọi "lệch" còn lại đều do thiết kế, không phải bias.** → không cần sửa split.

**User-side (hash theo username → độc lập feature):**
- Size train 262.676 / val 14.058 / test 14.267 (≈90/5/5). `gender_id` + `joined_bucket`: % train ≈ val ≈ test, lệch ≤0.9pp mọi bucket → không bias feature.
- Activity: support-history p50 = train 172 / val 144 / test 141. Eval thấp hơn KHÔNG phải eval ít active — support = 80% positive (20% giấu làm query); thực ra eligibility `n_pos≥11` loại đuôi user thưa nên **eval user hơi tích cực hơn nền chung** (caveat đã biết: metric nghiêng về active user, ít đại diện user rất thưa). val ≈ test.

**Item-side (warm query khớp train-positive = đại diện cái user THỰC SỰ xem):**
- `start_year` %: warm_val/test (≈42% 2010-17, ≈39% 2018+) ≈ train-positive freq-weighted (41.6/38.9). Catalog phẳng hơn (28/33) vì popularity ∝ độ mới → train-pos + eval CÙNG nghiêng về mới (đúng hành vi, không phải bias riêng của eval).
- `type` %: warm_val/test ≈70% TV ≈ train-positive 69.8%. Popularity `logq` warm query p10/p50/p90 ≈ [-10,-7.9,-6.5] cho cả val, test VÀ train-positive — không skew head/tail.
- Coverage: warm query phủ ~66% catalog warm (≈14.5k/21.7k distinct item), val ≈ test.

**Cold (H = 5% mới nhất — theo thiết kế):** 100% cold query ∈ 2018+, ~89% TV (anime mới đa số TV series) — đúng bản chất "anime mới", không phải anomaly. Coverage: cold_val 833/1.142 (72.9%) H, cold_test 843 (73.8%); val ≈ test.
