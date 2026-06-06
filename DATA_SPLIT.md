# Data Split

Hai hướng chia data cho Two-Tower retrieval. Hiện tại pipeline dùng **Cold-User**; **Double Cold Start** là hướng mở rộng để đo thêm khả năng gợi ý **anime mới**. Hai trục (user / item) độc lập nhau.

## 1. Cold-User (hiện tại)

Hold-out **trọn user** 90/5/5 (`scripts/build_train_data/`, chia tất định theo hash username). User val/test không xuất hiện trong train; positive của họ tách support (history) / query (target). **Mọi item đều warm** — đều có mặt lúc train.

- **Đo được:** gợi ý cho **user mới** (chưa từng train) tốt không. Đây là headline metric (recall@K / nDCG@K).
- **Không đo được:** gợi ý **anime mới** tốt không — vì lúc eval item nào cũng warm.

## 2. Double Cold Start (hướng mới)

Giữ nguyên trục user (cold-by-user 90/5/5), **thêm trục item**: tách **~5% anime có start_date muộn nhất** làm tập cold (H), cách ly khỏi *training* (KHÔNG xóa khỏi ratings — giữ tương tác (eval-user, H) làm query ground-truth). H vẫn nằm trong bảng feature để content tower encode được, và được đưa vào candidate pool lúc eval.

- **Trục item là 2-way:** 95% warm (trong train) / 5% cold (eval-only) — không phải 3-way 90/5/5.
- **Newest chứ không random:** anime cũ ngoài đời luôn có id để train → random holdout đo một kịch bản không tồn tại; newest mô phỏng đúng "anime vừa ra" và ít rating nên mất rất ít train data.
- **Metric:** vẫn recall@K / nDCG@K, nhưng **tách slice** warm-query vs cold-query (trộn chung → tín hiệu cold bị nhấn chìm).

## Trade-off

| | Cold-User (cũ) | Double Cold Start (mới) |
|---|---|---|
| Trục phủ | chỉ user mới | user mới **+ anime mới** |
| Đo cold-item | ❌ item luôn warm | ✅ honest; **bắt buộc** nếu bật `use_item_id` |
| Pipeline | đơn giản | phức tạp hơn (cách ly H + slice metric) |
| Data loss | không | mất (user, H) interaction — nhỏ vì newest ít rating |
| Rủi ro metric | thổi phồng cold-item nếu serve anime mới | cold-query có thể mỏng/noisy; doubly-cold bi quan hơn serving thực |

**Tóm lại:** Cold-User đủ và đúng cho mục tiêu chính (serve user mới). Double Cold Start không thay thế nó mà **phủ thêm trục item** — cần khi muốn gợi ý anime mới giữa các lần retrain, hoặc khi bật id embedding (lúc đó metric cold-by-user thổi phồng năng lực cold-item).
