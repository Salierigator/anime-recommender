# fixtures/

Dữ liệu mẫu cho backend (commit vào repo — không gitignore).

- `recommend_sample.json` — **output mẫu** cho `POST /api/recommend`, shape `{main, cold, meta}` khớp
  `service/API_CONTRACT.md`. Dùng cho **mock mode** (`MOCK_MODE=1`) → frontend dev không cần load model
  (torch/lightgbm/artifacts ~5s). **Không phải** output production.
- `dummy_mal_ids.txt` — **input mẫu** (list mal_id) cho CLI path `--mal-ids`, test pipeline thật offline
  (không gọi MAL API).

Tạo lại `recommend_sample.json` từ CLI thật rồi thêm block `meta`:
```bash
venv/bin/python service/backend/recommend.py --mal-ids service/backend/fixtures/dummy_mal_ids.txt --dump
```
