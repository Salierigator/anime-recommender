# DEPLOY.md — đưa web app lên public (free)

Runbook deploy v1. Backend + frontend + artifacts nằm ở 3 chỗ khác nhau, tất cả free.

> Thay `<HF_USER>` = username Hugging Face của bạn ở MỌI chỗ bên dưới. Nó **có thể khác**
> username GitHub (`Salierigator`) — kiểm tra ở https://huggingface.co/settings/profile.

## 0. Bức tranh tổng thể

```
  GitHub Pages                    Render.com (Docker, free)
  service/frontend/  ──fetch──►   FastAPI real mode        ──►  MAL API v2 + Jikan
  tĩnh, CDN, không ngủ            512MB RAM / 0.1 CPU           (crawl live mỗi username)
  salierigator.github.io          ngủ sau 15 phút không traffic
   /anime-recommender/                    ▲
                                          │ tải lúc BUILD, bake vào image
                                          │
                                 HF model repo (~55MB)
                                 artifacts/ + data/cleaned/details.csv
```

**Vì sao artifacts phải ở repo thứ 3**: `artifacts/` gitignored → không có trong GitHub, mà
backend build từ git. Nên artifacts đi đường riêng (HF **model repo** — vẫn free, chỉ Spaces bị
khoá). Tiện thể đây thành **backup offsite** — trước giờ chỉ có 1 bản local + 1 backup folder.

**RAM** (đo trong container Linux, không phải đoán): peak **393MB**, steady ~334MB, boot 4s.
Render free cho 512MB → vừa, nhưng sát (xem §8).

> Lịch sử chọn host (cập nhật 17/07/2026): kế hoạch ban đầu là **HF Space Docker** (free, 16GB,
> ngủ sau 48h) nhưng HF vừa khoá Docker/Gradio Space sau paywall PRO $9/tháng — free giờ chỉ còn
> Static. Các free tier khác đã soát: **Fly.io** bỏ free từ 2024, **Koyeb** đóng free compute
> 02/2026. Còn lại **Render**: 512MB, ngủ sau **15 phút** (spin-up ~1 phút), không cần thẻ.

## 1. Chuẩn bị

1. Tạo tài khoản https://huggingface.co (free, không cần thẻ).
2. Lấy token **WRITE**: https://huggingface.co/settings/tokens → New token → type `Write`.
3. Login (token lưu luôn vào git credential để bước 3 push được):

```bash
cd ~/Desktop/anime\ recommender
venv/bin/hf auth login --add-to-git-credential      # dán token WRITE vào
```

## 2. Đẩy artifacts lên HF model repo

Chỉ **6 file** cần cho web (không phải cả 228MB — `users_history.parquet` 163MB và các file
`eval_*` chỉ CLI/eval dùng, ở lại máy):

```bash
cd ~/Desktop/anime\ recommender

STAGE=/tmp/artifacts-upload
rm -rf $STAGE && mkdir -p $STAGE/artifacts $STAGE/data/cleaned
cp artifacts/user_tower.pt artifacts/item_vectors.npy artifacts/item_index.parquet \
   artifacts/ranker.txt artifacts/ranker_meta.json   $STAGE/artifacts/
cp -L data/cleaned/details.csv $STAGE/data/cleaned/       # -L BẮT BUỘC: details.csv là symlink

du -sh $STAGE            # 55M — đã chạy thử, đúng 6 file
venv/bin/hf upload <HF_USER>/anime-recommender-artifacts $STAGE . --repo-type=model
```

`hf upload` tự tạo repo nếu chưa có và tự đẩy file >10MB qua LFS. Xong vào
`https://huggingface.co/<HF_USER>/anime-recommender-artifacts` xem đủ 6 file chưa.

Cấu trúc thư mục trong repo (`artifacts/…`, `data/cleaned/…`) **phải giữ nguyên** — Dockerfile
tải thẳng vào đúng path code đang đọc.

> Repo này để **public**. Nếu để private thì image build sẽ 401 vì không có token lúc build.

## 3. Deploy backend lên Render

**3a. Kiểm tra Dockerfile trỏ đúng repo artifacts của bạn** — đúng 1 dòng:

```dockerfile
ARG ARTIFACTS_REPO=<HF_USER>/anime-recommender-artifacts
```

**3b. Commit + push code lên GitHub** (Render build thẳng từ GitHub, không cần remote riêng):

```bash
cd ~/Desktop/anime\ recommender
git add -A && git commit -m "deploy: Render + GitHub Pages"
git push origin test_deploy
```

**3c. Tạo Web Service**: https://dashboard.render.com (đăng ký free bằng GitHub, **không cần
thẻ**) → **New + → Web Service**:

- Connect repo `Salierigator/anime-recommender` (lần đầu phải authorize Render vào GitHub)
- **Branch**: `test_deploy` · Language: **Docker** (tự nhận Dockerfile ở root)
- **Region**: Singapore (gần VN nhất)
- **Instance type**: **Free** (512MB / 0.1 CPU)
- **Environment variables** — thêm ngay lúc tạo (hoặc sau ở tab *Environment*):

| Tên | Giá trị |
|---|---|
| `MAL_CLIENT_ID` | giá trị trong `service/.env` |
| `CORS_ORIGINS` | `["https://salierigator.github.io"]` |

⚠ **`CORS_ORIGINS` bắt buộc là JSON array** — pydantic-settings parse `List[str]` bằng JSON.
Đưa chuỗi trần `https://...` vào là app **crash lúc khởi động** (`SettingsError`). Đã test.

⚠ Origin **không có path**: `https://salierigator.github.io`, KHÔNG phải
`https://salierigator.github.io/anime-recommender/`.

Bấm **Deploy Web Service** → build ~5-10 phút (torch tải lâu), xem tab **Logs**. Không cần set
`PORT` — Render tự truyền, Dockerfile đã đọc `$PORT`.

Đổi env var → service tự restart. Backend xong ở URL hiện trên dashboard, dạng:
`https://anime-recommender-xxxx.onrender.com`.

**3d. (tuỳ chọn) Health check**: Settings → *Health Check Path* = `/api/health` — Render chỉ
đánh dấu deploy "live" khi endpoint trả 200, đỡ phải tự canh log.

## 4. Deploy frontend lên GitHub Pages

**4a.** Repo GitHub → **Settings → Pages** → *Build and deployment* → Source = **GitHub Actions**.

(`Salierigator/anime-recommender` đang **public** — đã kiểm tra — nên Pages free dùng được và
Actions không giới hạn phút. Đừng chuyển repo sang private: Pages sẽ cần tài khoản trả phí.)

**4b.** **Settings → Secrets and variables → Actions** → tab **Variables** → *New repository
variable*:

- Name `VITE_API_URL` · Value = URL Render ở §3c (vd `https://anime-recommender-xxxx.onrender.com`)

(Variable chứ không phải Secret: đây là URL public, và secret cũng không giấu được trong bundle JS.)

**4c.** Push → workflow `.github/workflows/deploy-frontend.yml` tự build và deploy:

```bash
git push origin test_deploy
```

Xem tab **Actions**. Xong: `https://salierigator.github.io/anime-recommender/`

> Workflow chỉ chạy khi push branch `test_deploy` và có đụng `service/frontend/**`. Muốn chạy tay:
> Actions → *Deploy frontend* → *Run workflow*.

## 5. Kiểm tra

```bash
# backend sống chưa (nếu đang ngủ: request đầu chờ ~1 phút spin-up)
curl https://anime-recommender-xxxx.onrender.com/api/health

# recommend không cần MAL key
curl -X POST https://anime-recommender-xxxx.onrender.com/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"mal_ids":[5114,9253,11061],"top_k":5,"cold_k":3}'
```

Rồi mở `https://salierigator.github.io/anime-recommender/`, nhập 1 MAL username thật.

Mong đợi `/api/health` trả **đúng** chuỗi này:

```json
{"status":"ok","mode":"live","model_loaded":true}
```

(`mode` là `live` chứ không phải `real` — health chỉ phân biệt mock/live.) `/api/map` trả **503**
— đúng, map đang tắt.

## 6. Test local bằng Docker (tuỳ chọn)

Không cần HF repo — bind-mount artifacts local vào:

```bash
cd ~/Desktop/anime\ recommender
docker build --build-arg ARTIFACTS_REPO= -t anime-rec-test .   # rỗng = bỏ qua bước tải

docker run --rm -p 8000:7860 \
  -v "$PWD/artifacts:/home/user/app/artifacts:ro" \
  -v "$PWD/legacy/data/cleaned-data/details.csv:/home/user/app/data/cleaned/details.csv:ro" \
  -e MOCK_MODE=0 -e MAP_ENABLED=0 \
  anime-rec-test
# → http://localhost:8000/docs
```

(mount thẳng `legacy/…` vì `data/cleaned/details.csv` là symlink — container không theo được.)

## 7. Retrain xong thì deploy lại thế nào

Artifacts **bake vào image lúc build**, không tải lúc chạy → up file mới thôi là chưa đủ:

1. Re-export artifacts (`model/retriever/export.py` — chỉ khi user yêu cầu, xem root `CLAUDE.md §4`)
2. Chạy lại bước 2 (upload đè)
3. Render → **Manual Deploy → Clear build cache & deploy** (bỏ cache mới tải lại artifacts)

Sửa code backend thì chỉ cần `git push origin test_deploy` — Render auto-deploy mỗi push vào
branch đang track.

## 8. Bẫy đã biết

- **`CORS_ORIGINS` không phải JSON** → app crash lúc boot. Xem §3c.
- **Service ngủ sau 15 phút không traffic** → request đầu chờ ~1 phút spin-up. Frontend trên
  Pages vẫn hiện ngay, chỉ API chờ. Hành vi cố hữu của Render free, không phải bug. Free tier
  cho 750 giờ chạy/tháng — 1 service có ngủ thì không bao giờ chạm trần.
- **RAM sát trần**: đo peak 393MB / trần 512MB. Nếu log Render báo *Out of memory* (OOM kill)
  thì đó là lý do — cân nhắc bấy giờ mới tối ưu (hoặc trả phí), đừng tối ưu trước.
- **0.1 CPU** (free) — chậm hơn máy local nhiều: boot và mỗi request recommend sẽ lâu hơn con
  số đo local. Chấp nhận được cho demo.
- **Build fail `RepositoryNotFoundError` / 401** ở bước tải artifacts = `ARTIFACTS_REPO` trong
  Dockerfile sai tên, hoặc HF model repo đang để private. Xem §3a + §2.
- **`pip install torch` mặc định kéo CUDA** → image 2.5GB+. Dockerfile đã ghim index CPU-only,
  đừng gộp torch vào `requirements-deploy.txt`.
- **Poster cache mất mỗi lần deploy/restart** (`service/backend/cache/posters.json`, filesystem
  ephemeral). Không sao — tự fetch lại, frontend có fallback Jikan qua `<img onError>`.
- **MAL API là single point of failure**: mọi username đều crawl live. MAL down / rate limit /
  list private → 404.
- **`map/` tắt bằng `MAP_ENABLED=0`** (mặc định), không phải xoá code. Bật lại: set `MAP_ENABLED=1`
  + đảm bảo `map/outputs/service/` khớp sha `item_vectors.npy`. Mock mode vẫn serve map fixture
  như cũ (frontend dev không bị ảnh hưởng).
