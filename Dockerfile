# Render.com (Docker, free tier) — backend real mode. Build context = repo ROOT.
#
# artifacts/ + data/cleaned/details.csv gitignored → KHÔNG có trong context. Chúng được tải từ
# HF model repo lúc BUILD và bake vào image (cold start khỏi tải lại). Xem DEPLOY.md.
FROM python:3.11-slim

# libgomp1 = OpenMP runtime của LightGBM. python:*-slim KHÔNG có → `import lightgbm` chết ngay
# lúc boot: "OSError: libgomp.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Chạy non-root uid 1000 (di sản HF Spaces, giữ lại vì là best practice — Render không đòi hỏi).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# 1) deps — layer đắt nhất, để trước cho cache. torch tách riêng vì cần index CPU-only.
COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 && \
    pip install --no-cache-dir -r requirements-deploy.txt && \
    pip install --no-cache-dir huggingface_hub==0.36.0

# 2) artifacts (~56MB) từ HF model repo → ./artifacts/ + ./data/cleaned/.
#    Đổi repo = sửa default ARG ở đây.
#    ARTIFACTS_REPO="" → bỏ qua bước này, dùng bind-mount (test local, xem DEPLOY.md §6).
ARG ARTIFACTS_REPO=Salierigator/anime-recommender-artifacts
ENV ARTIFACTS_REPO=${ARTIFACTS_REPO}
RUN if [ -n "$ARTIFACTS_REPO" ]; then \
      python -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download(repo_id=os.environ['ARTIFACTS_REPO'], local_dir='.', \
allow_patterns=['artifacts/*', 'data/cleaned/*'])" && rm -rf .cache ; \
    else echo '>> skip artifacts download — expect bind-mount at runtime' ; fi

# 3) code — đổi nhiều nhất nên để cuối, chỉ layer này rebuild khi sửa code.
COPY --chown=user . .

# MAP_ENABLED=0: map/ đóng băng → /api/map 503, map_xy=null, recommend vẫn chạy.
# MAL_CLIENT_ID + CORS_ORIGINS: set ở Render Environment, KHÔNG hardcode ở đây.
ENV MOCK_MODE=0 \
    MAP_ENABLED=0

WORKDIR $HOME/app/service/backend
EXPOSE 7860
# Render truyền $PORT (10000) lúc chạy; local/docker test không có $PORT → fallback 7860.
# Shell-form CMD bắt buộc để expand biến.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
