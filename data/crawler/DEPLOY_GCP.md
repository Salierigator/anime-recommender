# Đẩy crawler lên Google Cloud (e2-micro Always Free) — hướng dẫn từng bước

> File tạm cho một lần deploy, dùng xong xoá. Viết 2026-07-20, số liệu GCP verify từ
> docs chính thức ngày đó (nguồn ở cuối file). Account dùng: hieulhp6@gmail.com.

## 0. Tóm tắt sự thật đã verify (20/07/2026)

| Điều | Thực tế |
|---|---|
| Free thật không? | **Có**, không giới hạn thời gian — nhưng **rất kén cấu hình**: chỉ đúng 1 combo được free (xem dòng dưới), sai một ô là bị tính tiền ngay dù vẫn ở US region. |
| Combo free duy nhất | **1× `e2-micro`** (0.25–2 vCPU burstable, **1GB RAM**), **chỉ** ở `us-west1` (Oregon) / `us-central1` (Iowa) / `us-east1` (S. Carolina); **30GB Standard persistent disk**; **1GB egress/tháng** ra internet (từ Bắc Mỹ, trừ China & Australia). |
| Đủ cho crawler? | Đủ. Crawler cần ~1 core, <500MB RAM, ~10GB disk, chủ yếu **inbound** (tải data về = ingress = luôn free). |
| Cần billing account? | **Có** — phải bật Cloud Billing (gắn thẻ), nhưng **không bị charge** chừng nào ở trong hạn mức Always Free. Bạn đã có account rồi nên chỉ cần chắc billing đang active. |
| Idle-reclaim? | **KHÔNG.** GCP không stop instance vì CPU thấp (khác hẳn Oracle). Đây là lý do lớn để chọn GCP — bỏ được toàn bộ trò né reclaim. |
| Bẫy #1 — boot disk | VM mới mặc định là **Balanced PD (tính tiền)**. **Phải đổi sang "Standard persistent disk", ≤30GB** mới nằm trong free tier. |
| Bẫy #2 — egress | Kéo data về Mac là **egress**, chỉ free 1GB/tháng, vượt ~$0.12/GB. ratings.csv có thể vài GB → **gzip trước khi rsync** (bước 8), nén ~5–10× nên thường về dưới/quanh 1GB. |
| Bẫy #3 — máy/region | `e2-small`, `e2-medium`, Balanced/SSD disk, hay region ngoài 3 cái trên = **charge ngay**. Chọn đúng `e2-micro` + `us-west1` + Standard disk. |

Nếu account của bạn **vẫn còn $300 credit Free Trial (90 ngày)** thì càng chắc: kể cả lỡ
vượt free tier chút cũng được credit gánh. Kiểm ở Console → Billing.

## 1. Chuẩn bị project & billing

Đăng nhập https://console.cloud.google.com bằng **hieulhp6@gmail.com**.

1. Tạo project mới (thanh trên cùng → **New Project**) tên `mal-crawler`, hoặc dùng
   project sẵn có.
2. **Billing**: Console → **Billing** → chắc chắn project được **link** tới một billing
   account đang *active*. Chưa có thì tạo (gắn thẻ; không charge nếu ở free tier).
3. Bật API Compute Engine: Console → **Compute Engine → VM instances** (lần đầu mở nó tự
   hỏi *Enable* → bấm, chờ ~1 phút).
4. (Khuyên) Đặt lưới an toàn: **Billing → Budgets & alerts → Create budget** → $1, alert
   50/90/100% → email. Chỉ để báo động nếu lỡ tay tạo tài nguyên tính tiền.

## 2. Tạo VM

**Console → Compute Engine → VM instances → Create instance:**

- **Name**: `mal-crawler`.
- **Region**: **`us-west1` (Oregon)** — bắt buộc 1 trong 3 region free; Oregon gần MAL/CDN.
  **Zone**: để mặc định (vd `us-west1-b`).
- **Machine configuration**: series **E2** → machine type **`e2-micro`** (mục *Shared-core*).
  ⚠ Đừng để nó nhảy sang e2-small/medium.
- **Boot disk** → **Change**:
  - **Image**: Ubuntu → **Ubuntu 24.04 LTS** (x86/amd64).
  - **Boot disk type**: **Standard persistent disk** ← *đổi cái này, mặc định là Balanced*.
  - **Size**: **30 GB** (đúng mức free tối đa).
- **Networking / Firewall**: mặc định là đủ (external ephemeral IP + cho SSH). Không cần
  mở port gì thêm — crawler chỉ outbound.
- **Observability/Ops agent**: bỏ tick nếu có (agent ăn thêm RAM trên máy 1GB).

**Create** → chờ trạng thái xanh → ghi lại **External IP**.

### SSH vào máy — chọn 1 trong 2

**Cách A — gcloud CLI (gọn nhất nếu đã cài `gcloud` trên Mac):**
```bash
gcloud auth login              # nếu chưa đăng nhập, chọn hieulhp6@gmail.com
gcloud config set project mal-crawler
gcloud compute ssh mal-crawler --zone us-west1-b     # tự tạo & nạp SSH key
```
**Cách B — không cần gcloud:** ở dòng VM trong Console bấm nút **SSH** (mở terminal
trong trình duyệt). Hoặc thêm key thủ công: Console → VM → *Edit* → *SSH Keys* → paste
`cat ~/.ssh/id_ed25519.pub` (chưa có key thì `ssh-keygen -t ed25519`), rồi
`ssh <user>@<EXTERNAL_IP>` (user = phần trước @ trong email key, thường là `hieulhp6`).

> Dưới đây giả định user trên VM là `USER` — thay bằng user thật của bạn. Với
> `gcloud compute ssh` thì thường trùng tên google (`hieulhp6`).

## 3. Chuẩn bị máy (chạy trên VM)

```bash
sudo apt update && sudo apt install -y python3-venv tmux sqlite3

# swap 2GB — e2-micro chỉ 1GB RAM, swap tránh OOM khi apt/pip
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile \
  && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

mkdir -p ~/anime-recommender/data/crawler ~/anime-recommender/data/raw
python3 -m venv ~/anime-recommender/venv
~/anime-recommender/venv/bin/pip install requests
```

Crawler chỉ cần 5 file trong `data/crawler/` (common.py tự suy ROOT từ vị trí của nó,
nên **phải giữ đúng cấu trúc `<gì-đó>/data/crawler/`**). Copy từ **máy Mac**:

**Nếu dùng gcloud:**
```bash
cd "/Users/aiguystory/Desktop/anime recommender"
gcloud compute scp data/crawler/*.py mal-crawler:anime-recommender/data/crawler/ \
  --zone us-west1-b
```
**Nếu dùng SSH key thường (rsync):**
```bash
cd "/Users/aiguystory/Desktop/anime recommender"
rsync -av data/crawler/*.py USER@<EXTERNAL_IP>:anime-recommender/data/crawler/
```

**MAL_CLIENT_ID** — `service/.env` gitignored nên KHÔNG tự theo lên. Lấy giá trị ở Mac:
```bash
grep MAL_CLIENT_ID "/Users/aiguystory/Desktop/anime recommender/service/.env"
```
rồi trên VM ghi vào `~/.bashrc`:
```bash
echo 'export MAL_CLIENT_ID=<giá_trị_vừa_copy>' >> ~/.bashrc && source ~/.bashrc
```

## 4. Egress — điều duy nhất có thể tốn tiền

Crawl = tải data **về** VM = ingress = **luôn free**, không lo. Chỉ có 2 thứ là egress:
- Kéo kết quả từ VM về Mac (bước 8) — free 1GB/tháng, vượt ~$0.12/GB. → **gzip trước khi
  chuyển** (đã hướng dẫn ở bước 8), ratings.csv nén rất mạnh nên thường về quanh/dưới 1GB.
- Bản thân request ra MAL/Jikan (headers, query) là vài KB mỗi cái — tổng cả tuần vẫn
  cỡ vài chục MB, không đáng kể.

Không có gì phải làm ở bước này ngoài **nhớ gzip lúc lấy data về**. Budget alert $1 ở
bước 1 sẽ kêu nếu có gì bất thường.

## 5. ⚠ SMOKE TEST TRƯỚC KHI CAM KẾT — IP datacenter

MAL có thể đối xử IP datacenter khác IP nhà, **đặc biệt 2 đường HTML scrape**
(`users.php`, trang profile). Chạy đúng bộ smoke này trên VM trước:

```bash
cd ~/anime-recommender
venv/bin/python data/crawler/collect_usernames.py --iterations 3   # phải ra ~+20/poll
venv/bin/python data/crawler/crawl_details.py --pages 2            # Jikan, phải 200
venv/bin/python data/crawler/crawl_ratings.py --limit 5            # MAL API chính thức
venv/bin/python data/crawler/crawl_profiles.py --source html --limit 5
```

- Cả 4 ok → sang bước 6.
- `collect_usernames`/`crawl_profiles --source html` fail (403/503/trang lạ = WAF chặn)
  nhưng ratings + details ok → dự phòng: chạy **ratings + details trên VM** (đường API,
  ổn), còn **collector + profiles chạy ở máy nhà** (nhẹ, ~1 req/s); state DB lúc đó tách
  2 nơi → đừng làm song song mù, hỏi lại Claude để ghép quy trình.

## 6. Chạy thật trong tmux

Lưu ý 2 điều đã chốt từ trước:
- **Profiles bắt buộc `--source html`** cho tới khi Jikan user-endpoint hết outage
  (còn 504 toàn phần các ngày gần đây) VÀ nhánh from_jikan được verify. Đừng để mode auto.
- `crawl_ratings`/`crawl_profiles` chỉ lấy danh sách pending **tại lúc launch** → bọc
  vòng lặp để tự bắt user mới do collector nạp vào.

```bash
tmux new -s crawl
# --- window 0: collector (chạy nhiều ngày, nguồn nuôi 2 crawler kia)
cd ~/anime-recommender && venv/bin/python data/crawler/collect_usernames.py

# Ctrl-b c  → window 1: details (one-shot ~40 phút, xong là thôi)
cd ~/anime-recommender && venv/bin/python data/crawler/crawl_details.py

# Ctrl-b c  → window 2: ratings loop
cd ~/anime-recommender
while true; do venv/bin/python data/crawler/crawl_ratings.py; sleep 60; done

# Ctrl-b c  → window 3: profiles loop (HTML — xem lưu ý trên)
cd ~/anime-recommender
while true; do venv/bin/python data/crawler/crawl_profiles.py --source html; sleep 60; done
```

Thoát mà không giết gì: `Ctrl-b d` (detach). Vào lại: `gcloud compute ssh mal-crawler
--zone us-west1-b -- -t tmux attach -t crawl` (hoặc `ssh USER@<IP> -t tmux attach -t crawl`).

## 7. Theo dõi tiến độ

```bash
# đứng ở ~/anime-recommender trên VM:
sqlite3 data/raw/crawl_state.sqlite \
  "SELECT ratings_status, profile_status, COUNT(*) FROM users GROUP BY 1,2"
sqlite3 data/raw/crawl_state.sqlite "SELECT COUNT(*) FROM users"   # tổng username
du -sh data/raw/* && df -h /                                       # dung lượng (đừng vượt 30GB)
```

Mốc tham khảo: collector ~240 username/phút lúc đầu (≈300k sau ~2–3 ngày poll, tỉ lệ
trùng tăng dần); ratings ~1–2 ngày/300k user; profiles (html, ~0.8 req/s) là đường chậm
nhất, ~4 ngày/300k. Đủ số user mong muốn thì Ctrl-C collector trước, 2 loop kia chạy tiếp
đến khi `pending=0` (log `0 users to crawl` lặp lại là xong).

**Dừng/khởi động lại thoải mái**: mọi script resume từ state DB (đã test cả kill -9).
VM reboot → `tmux` mất, chạy lại 4 lệnh ở bước 6 là tiếp tục đúng chỗ cũ.
⚠ Theo dõi `df -h /` — disk chỉ 30GB; nếu ratings.csv phình gần đầy thì kéo về Mac
(bước 8) rồi có thể xoá bớt file `.bak` cũ để lấy chỗ.

## 8. Lấy data về (gzip để tiết kiệm egress) & dọn dẹp

**Trên VM** — nén trước (giảm mạnh egress; ratings.csv nén ~5–10×):
```bash
cd ~/anime-recommender/data/raw
gzip -k ratings.csv profiles.csv        # -k giữ bản gốc để crawler chạy tiếp nếu chưa xong
```
**Trên Mac** — kéo về (chạy được giữa chừng để backup; bản cuối mới là bản dùng):
```bash
# gcloud:
gcloud compute scp mal-crawler:'anime-recommender/data/raw/*.gz' \
  "/Users/aiguystory/Desktop/anime recommender/data/raw/" --zone us-west1-b
gcloud compute scp mal-crawler:anime-recommender/data/raw/details.jsonl.gz \
  "/Users/aiguystory/Desktop/anime recommender/data/raw/" --zone us-west1-b
# hoặc rsync (đã có SSH key). rsync -z nén trên đường truyền, giảm luôn egress bị tính:
rsync -avz --progress USER@<EXTERNAL_IP>:anime-recommender/data/raw/ \
  "/Users/aiguystory/Desktop/anime recommender/data/raw/"
```
Về tới Mac thì `gunzip` các file `.gz` (details.jsonl.gz giữ nguyên — vốn là gzip).

Kiểm nhanh: `wc -l data/raw/*.csv`, so số user `ok` trong state DB với số username distinct
trong CSV. Raw **có thể có dòng duplicate** (thiết kế crash-safe append-first) — cleaning
pipeline sẽ dedup, không phải lỗi.

**Xong hẳn** → Console → VM → **Delete** (xoá cả boot disk để không tốn 30GB-month). VM
đã tắt/xoá thì $0. Nếu muốn giữ máy để crawl đợt sau: **Stop** thay vì Delete — nhưng disk
dừng vẫn tính vào 30GB free, còn IP ephemeral sẽ đổi khi start lại.

## Nguồn đã kiểm (20/07/2026)

- Always Free combo (e2-micro; us-west1/us-central1/us-east1; 30GB Standard PD; 1GB egress
  Bắc Mỹ trừ China/Australia; cần billing active):
  https://docs.cloud.google.com/free/docs/free-cloud-features
- Chỉ e2-micro + đúng region + Standard disk mới free, sai là charge ngay; billing phải
  bật: https://cloudwebschool.com/docs/gcp/fundamentals/free-tier/
- Hướng dẫn cộng đồng (chọn Standard PD, 3 region, $0 side-project):
  https://dev.to/jeaniscoding/how-to-host-your-side-projects-for-0-the-ultimate-gcp-free-tier-guide-3p07
