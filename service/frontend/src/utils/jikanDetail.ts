/* eslint-disable @typescript-eslint/no-explicit-any */
// Jikan detail cho MODAL (client-side → đẩy gánh nặng fetch synopsis/genres sang browser user,
// né rate-limit token MAL backend). CHỈ modal dùng (click-triggered, 1 lần/lượt) nên không cần
// queue phức tạp như bản cũ — chỉ cache + dedup in-flight + lùi nhẹ khi 429.
// Số hiển thị (MAL Score) KHÔNG lấy ở đây — modal ghim theo bản MAL v2 của card để khỏi lệch.
const CACHE = new Map<number, any>();
const inflight = new Map<number, Promise<any | null>>();
const JIKAN_BASE = 'https://api.jikan.moe/v4';

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Đọc cache đồng bộ (mở lại cùng anime → hiện ngay, khỏi fetch). */
export function getCachedJikanDetail(malId: number): any | undefined {
  return CACHE.get(malId);
}

/**
 * Object detail đầy đủ của 1 anime từ Jikan. null = lấy không được (network/429/outage) →
 * caller (modal) tự fallback sang backend /api/anime/{id}. Lỗi KHÔNG cache để còn retry.
 */
export function fetchJikanDetail(malId: number): Promise<any | null> {
  const cached = CACHE.get(malId);
  if (cached !== undefined) return Promise.resolve(cached);

  const existing = inflight.get(malId);
  if (existing) return existing;

  const p = (async () => {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const res = await fetch(`${JIKAN_BASE}/anime/${malId}`);
        if (res.status === 429) {
          await sleep(1000); // rate limit Jikan (3 req/s) → đợi rồi thử lại
          continue;
        }
        if (!res.ok) throw new Error(`status ${res.status}`);
        const json = await res.json();
        const data = json?.data ?? null;
        if (data) CACHE.set(malId, data);
        return data;
      } catch {
        if (attempt === 2) return null;
        await sleep(500);
      }
    }
    return null;
  })();

  inflight.set(malId, p);
  return p.finally(() => inflight.delete(malId));
}
