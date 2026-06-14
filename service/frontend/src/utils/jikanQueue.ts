// Cache to store resolved image URLs
const CACHE = new Map<number, string | null>();

interface QueueItem {
  malId: number;
  resolve: (url: string | null) => void;
}

const queue: QueueItem[] = [];
let isProcessing = false;

const JIKAN_BASE = 'https://api.jikan.moe/v4';

/**
 * Processes the queue of image requests, ensuring we don't exceed 3 requests per second.
 * We add a 350ms delay between requests to be safe.
 */
async function processQueue() {
  if (isProcessing || queue.length === 0) return;
  isProcessing = true;

  while (queue.length > 0) {
    const item = queue.shift();
    if (!item) continue;

    // Check cache again in case a duplicate was queued before the first one resolved
    if (CACHE.has(item.malId)) {
      item.resolve(CACHE.get(item.malId)!);
      continue;
    }

    try {
      const res = await fetch(`${JIKAN_BASE}/anime/${item.malId}`);
      if (!res.ok) {
        if (res.status === 429) {
          // Rate limited, push back to front and wait a bit longer
          queue.unshift(item);
          await new Promise(r => setTimeout(r, 1000));
          continue;
        }
        throw new Error(`Status ${res.status}`);
      }
      
      const data = await res.json();
      const images = data?.data?.images?.jpg;
      const imageUrl = images?.large_image_url || images?.image_url || null;
      
      CACHE.set(item.malId, imageUrl);
      item.resolve(imageUrl);
    } catch (err) {
      console.error(`Error fetching image for ${item.malId}:`, err);
      // Even on error, cache null so we don't retry repeatedly and get blocked
      CACHE.set(item.malId, null);
      item.resolve(null);
    }

    // Delay ~350ms to respect 3 req/s rate limit
    await new Promise(r => setTimeout(r, 350));
  }

  isProcessing = false;
}

/**
 * Fetches the poster image URL for a given anime using Jikan API.
 * Uses a global queue to avoid hitting rate limits.
 */
export function fetchAnimePoster(malId: number): Promise<string | null> {
  if (CACHE.has(malId)) {
    return Promise.resolve(CACHE.get(malId)!);
  }

  return new Promise((resolve) => {
    queue.push({ malId, resolve });
    processQueue();
  });
}
