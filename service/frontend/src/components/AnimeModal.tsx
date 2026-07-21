import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { X, ExternalLink, AlertCircle } from 'lucide-react';
import { fetchJikanDetail, getCachedJikanDetail } from '../utils/jikanDetail';
import { fetchAnimeDetailAPI } from '../api';
import type { AnimeItem, PosterEntry } from '../types';

// Jikan JSON -> shape "enrich" modal cần. (Backend /api/anime đã trả sẵn các field này nên chỉ
// cần normalize Jikan.) KHÔNG lấy score ở đây — score ghim theo bản MAL v2 của card.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function normalizeJikan(d: any) {
  const tags = [
    ...(d?.genres || []),
    ...(d?.themes || []),
    ...(d?.demographics || []),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ].map((t: any) => t.name);
  return {
    title_english: d?.title_english ?? null,
    image_url: d?.images?.jpg?.large_image_url || d?.images?.jpg?.image_url || null,
    rank: d?.rank ?? null,
    popularity: d?.popularity ?? null,
    episodes: d?.episodes ?? null,
    status: d?.status ?? null,
    genres: tags,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    studios: d?.studios?.map((s: any) => s.name) || [],
    synopsis: d?.synopsis ?? null,
    type: d?.type ?? null,
    year: d?.year || d?.aired?.prop?.from?.year || null,
  };
}

interface AnimeModalProps {
  anime: AnimeItem | null;
  entry?: PosterEntry;   // poster + score/members MAL v2 (từ card) — ghim số cho khớp card
  isOpen: boolean;
  onClose: () => void;
}

export function AnimeModal({ anime, entry, isOpen, onClose }: AnimeModalProps) {
  const malId = anime?.mal_id ?? null;

  // enrich = phần NẶNG, lấy async: synopsis/genres/episodes/status/rank/studio/title_english.
  // Nguồn: Jikan CLIENT-SIDE (đẩy tải khỏi token backend) → fallback backend /api/anime nếu Jikan lỗi.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [enrich, setEnrich] = useState<any | null>(null);
  const [enriching, setEnriching] = useState<boolean>(false);
  const [enrichError, setEnrichError] = useState<boolean>(false);
  const [retryKey, setRetryKey] = useState<number>(0);

  useEffect(() => {
    if (!isOpen || !malId) {
      setEnrich(null);
      setEnriching(false);
      setEnrichError(false);
      return;
    }

    // Mở lại cùng anime đã fetch → hiện ngay, khỏi gọi Jikan.
    const cached = getCachedJikanDetail(malId);
    if (cached) {
      setEnrich(normalizeJikan(cached));
      setEnriching(false);
      setEnrichError(false);
      return;
    }

    let alive = true;
    setEnrich(null);
    setEnriching(true);
    setEnrichError(false);

    const backendFallback = async () => {
      try {
        const be = await fetchAnimeDetailAPI(malId);
        if (alive) { setEnrich(be); setEnriching(false); }
      } catch {
        if (alive) { setEnrichError(true); setEnriching(false); }
      }
    };

    fetchJikanDetail(malId)
      .then((jk) => {
        if (!alive) return;
        if (jk) { setEnrich(normalizeJikan(jk)); setEnriching(false); }
        else backendFallback();                     // Jikan 504/429/outage → backend MAL v2
      })
      .catch(() => { if (alive) backendFallback(); });

    return () => { alive = false; };
  }, [malId, isOpen, retryKey]);

  // Handle escape key
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      window.addEventListener('keydown', handleEsc);
    }
    return () => window.removeEventListener('keydown', handleEsc);
  }, [isOpen, onClose]);

  if (!isOpen || !anime) return null;

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  // ---- SEED (hiện NGAY, từ card): số + poster ghim theo MAL v2 để KHỚP card, không lệch.
  const seedScore = entry?.score ?? anime.mal_score ?? null;         // MAL Score — cùng nguồn card
  const seedPoster = entry?.poster ?? null;                          // poster MAL v2 (card đã load)
  const seedType = anime.type && anime.type !== '?' ? anime.type : null;

  const imageUrl = seedPoster ?? enrich?.image_url ?? null;
  const score = seedScore ?? enrich?.score ?? null;
  const title = anime.title;
  const titleEnglish = enrich?.title_english ?? null;
  const rank = enrich?.rank ?? null;
  const popularity = enrich?.popularity ?? anime.popularity ?? null;
  const type = seedType ?? enrich?.type ?? null;
  const year = anime.year ?? enrich?.year ?? null;
  const episodes = enrich?.episodes ?? null;
  const status = enrich?.status ?? null;
  const tags: string[] = enrich?.genres?.length ? enrich.genres : (anime.genres ?? []);
  const studios = (enrich?.studios?.length ? enrich.studios : anime.studios ?? []).join(', ');
  const synopsis = enrich?.synopsis ?? null;

  const bodyPending = enriching && !enrich;

  return createPortal(
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm transition-opacity"
      onClick={handleOverlayClick}
    >
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-white rounded-none shadow-2xl overflow-hidden flex flex-col md:flex-row animate-in fade-in zoom-in-95 duration-200 border border-gray-200">

        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-10 p-2 bg-white/80 hover:bg-white text-gray-700 rounded-none border border-gray-200 shadow-sm transition-colors cursor-pointer"
          aria-label="Close modal"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Left: Poster (seed — hiện ngay từ card) */}
        <div className="md:w-1/3 bg-gray-100 flex-shrink-0 hidden md:flex items-center justify-center overflow-hidden relative">
          {imageUrl ? (
            <>
              <img
                src={imageUrl}
                alt=""
                aria-hidden="true"
                className="absolute inset-0 w-full h-full object-cover blur-2xl scale-110 opacity-30"
              />
              <div className="absolute inset-0 bg-white/20"></div>
              <img
                src={imageUrl}
                alt={title}
                className="relative w-full aspect-[2/3] object-cover shadow-lg"
              />
            </>
          ) : (
            <div className="w-full aspect-[2/3] flex items-center justify-center text-gray-400 relative">
              No Image
            </div>
          )}
        </div>

        {/* Right: Content */}
        <div className="flex-1 flex flex-col min-h-0 min-w-0 overflow-hidden">
          {/* Header */}
          <div className="p-6 pb-4 border-b border-gray-100 flex-shrink-0">
            <div className="pr-10">
              <h2 className="text-2xl md:text-3xl font-bold text-gray-900 leading-tight">
                {title}
              </h2>
              {titleEnglish && titleEnglish !== title && (
                <p className="text-gray-500 mt-1 text-sm">{titleEnglish}</p>
              )}
            </div>

            {/* Stats Row */}
            <div className="flex flex-wrap items-center gap-4 mt-4 text-sm">
              {score != null && (
                <div className="flex flex-col">
                  <span className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">MAL Score</span>
                  <span className="font-bold text-gray-900">{score}</span>
                </div>
              )}
              {rank != null && (
                <div className="flex flex-col">
                  <span className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">Rank</span>
                  <span className="font-bold text-gray-900">#{rank}</span>
                </div>
              )}
              {popularity != null && (
                <div className="flex flex-col">
                  <span className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">Popularity</span>
                  <span className="font-bold text-gray-900">#{popularity}</span>
                </div>
              )}
            </div>
          </div>

          {/* Scrollable Body */}
          <div className="p-6 overflow-y-auto flex-1 min-h-0">
             {/* Meta Info */}
            <div className="flex flex-wrap items-center gap-y-2 gap-x-4 text-sm text-gray-700 mb-6 bg-gray-50 p-3 rounded-none border border-gray-100">
              {type && <div><span className="font-medium text-gray-500 mr-1">Type:</span> {type}</div>}
              {year && <div><span className="font-medium text-gray-500 mr-1">Year:</span> {year}</div>}
              {episodes && <div><span className="font-medium text-gray-500 mr-1">Episodes:</span> {episodes}</div>}
              {status && <div><span className="font-medium text-gray-500 mr-1">Status:</span> {status}</div>}
              {studios && <div className="w-full mt-1"><span className="font-medium text-gray-500 mr-1">Studio:</span> {studios}</div>}
            </div>

            {/* Tags */}
            {tags.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-6">
                {tags.map((tag: string, idx: number) => (
                  <span key={idx} className="px-2.5 py-1 bg-gray-100 text-gray-700 text-xs rounded-sm border border-gray-200">
                    {tag}
                  </span>
                ))}
              </div>
            )}

            {/* Synopsis */}
            {synopsis && (
              <div>
                <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider mb-2">Synopsis</h3>
                <p className="text-gray-600 text-sm leading-relaxed whitespace-pre-line">
                  {synopsis}
                </p>
              </div>
            )}

            {/* Đang lấy phần nặng (Jikan → backend) */}
            {bodyPending && (
              <div className="flex items-center gap-3 text-gray-400 text-sm py-2">
                <div className="h-4 w-4 rounded-full border-2 border-gray-200 border-t-gray-500 animate-spin"></div>
                Loading details…
              </div>
            )}

            {/* Cả Jikan lẫn backend fail — vẫn giữ seed (poster/title/score) ở trên, chỉ báo phần nặng lỗi */}
            {enrichError && !enrich && (
              <div className="flex items-center gap-3 text-sm text-gray-500 py-2">
                <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
                <span>Couldn't load full details.</span>
                <button
                  onClick={() => { setEnrichError(false); setRetryKey((k) => k + 1); }}
                  className="underline text-gray-600 hover:text-gray-900 cursor-pointer"
                >
                  Retry
                </button>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-gray-100 bg-gray-50 flex justify-end flex-shrink-0">
            <a
              href={`https://myanimelist.net/anime/${malId}`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 bg-gray-900 text-white text-sm font-medium rounded-none hover:bg-gray-800 transition-colors shadow-sm cursor-pointer"
            >
              View on MAL
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
