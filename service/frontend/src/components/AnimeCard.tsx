import { useEffect, useRef, useState } from 'react';
import { ExternalLink, Eye, Heart, Users } from 'lucide-react';
import type { AnimeItem, PosterEntry } from '../types';

// 1234567 → "1.2M", 847000 → "847K" — gọn hơn số thô trên card.
const membersFmt = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });

interface Props {
  anime: AnimeItem;
  rank?: number;
  onClick?: () => void;
  entry?: PosterEntry; // từ batch /api/posters (MAL v2). undefined = batch chưa xong.
  isGuestMode?: boolean;
  isSeen?: boolean;
  isLoved?: boolean;
  onSeen?: () => void;
  onLove?: () => void;
}

export function AnimeCard({ anime, rank, onClick, entry, isGuestMode, isSeen, isLoved, onSeen, onLove }: Props) {
  // undefined = batch /api/posters chưa xong (spinner); null = MAL v2 không có ảnh; string = URL.
  const posterUrl = entry ? entry.poster : undefined;
  const [isImageLoaded, setIsImageLoaded] = useState<boolean>(false);
  const [broken, setBroken] = useState<boolean>(false);
  const imgRef = useRef<HTMLImageElement>(null);

  // Poster đến TỪ /api/posters (MAL v2, cache backend). <img> load lỗi → coi như không có ảnh.
  const imageUrl = typeof posterUrl === 'string' && !broken ? posterUrl : null;

  // Đổi URL ảnh → reset fade-in. Nếu ảnh ĐÃ nằm trong browser cache (modal vừa load, hoặc search
  // lại phim trùng) thì <img> onLoad KHÔNG bắn → tự set loaded qua img.complete để không kẹt spinner.
  useEffect(() => {
    if (!imageUrl) return;
    const img = imgRef.current;
    setIsImageLoaded(!!(img && img.complete && img.naturalWidth > 0));
  }, [imageUrl]);

  const handleImageError = () => setBroken(true);

  // Batch chưa xong → còn khả năng ra ảnh → spinner (đừng vội show "No Image").
  const awaitingBatch = posterUrl === undefined;
  const showSpinner = awaitingBatch || (!!imageUrl && !isImageLoaded);
  const showNoImage = !imageUrl && !awaitingBatch;

  // MAL score + members: ưu tiên bản MỚI từ batch backend (MAL v2, khớp MAL) → fallback snapshot pool.
  // Lọc qua Number.isFinite → null/undefined/NaN đều thành "—" (không còn "NaN" trên card).
  const rawScore = entry?.score ?? anime.mal_score;
  const rawMembers = entry?.members ?? anime.members;
  const score = typeof rawScore === 'number' && Number.isFinite(rawScore) ? rawScore : null;
  const members = typeof rawMembers === 'number' && Number.isFinite(rawMembers) ? rawMembers : null;

  const malUrl = `https://myanimelist.net/anime/${anime.mal_id}`;

  return (
    <div
      className={`flex flex-col border border-gray-200 bg-white hover:shadow-md transition-shadow relative group ${onClick ? 'cursor-pointer' : ''}`}
      onClick={onClick}
    >
      {isGuestMode && (
        <div className="absolute top-2 right-2 flex gap-1 z-20">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onSeen?.();
            }}
            className={`p-1.5 border rounded-none shadow-sm transition-all duration-200 cursor-pointer ${
              isSeen
                ? 'bg-sky-50 border-sky-500 text-sky-600 opacity-100'
                : 'bg-white border-gray-200 text-gray-400 hover:text-gray-900 sm:opacity-0 sm:group-hover:opacity-100'
            }`}
            title="Seen it — hide & don't recommend"
          >
            <Eye className="w-4 h-4" />
          </button>
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onLove?.();
            }}
            className={`p-1.5 border rounded-none shadow-sm transition-all duration-200 cursor-pointer ${
              isLoved
                ? 'bg-red-50 border-red-500 text-red-500 opacity-100'
                : 'bg-white border-gray-200 text-gray-400 hover:text-red-500 sm:opacity-0 sm:group-hover:opacity-100'
            }`}
            title="Love it — add to favorites"
          >
            <Heart className={`w-4 h-4 ${isLoved ? 'fill-red-500' : ''}`} />
          </button>
        </div>
      )}

      <div className="flex h-36">
        {/* Poster Image or Placeholder */}
        <div className="w-24 h-36 bg-gray-100 flex items-center justify-center flex-shrink-0 border-r border-gray-200 overflow-hidden relative">
          {/* Loading Spinner */}
          {showSpinner && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-100 z-10">
              <div className="h-4 w-4 rounded-full border-2 border-gray-300 border-t-gray-500 animate-spin"></div>
            </div>
          )}

          {/* Actual Image */}
          {imageUrl && (
            <img
              ref={imgRef}
              src={imageUrl}
              alt={anime.title}
              onLoad={() => setIsImageLoaded(true)}
              onError={handleImageError}
              className={`w-full h-full object-cover transition-opacity duration-500 ease-in-out ${isImageLoaded ? 'opacity-100' : 'opacity-0'}`}
            />
          )}

          {/* Fallback Text if No Image */}
          {showNoImage && (
            <span className="text-xs text-gray-400 font-medium px-2 text-center z-0">No Image</span>
          )}
        </div>

        {/* Info */}
        <div className="p-4 flex flex-col justify-between flex-1 min-w-0">
          <div>
            <div className="flex items-start justify-between gap-2">
              <h3 className="font-semibold text-gray-900 leading-tight line-clamp-2" title={anime.title}>
                {rank !== undefined && <span className="text-gray-400 mr-2">#{rank}</span>}
                {anime.title}
              </h3>
            </div>

            <div className="mt-1 flex items-center gap-2">
              <span className="text-xs text-gray-600 bg-gray-100 px-2 py-0.5 rounded">
                {anime.type !== '?' ? anime.type : 'Unknown'}
              </span>
              {anime.year && (
                <span className="text-xs text-gray-500">{anime.year}</span>
              )}
            </div>
          </div>

          {/* Scores — MAL Score + Popularity từ batch /api/posters (MAL v2) */}
          <div className="flex items-center gap-4 mt-2">
            <div className="flex flex-col">
              <span className="text-[10px] text-gray-400 uppercase tracking-wider">MAL Score</span>
              {score !== null ? (
                <span className="text-sm font-medium text-gray-700">{score.toFixed(2)}</span>
              ) : (
                <span className="text-sm font-medium text-gray-300">—</span>
              )}
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] text-gray-400 uppercase tracking-wider">Popularity</span>
              {members !== null ? (
                <span className="text-sm font-medium text-gray-700 flex items-center gap-1">
                  <Users className="w-3 h-3 text-gray-400" />
                  {membersFmt.format(members)}
                </span>
              ) : (
                <span className="text-sm font-medium text-gray-300">—</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Footer / Actions */}
      <div className="border-t border-gray-100 bg-gray-50 px-4 py-2 flex justify-end">
        <a
          href={malUrl}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          className="inline-flex items-center gap-1.5 text-xs font-medium text-gray-600 hover:text-gray-900 transition-colors"
        >
          View on MAL
          <ExternalLink className="w-3.5 h-3.5" />
        </a>
      </div>
    </div>
  );
}
