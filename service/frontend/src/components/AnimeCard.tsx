/* eslint-disable react-hooks/set-state-in-effect */
import { useEffect, useState } from 'react';
import { ExternalLink, Eye, Heart } from 'lucide-react';
import type { AnimeItem } from '../types';
import { fetchAnimePoster } from '../utils/jikanQueue';

interface Props {
  anime: AnimeItem;
  rank?: number;
  onClick?: () => void;
  posterUrl?: string | null;
  isGuestMode?: boolean;
  isSeen?: boolean;
  isLoved?: boolean;
  onSeen?: () => void;
  onLove?: () => void;
}

export function AnimeCard({ anime, rank, onClick, posterUrl, isGuestMode, isSeen, isLoved, onSeen, onLove }: Props) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isUrlResolved, setIsUrlResolved] = useState<boolean>(false);
  const [isImageLoaded, setIsImageLoaded] = useState<boolean>(false);

  // posterUrl?: string | null   (undefined = batch chưa fetch; null = batch báo không có; string = URL)
  useEffect(() => {
    let mounted = true;
    if (posterUrl === undefined) {          // đang chờ batch: spinner, KHÔNG gọi Jikan
      setIsUrlResolved(false);
      setIsImageLoaded(false);
      return;
    }
    if (posterUrl) {                         // có URL từ backend
      setImageUrl(posterUrl);
      setIsUrlResolved(true);
      setIsImageLoaded(false);               // onLoad sẽ bật lại; chỉ chạy khi posterUrl ĐỔI
      return;
    }
    setIsUrlResolved(false);                 // posterUrl === null → fallback Jikan
    fetchAnimePoster(anime.mal_id).then((url) => {
      if (!mounted) return;
      setImageUrl(url);
      setIsUrlResolved(true);
      setIsImageLoaded(!url);                // không có ảnh → tắt spinner, hiện "No Image"
    });
    return () => { mounted = false; };
  }, [anime.mal_id, posterUrl]);

  const handleImageError = () => {
    // If the image fails to load, fallback to Jikan
    fetchAnimePoster(anime.mal_id).then((url) => {
      if (url && url !== imageUrl) {
        setImageUrl(url);
        setIsImageLoaded(false);
      } else {
        setImageUrl(null);
        setIsImageLoaded(true);
      }
    });
  };

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
          {!isImageLoaded && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-100 z-10">
              <div className="h-4 w-4 rounded-full border-2 border-gray-300 border-t-gray-500 animate-spin"></div>
            </div>
          )}

          {/* Actual Image */}
          {imageUrl && (
            <img 
              src={imageUrl} 
              alt={anime.title} 
              onLoad={() => setIsImageLoaded(true)}
              onError={handleImageError}
              className={`w-full h-full object-cover transition-opacity duration-500 ease-in-out ${isImageLoaded ? 'opacity-100' : 'opacity-0'}`}
            />
          )}

          {/* Fallback Text if No Image */}
          {!imageUrl && isUrlResolved && (
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

          {/* Scores */}
          <div className="flex items-center gap-4 mt-2">
            {anime.mal_score !== null && (
              <div className="flex flex-col">
                <span className="text-[10px] text-gray-400 uppercase tracking-wider">MAL Score</span>
                <span className="text-sm font-medium text-gray-700">{anime.mal_score.toFixed(2)}</span>
              </div>
            )}
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
