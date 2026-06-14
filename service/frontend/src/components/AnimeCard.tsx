import { useEffect, useState } from 'react';
import { ExternalLink } from 'lucide-react';
import type { AnimeItem } from '../types';
import { fetchAnimePoster } from '../utils/jikanQueue';

interface Props {
  anime: AnimeItem;
  rank?: number;
}

export function AnimeCard({ anime, rank }: Props) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isUrlFetched, setIsUrlFetched] = useState<boolean>(false);
  const [isImageLoaded, setIsImageLoaded] = useState<boolean>(false);

  useEffect(() => {
    let isMounted = true;
    setIsUrlFetched(false);
    setIsImageLoaded(false);
    
    fetchAnimePoster(anime.mal_id).then((url) => {
      if (isMounted) {
        setImageUrl(url);
        setIsUrlFetched(true);
        // If there's no URL, we don't wait for onLoad
        if (!url) {
          setIsImageLoaded(true);
        }
      }
    });

    return () => {
      isMounted = false;
    };
  }, [anime.mal_id]);

  const malUrl = `https://myanimelist.net/anime/${anime.mal_id}`;

  return (
    <div className="flex flex-col border border-gray-200 bg-white hover:shadow-md transition-shadow">
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
              className={`w-full h-full object-cover transition-opacity duration-500 ease-in-out ${isImageLoaded ? 'opacity-100' : 'opacity-0'}`}
              loading="lazy"
            />
          )}

          {/* Fallback Text if No Image */}
          {!imageUrl && isUrlFetched && (
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
                <span className="text-[10px] text-gray-400 uppercase tracking-wider">MAL</span>
                <span className="text-sm font-medium text-gray-700">{anime.mal_score.toFixed(2)}</span>
              </div>
            )}
            
            {anime.pred !== undefined && anime.pred !== null && (
              <div className="flex flex-col">
                <span className="text-[10px] text-gray-400 uppercase tracking-wider">Pred</span>
                <span className="text-sm font-medium text-gray-700">{anime.pred.toFixed(2)}</span>
              </div>
            )}
            
            {anime.cos !== undefined && anime.cos !== null && (
              <div className="flex flex-col">
                <span className="text-[10px] text-gray-400 uppercase tracking-wider">Cos</span>
                <span className="text-sm font-medium text-gray-700">{anime.cos.toFixed(2)}</span>
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
          className="inline-flex items-center gap-1.5 text-xs font-medium text-gray-600 hover:text-gray-900 transition-colors"
        >
          View on MAL
          <ExternalLink className="w-3.5 h-3.5" />
        </a>
      </div>
    </div>
  );
}
