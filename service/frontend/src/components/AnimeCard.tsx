import type { AnimeItem } from '../types';

interface Props {
  anime: AnimeItem;
  rank?: number;
}

export function AnimeCard({ anime, rank }: Props) {
  return (
    <div className="flex border border-gray-200 bg-white hover:bg-gray-50 transition-colors">
      {/* Poster placeholder */}
      <div className="w-24 h-36 bg-gray-100 flex items-center justify-center flex-shrink-0 border-r border-gray-200">
        <span className="text-xs text-gray-400 font-medium">No Image</span>
      </div>

      {/* Info */}
      <div className="p-4 flex flex-col justify-between flex-1 min-w-0">
        <div>
          <div className="flex items-start justify-between gap-2">
            <h3 className="font-semibold text-gray-900 leading-tight truncate" title={anime.title}>
              {rank !== undefined && <span className="text-gray-400 mr-2">#{rank}</span>}
              {anime.title}
            </h3>
            {anime.year && (
              <span className="text-xs text-gray-500 whitespace-nowrap">{anime.year}</span>
            )}
          </div>
          
          <div className="mt-1 text-sm text-gray-600">
            {anime.type !== '?' ? anime.type : 'Unknown Type'}
          </div>
        </div>

        {/* Scores */}
        <div className="flex items-center gap-4 mt-3">
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
  );
}
