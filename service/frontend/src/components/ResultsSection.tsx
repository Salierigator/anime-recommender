import type { ReactNode } from 'react';
import { Loader2 } from 'lucide-react';
import { AnimeCard } from './AnimeCard';
import type { AnimeItem, PosterEntry, SearchResultItem } from '../types';

interface Props {
  variant: 'main' | 'cold'; // cold: title/count nhạt hơn, grid mờ nhẹ, không đánh số rank
  title: string;
  isLoading: boolean;
  totalCount: number; // số item sau filter (trước khi cắt theo Show Main/Cold)
  isFiltered: boolean;
  headerExtra?: ReactNode;
  items: AnimeItem[]; // đã sort + slice
  posters: Record<number, PosterEntry>;
  isGuestMode: boolean;
  watchedSet: Set<number>;
  guestPicks: SearchResultItem[];
  onSelect: (anime: AnimeItem) => void;
  onSeen: (malId: number) => void;
  onLove: (anime: AnimeItem, posterUrl?: string | null) => void;
}

export function ResultsSection({
  variant,
  title,
  isLoading,
  totalCount,
  isFiltered,
  headerExtra,
  items,
  posters,
  isGuestMode,
  watchedSet,
  guestPicks,
  onSelect,
  onSeen,
  onLove
}: Props) {
  const isCold = variant === 'cold';

  return (
    <section>
      <div className="flex items-end justify-between mb-6 pb-2 border-b border-gray-100">
        <div className="flex items-center gap-4">
          <h2 className={`text-2xl font-bold tracking-tight${isCold ? ' text-gray-600' : ''}`}>{title}</h2>
          {isLoading && (
            <Loader2 className="w-4 h-4 text-gray-400 animate-spin" />
          )}
          {headerExtra}
        </div>
        <span className={`text-sm ${isCold ? 'text-gray-400' : 'text-gray-500'}`}>
          {isFiltered
            ? `${totalCount} matching items`
            : `${totalCount} items`}
        </span>
      </div>
      {items.length > 0 ? (
        <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4${isCold ? ' opacity-90 hover:opacity-100 transition-opacity' : ''}`}>
          {items.map((anime, idx) => (
            <AnimeCard
              key={anime.mal_id}
              anime={anime}
              rank={isCold ? undefined : idx + 1}
              onClick={() => onSelect(anime)}
              entry={posters[anime.mal_id]}
              isGuestMode={isGuestMode}
              isSeen={watchedSet.has(anime.mal_id)}
              isLoved={guestPicks.some(p => p.mal_id === anime.mal_id)}
              onSeen={() => onSeen(anime.mal_id)}
              onLove={() => onLove(anime, posters[anime.mal_id]?.poster)}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-12 text-gray-500 border border-dashed border-gray-200 bg-gray-50">
          No matches
        </div>
      )}
    </section>
  );
}
