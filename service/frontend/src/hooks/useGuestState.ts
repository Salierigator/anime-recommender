import { useEffect, useState } from 'react';
import type { RefObject } from 'react';
import type { AnimeItem, SearchResultItem } from '../types';

export const GUEST_PICKS_KEY = 'anime_guest_picks';
export const WATCHED_SET_KEY = 'anime_watched_set';

/** Guest picks (Love) + watched set (Seen), persist localStorage sau khi app init xong. */
export function useGuestState(hasInitializedRef: RefObject<boolean>) {
  const [guestPicks, setGuestPicks] = useState<SearchResultItem[]>([]);
  const [watchedSet, setWatchedSet] = useState<Set<number>>(new Set());

  useEffect(() => {
    if (hasInitializedRef.current) {
      localStorage.setItem(GUEST_PICKS_KEY, JSON.stringify(guestPicks));
    }
  }, [guestPicks, hasInitializedRef]);

  useEffect(() => {
    if (hasInitializedRef.current) {
      localStorage.setItem(WATCHED_SET_KEY, JSON.stringify(Array.from(watchedSet)));
    }
  }, [watchedSet, hasInitializedRef]);

  const toggleSeen = (malId: number) => {
    setWatchedSet(prev => {
      const next = new Set(prev);
      if (next.has(malId)) {
        next.delete(malId);
      } else {
        next.add(malId);
      }
      return next;
    });
  };

  const toggleLove = (anime: AnimeItem, posterUrl?: string | null) => {
    setGuestPicks(prev => {
      if (prev.some(p => p.mal_id === anime.mal_id)) {
        return prev.filter(p => p.mal_id !== anime.mal_id);
      }
      return [...prev, {
        mal_id: anime.mal_id,
        title: anime.title,
        title_english: null,
        type: anime.type,
        year: anime.year,
        mal_score: anime.mal_score,
        image_url: posterUrl || null,
        in_corpus: true
      }];
    });
  };

  return { guestPicks, setGuestPicks, watchedSet, setWatchedSet, toggleSeen, toggleLove };
}
