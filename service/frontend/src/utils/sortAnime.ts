import type { AnimeItem, SortKey } from '../types';

// Sort client-side thuần — 'relevance' = giữ thứ tự mảng từ backend (đảo khi desc), null xuống cuối.
export function sortAnimeItems(items: AnimeItem[], sortBy: SortKey, isAsc: boolean): AnimeItem[] {
  const sorted = [...items];

  if (sortBy === 'relevance') {
    // desc (mũi tên xuống, mặc định) = giữ thứ tự backend (liên quan nhất trước); asc = đảo.
    if (isAsc) {
      sorted.reverse();
    }
    return sorted;
  }

  sorted.sort((a, b) => {
    let valA: string | number | null = null;
    let valB: string | number | null = null;

    if (sortBy === 'score') {
      valA = a.mal_score;
      valB = b.mal_score;
    } else if (sortBy === 'popularity') {
      valA = a.popularity;
      valB = b.popularity;
    } else if (sortBy === 'date') {
      valA = a.start_date;
      valB = b.start_date;
    }

    const isNullA = valA === null || valA === undefined;
    const isNullB = valB === null || valB === undefined;

    if (isNullA && isNullB) return 0;
    if (isNullA) return 1;
    if (isNullB) return -1;

    if (sortBy === 'date') {
      const cmp = (valA as string).localeCompare(valB as string);
      return isAsc ? cmp : -cmp;
    } else if (sortBy === 'popularity') {
      // popularity là rank (1 = phổ biến nhất) → đảo so với score: desc (xuống) = phổ biến nhất trước.
      return isAsc ? (valB as number) - (valA as number) : (valA as number) - (valB as number);
    } else {
      return isAsc ? (valA as number) - (valB as number) : (valB as number) - (valA as number);
    }
  });

  return sorted;
}
