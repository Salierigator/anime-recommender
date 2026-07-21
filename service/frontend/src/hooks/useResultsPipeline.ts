import { useCallback, useEffect, useMemo } from 'react';
import { sortAnimeItems } from '../utils/sortAnime';
import type { AnimeItem, FacetOptions, RecommendResponse, TabPrefs } from '../types';
import type { UpdatePrefs } from './useTabPrefs';

/** Pool hiện tại → facetOptions → filter → sort → slice; kèm dọn filter stale khi pool đổi. */
export function useResultsPipeline(
  currentPool: RecommendResponse | null,
  prefs: TabPrefs,
  updatePrefs: UpdatePrefs
) {
  const facetOptions = useMemo<FacetOptions>(() => {
    if (!currentPool) return { genres: [], themes: [], studios: [], types: [] };

    const genresSet = new Set<string>();
    const themesSet = new Set<string>();
    const studiosSet = new Set<string>();
    const typesSet = new Set<string>();

    const allItems = [...currentPool.main, ...currentPool.cold];
    allItems.forEach(item => {
      if (item.genres) item.genres.forEach(g => genresSet.add(g));
      if (item.themes) item.themes.forEach(t => themesSet.add(t));
      if (item.studios) item.studios.forEach(s => studiosSet.add(s));
      if (item.type) typesSet.add(item.type);
    });

    return {
      genres: Array.from(genresSet).sort((a, b) => a.localeCompare(b)),
      themes: Array.from(themesSet).sort((a, b) => a.localeCompare(b)),
      studios: Array.from(studiosSet).sort((a, b) => a.localeCompare(b)),
      types: Array.from(typesSet).filter(t => t && t !== '?').sort((a, b) => a.localeCompare(b)),
    };
  }, [currentPool]);

  // Filter đang chọn không còn trong pool mới → bỏ
  useEffect(() => {
    if (currentPool) {
      updatePrefs(prev => ({
        genres: prev.genres.filter(g => facetOptions.genres.includes(g)),
        themes: prev.themes.filter(t => facetOptions.themes.includes(t)),
        studios: prev.studios.filter(s => facetOptions.studios.includes(s)),
        types: prev.types.filter(t => facetOptions.types.includes(t)),
      }));
    }
  }, [facetOptions, currentPool, updatePrefs]);

  const matchFilters = useCallback((item: AnimeItem) => {
    if (prefs.genres.length > 0) {
      if (!item.genres || !prefs.genres.every(g => item.genres.includes(g))) return false;
    }
    if (prefs.themes.length > 0) {
      if (!item.themes || !prefs.themes.every(t => item.themes.includes(t))) return false;
    }
    if (prefs.studios.length > 0) {
      if (!item.studios || !prefs.studios.some(s => item.studios.includes(s))) return false;
    }
    if (prefs.types.length > 0) {
      if (!prefs.types.includes(item.type)) return false;
    }
    if (prefs.minScore > 0) {
      if (item.mal_score === null || item.mal_score < prefs.minScore) return false;
    }
    return true;
  }, [prefs]);

  const filteredMain = useMemo(() => {
    if (!currentPool) return [];
    return currentPool.main.filter(matchFilters);
  }, [currentPool, matchFilters]);

  const filteredCold = useMemo(() => {
    if (!currentPool) return [];
    return currentPool.cold.filter(matchFilters);
  }, [currentPool, matchFilters]);

  const sortedMain = useMemo(() => {
    return sortAnimeItems(filteredMain, prefs.sortBy, prefs.sortAsc);
  }, [filteredMain, prefs.sortBy, prefs.sortAsc]);

  const sortedCold = useMemo(() => {
    return sortAnimeItems(filteredCold, prefs.sortBy, prefs.sortAsc);
  }, [filteredCold, prefs.sortBy, prefs.sortAsc]);

  const displayedMain = useMemo(() => {
    return sortedMain.slice(0, prefs.mainK);
  }, [sortedMain, prefs.mainK]);

  const displayedCold = useMemo(() => {
    return sortedCold.slice(0, prefs.coldK);
  }, [sortedCold, prefs.coldK]);

  const isFiltered = prefs.genres.length > 0 ||
    prefs.types.length > 0 ||
    prefs.themes.length > 0 ||
    prefs.studios.length > 0 ||
    prefs.minScore > 0;

  return { facetOptions, filteredMain, filteredCold, displayedMain, displayedCold, isFiltered };
}
