/* eslint-disable react-hooks/set-state-in-effect */
import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { SearchForm } from './components/SearchForm';
import { AnimeCard } from './components/AnimeCard';
import { AnimeModal } from './components/AnimeModal';
import { MapPreview } from './components/MapPreview';
import { MapExplorer } from './components/MapExplorer';
import { recommendAPI, fetchPostersAPI, fetchMapAPI, pingHealthAPI, fetchAnimeDetailFallbackAPI } from './api';
import type { AnimeItem, RecommendResponse, MapResponse, SearchResultItem } from './types';

function App() {
  const [loadingStates, setLoadingStates] = useState<Record<'username' | 'guest', boolean>>({
    username: false,
    guest: false
  });
  const [slowLoadingStates, setSlowLoadingStates] = useState<Record<'username' | 'guest', boolean>>({
    username: false,
    guest: false
  });
  const [error, setError] = useState<string | null>(null);
  
  const [tabPools, setTabPools] = useState<Record<'username' | 'guest', RecommendResponse | null>>({
    username: null,
    guest: null
  });
  
  // Client-side filters state (keyed by tab)
  const [genres, setGenres] = useState<Record<'username' | 'guest', string[]>>({ username: [], guest: [] });
  const [types, setTypes] = useState<Record<'username' | 'guest', string[]>>({ username: [], guest: [] });
  const [themes, setThemes] = useState<Record<'username' | 'guest', string[]>>({ username: [], guest: [] });
  const [studios, setStudios] = useState<Record<'username' | 'guest', string[]>>({ username: [], guest: [] });
  const [minScores, setMinScores] = useState<Record<'username' | 'guest', number>>({ username: 0, guest: 0 });
  
  // Client-side display counts (keyed by tab)
  const [mainKs, setMainKs] = useState<Record<'username' | 'guest', number>>({ username: 20, guest: 20 });
  const [coldKs, setColdKs] = useState<Record<'username' | 'guest', number>>({ username: 10, guest: 10 });

  const [searchedUsername, setSearchedUsername] = useState<string>(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('u') || '';
  });
  
  // Posters cache map and tracking
  const [posters, setPosters] = useState<Record<number, string | null>>({});
  const requestedIdsRef = useRef<Set<number>>(new Set());
  
  const [selectedMalId, setSelectedMalId] = useState<number | null>(null);
  const [mapData, setMapData] = useState<MapResponse | null>(null);
  const [isMapOpen, setIsMapOpen] = useState(false);

  // Lifted state
  const [activeTab, setActiveTab] = useState<'username' | 'guest'>('username');
  const [guestPicks, setGuestPicks] = useState<SearchResultItem[]>([]);
  const [watchedSet, setWatchedSet] = useState<Set<number>>(new Set());
  const hasInitialized = useRef(false);

  const handleSearch = async (
    params: { username?: string; mal_ids?: number[] },
    searchTab: 'username' | 'guest',
    skipUrlSync = false,
    passedWatchedSet?: number[]
  ) => {
    setLoadingStates(prev => ({ ...prev, [searchTab]: true }));
    setError(null);
    setTabPools(prev => ({ ...prev, [searchTab]: null }));
    setPosters({});
    requestedIdsRef.current = new Set();
    
    if (searchTab === 'username' && params.username) {
      setSearchedUsername(params.username);
    }

    if (!skipUrlSync) {
      const url = new URL(window.location.href);
      if (searchTab === 'username' && params.username) {
        url.search = `?u=${encodeURIComponent(params.username)}`;
      } else if (searchTab === 'guest' && params.mal_ids && params.mal_ids.length > 0) {
        url.search = `?ids=${params.mal_ids.join(',')}`;
        const wSet = passedWatchedSet ? new Set(passedWatchedSet) : watchedSet;
        if (wSet.size > 0) {
          const wArr = Array.from(wSet);
          if (url.search.length + wArr.join(',').length < 2000) {
            url.searchParams.set('watched', wArr.join(','));
          }
        }
      }
      window.history.replaceState(null, '', url.toString());
    }

    try {
      const exIds = passedWatchedSet || Array.from(watchedSet);
      const response = await recommendAPI({
        ...params,
        exclude_ids: exIds.length > 0 ? exIds : undefined,
        top_k: 500,
        cold_k: 200,
        sfw: true
      });
      setTabPools(prev => ({ ...prev, [searchTab]: response }));
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: unknown } } };
      if (axiosErr.response?.data?.detail) {
        setError(typeof axiosErr.response.data.detail === 'string' 
          ? axiosErr.response.data.detail 
          : JSON.stringify(axiosErr.response.data.detail));
      } else {
        setError('An unexpected error occurred while connecting to the server.');
      }
    } finally {
      setLoadingStates(prev => ({ ...prev, [searchTab]: false }));
    }
  };

  // Initialize from localStorage and URL
  useEffect(() => {
    if (hasInitialized.current) return;
    hasInitialized.current = true;

    // Load from localStorage first
    const savedPicks = localStorage.getItem('anime_guest_picks');
    const savedWatched = localStorage.getItem('anime_watched_set');
    let picks: SearchResultItem[] = savedPicks ? JSON.parse(savedPicks) : [];
    let watched: Set<number> = savedWatched ? new Set(JSON.parse(savedWatched)) : new Set();

    // Check URL params
    const params = new URLSearchParams(window.location.search);
    const uParam = params.get('u');
    const idsParam = params.get('ids');
    const watchedParam = params.get('watched');

    if (uParam) {
      setActiveTab('username');
      if (watchedParam) {
        const wIds = watchedParam.split(',').map(Number).filter(n => !isNaN(n));
        watched = new Set(wIds);
        setWatchedSet(watched);
        localStorage.setItem('anime_watched_set', JSON.stringify(Array.from(watched)));
      }
      handleSearch({ username: uParam }, 'username', true);
    } else if (idsParam) {
      setActiveTab('guest');
      const pickIds = idsParam.split(',').map(Number).filter(n => !isNaN(n));
      picks = pickIds.map(id => ({
        mal_id: id,
        title: `ID ${id}`,
        title_english: null,
        type: null,
        year: null,
        mal_score: null,
        image_url: null,
        in_corpus: true
      }));
      setGuestPicks(picks);
      localStorage.setItem('anime_guest_picks', JSON.stringify(picks));

      if (watchedParam) {
        const wIds = watchedParam.split(',').map(Number).filter(n => !isNaN(n));
        watched = new Set(wIds);
        setWatchedSet(watched);
        localStorage.setItem('anime_watched_set', JSON.stringify(Array.from(watched)));
      }
      
      handleSearch({ mal_ids: pickIds }, 'guest', true, Array.from(watched));

      // Hydrate chips asynchronously
      pickIds.forEach(id => {
        fetchAnimeDetailFallbackAPI(id).then(res => {
          if (res) {
            setGuestPicks(prev => prev.map(p => p.mal_id === id ? {
              ...p,
              title: res.title,
              image_url: res.image_url,
              year: res.year,
              type: res.type
            } : p));
          }
        }).catch(() => {});
      });
    } else {
      setGuestPicks(picks);
      setWatchedSet(watched);
    }
  }, []);

  // Save to localStorage when changed
  useEffect(() => {
    if (hasInitialized.current) {
      localStorage.setItem('anime_guest_picks', JSON.stringify(guestPicks));
    }
  }, [guestPicks]);

  useEffect(() => {
    if (hasInitialized.current) {
      localStorage.setItem('anime_watched_set', JSON.stringify(Array.from(watchedSet)));
    }
  }, [watchedSet]);

  // Synchronize URL parameters with active tab state
  useEffect(() => {
    if (!hasInitialized.current) return;

    const url = new URL(window.location.href);
    if (activeTab === 'username') {
      // Clear guest params
      url.searchParams.delete('ids');
      url.searchParams.delete('watched');
      
      // Sync username param
      if (tabPools.username && searchedUsername) {
        url.searchParams.set('u', searchedUsername);
      } else if (loadingStates.username && searchedUsername) {
        url.searchParams.set('u', searchedUsername);
      } else {
        url.searchParams.delete('u');
      }
    } else {
      // Clear username params
      url.searchParams.delete('u');
      
      // Sync guest params
      if (tabPools.guest && guestPicks.length > 0) {
        url.searchParams.set('ids', guestPicks.map(p => p.mal_id).join(','));
        if (watchedSet.size > 0) {
          url.searchParams.set('watched', Array.from(watchedSet).join(','));
        } else {
          url.searchParams.delete('watched');
        }
      } else if (loadingStates.guest && guestPicks.length > 0) {
        url.searchParams.set('ids', guestPicks.map(p => p.mal_id).join(','));
        if (watchedSet.size > 0) {
          url.searchParams.set('watched', Array.from(watchedSet).join(','));
        } else {
          url.searchParams.delete('watched');
        }
      } else {
        url.searchParams.delete('ids');
        url.searchParams.delete('watched');
      }
    }
    window.history.replaceState(null, '', url.toString());
  }, [activeTab, tabPools.username, tabPools.guest, searchedUsername, guestPicks, watchedSet, loadingStates.username, loadingStates.guest]);

  useEffect(() => {
    pingHealthAPI();
  }, []);

  useEffect(() => {
    if (!loadingStates.username) {
      setSlowLoadingStates(prev => ({ ...prev, username: false }));
      return;
    }
    const timer = setTimeout(() => {
      setSlowLoadingStates(prev => ({ ...prev, username: true }));
    }, 8000);
    return () => clearTimeout(timer);
  }, [loadingStates.username]);

  useEffect(() => {
    if (!loadingStates.guest) {
      setSlowLoadingStates(prev => ({ ...prev, guest: false }));
      return;
    }
    const timer = setTimeout(() => {
      setSlowLoadingStates(prev => ({ ...prev, guest: true }));
    }, 8000);
    return () => clearTimeout(timer);
  }, [loadingStates.guest]);


  const currentPool = tabPools[activeTab];

  const facetOptions = useMemo(() => {
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

  // Drop stale filters when facetOptions change
  useEffect(() => {
    if (currentPool) {
      setGenres(prev => ({
        ...prev,
        [activeTab]: prev[activeTab].filter(g => facetOptions.genres.includes(g))
      }));
      setThemes(prev => ({
        ...prev,
        [activeTab]: prev[activeTab].filter(t => facetOptions.themes.includes(t))
      }));
      setStudios(prev => ({
        ...prev,
        [activeTab]: prev[activeTab].filter(s => facetOptions.studios.includes(s))
      }));
      setTypes(prev => ({
        ...prev,
        [activeTab]: prev[activeTab].filter(t => facetOptions.types.includes(t))
      }));
    }
  }, [facetOptions, currentPool, activeTab]);

  const matchFilters = useCallback((item: AnimeItem) => {
    const activeGenres = genres[activeTab];
    const activeTypes = types[activeTab];
    const activeThemes = themes[activeTab];
    const activeStudios = studios[activeTab];
    const activeMinScore = minScores[activeTab];

    if (activeGenres.length > 0) {
      if (!item.genres || !activeGenres.every(g => item.genres.includes(g))) return false;
    }
    if (activeThemes.length > 0) {
      if (!item.themes || !activeThemes.every(t => item.themes.includes(t))) return false;
    }
    if (activeStudios.length > 0) {
      if (!item.studios || !activeStudios.some(s => item.studios.includes(s))) return false;
    }
    if (activeTypes.length > 0) {
      if (!activeTypes.includes(item.type)) return false;
    }
    if (activeMinScore > 0) {
      if (item.mal_score === null || item.mal_score < activeMinScore) return false;
    }
    return true;
  }, [genres, types, themes, studios, minScores, activeTab]);

  const filteredMain = useMemo(() => {
    if (!currentPool) return [];
    return currentPool.main.filter(matchFilters);
  }, [currentPool, matchFilters]);

  const filteredCold = useMemo(() => {
    if (!currentPool) return [];
    return currentPool.cold.filter(matchFilters);
  }, [currentPool, matchFilters]);

  const displayedMain = useMemo(() => {
    return filteredMain.slice(0, mainKs[activeTab]);
  }, [filteredMain, mainKs, activeTab]);

  const displayedCold = useMemo(() => {
    return filteredCold.slice(0, coldKs[activeTab]);
  }, [filteredCold, coldKs, activeTab]);

  const isFiltered = genres[activeTab].length > 0 ||
    types[activeTab].length > 0 ||
    themes[activeTab].length > 0 ||
    studios[activeTab].length > 0 ||
    minScores[activeTab] > 0;

  useEffect(() => {
    if (!currentPool) return;
    const displayedIds = [
      ...displayedMain.map(a => a.mal_id),
      ...displayedCold.map(a => a.mal_id)
    ];
    const missingIds = displayedIds.filter(
      id => !(id in posters) && !requestedIdsRef.current.has(id)
    );
    if (missingIds.length === 0) return;
    const timer = setTimeout(async () => {
      missingIds.forEach(id => requestedIdsRef.current.add(id));
      try {
        const postersData = await fetchPostersAPI(missingIds);
        setPosters(prev => ({ ...prev, ...postersData }));
      } catch {
        const fallbackData: Record<number, string | null> = {};
        missingIds.forEach(id => { fallbackData[id] = null; });
        setPosters(prev => ({ ...prev, ...fallbackData }));
      }
    }, 200);
    return () => clearTimeout(timer);
  }, [displayedMain, displayedCold, currentPool]);

  useEffect(() => {
    const hasAnyPool = tabPools.username || tabPools.guest;
    if (hasAnyPool && !mapData) {
      fetchMapAPI()
        .then((data) => setMapData(data))
        .catch((err) => {
          if (err.response?.status === 503) {
            console.log('Map feature is currently disabled on the server (503).');
          } else {
            console.warn('Failed to load map data:', err);
          }
        });
    }
  }, [tabPools, mapData]);

  const handleSeen = (malId: number) => {
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

  const handleLove = (anime: AnimeItem, posterUrl?: string | null) => {
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

  return (
    <div className="min-h-screen bg-white text-gray-900 font-sans selection:bg-gray-200">
      <header className="py-12 px-4 border-b border-gray-100">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4">
            Anime Recommender
          </h1>
          <p className="text-gray-500 max-w-xl mx-auto mb-10">
            Discover your next favorite anime. Enter your MyAnimeList username to get personalized recommendations.
          </p>
          
          <SearchForm 
            onSubmit={handleSearch} 
            isLoading={loadingStates[activeTab]}
            facetOptions={facetOptions}
            selectedGenres={genres[activeTab]}
            setSelectedGenres={(val) => setGenres(prev => ({ ...prev, [activeTab]: val }))}
            selectedTypes={types[activeTab]}
            setSelectedTypes={(val) => setTypes(prev => ({ ...prev, [activeTab]: val }))}
            selectedThemes={themes[activeTab]}
            setSelectedThemes={(val) => setThemes(prev => ({ ...prev, [activeTab]: val }))}
            selectedStudios={studios[activeTab]}
            setSelectedStudios={(val) => setStudios(prev => ({ ...prev, [activeTab]: val }))}
            minScore={minScores[activeTab]}
            setMinScore={(val) => setMinScores(prev => ({ ...prev, [activeTab]: val }))}
            mainK={mainKs[activeTab]}
            setMainK={(val) => setMainKs(prev => ({ ...prev, [activeTab]: val }))}
            coldK={coldKs[activeTab]}
            setColdK={(val) => setColdKs(prev => ({ ...prev, [activeTab]: val }))}
            hasPool={!!currentPool}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            guestPicks={guestPicks}
            setGuestPicks={setGuestPicks}
          />
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-12">
        {loadingStates[activeTab] && (
          <div className="text-center py-20">
            <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-gray-900 border-r-transparent align-[-0.125em] motion-reduce:animate-[spin_1.5s_linear_infinite]" role="status">
              <span className="!absolute !-m-px !h-px !w-px !overflow-hidden !whitespace-nowrap !border-0 !p-0 ![clip:rect(0,0,0,0)]">Loading...</span>
            </div>
            <p className="mt-4 text-gray-500 font-medium">Analyzing your taste...</p>
            {slowLoadingStates[activeTab] && (
              <p className="mt-2 text-sm text-gray-400">
                The server is waking up — the first request can take up to a minute.
              </p>
            )}
          </div>
        )}

        {error && (
          <div className="max-w-2xl mx-auto p-4 border border-red-200 bg-red-50 text-red-700 text-center">
            <p className="font-semibold">Error</p>
            <p className="text-sm mt-1">{error}</p>
          </div>
        )}

        {currentPool && !loadingStates[activeTab] && (
          <div className="space-y-16 animate-in fade-in duration-500">
            <div className="text-center text-sm text-gray-400">
              {activeTab === 'guest' ? (
                <p>Based on your {currentPool.meta.total_entries ?? guestPicks.length} favorite anime</p>
              ) : (
                <p>Based on {currentPool.meta.total_entries} entries from your list</p>
              )}
            </div>

            {mapData && (
              <MapPreview
                mapData={mapData}
                mapXy={currentPool.meta.map_xy ?? null}
                onClick={() => setIsMapOpen(true)}
              />
            )}

            <section>
              <div className="flex items-end justify-between mb-6 pb-2 border-b border-gray-100">
                <div className="flex items-center gap-4">
                  <h2 className="text-2xl font-bold tracking-tight">Main Recommendations</h2>
                  {activeTab === 'guest' && (
                    <div className="hidden sm:flex items-center gap-3 mt-1">
                      <span className="text-xs text-gray-400">
                        Mark cards as seen or favorite, then press Get recommendations to refresh
                      </span>
                      {watchedSet.size > 0 && (
                        <button
                          type="button"
                          onClick={() => setWatchedSet(new Set())}
                          className="text-xs underline text-gray-500 hover:text-gray-900 cursor-pointer"
                        >
                          Clear seen ({watchedSet.size})
                        </button>
                      )}
                    </div>
                  )}
                </div>
                <span className="text-sm text-gray-500">
                  {isFiltered
                    ? `${filteredMain.length} matching items`
                    : `${filteredMain.length} most relevant items`}
                </span>
              </div>
              {displayedMain.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {displayedMain.map((anime, idx) => (
                    <AnimeCard 
                      key={anime.mal_id} 
                      anime={anime} 
                      rank={idx + 1} 
                      onClick={() => setSelectedMalId(anime.mal_id)}
                      posterUrl={posters[anime.mal_id]}
                      isGuestMode={activeTab === 'guest'}
                      isSeen={watchedSet.has(anime.mal_id)}
                      isLoved={guestPicks.some(p => p.mal_id === anime.mal_id)}
                      onSeen={() => handleSeen(anime.mal_id)}
                      onLove={() => handleLove(anime, posters[anime.mal_id])}
                    />
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500 border border-dashed border-gray-200 bg-gray-50">
                  No matches
                </div>
              )}
            </section>

            <section>
              <div className="flex items-end justify-between mb-6 pb-2 border-b border-gray-100">
                <h2 className="text-2xl font-bold tracking-tight text-gray-600">New & Trending (Cold)</h2>
                <span className="text-sm text-gray-400">
                  {isFiltered
                    ? `${filteredCold.length} matching items`
                    : `${filteredCold.length} newest items`}
                </span>
              </div>
              {displayedCold.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 opacity-90 hover:opacity-100 transition-opacity">
                  {displayedCold.map((anime) => (
                    <AnimeCard 
                      key={anime.mal_id} 
                      anime={anime} 
                      onClick={() => setSelectedMalId(anime.mal_id)}
                      posterUrl={posters[anime.mal_id]}
                      isGuestMode={activeTab === 'guest'}
                      isSeen={watchedSet.has(anime.mal_id)}
                      isLoved={guestPicks.some(p => p.mal_id === anime.mal_id)}
                      onSeen={() => handleSeen(anime.mal_id)}
                      onLove={() => handleLove(anime, posters[anime.mal_id])}
                    />
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500 border border-dashed border-gray-200 bg-gray-50">
                  No matches
                </div>
              )}
            </section>

            {currentPool.main.length === 0 && currentPool.cold.length === 0 && (
              <div className="text-center py-20 text-gray-500">
                No recommendations found.
              </div>
            )}
          </div>
        )}

        <AnimeModal 
          malId={selectedMalId} 
          isOpen={!!selectedMalId} 
          onClose={() => setSelectedMalId(null)} 
        />

        {mapData && (
          <MapExplorer
            isOpen={isMapOpen}
            onClose={() => setIsMapOpen(false)}
            mapData={mapData}
            mapXy={currentPool?.meta.map_xy ?? null}
            mainRecs={currentPool?.main ?? []}
            coldRecs={currentPool?.cold ?? []}
            onSelectAnime={(malId) => setSelectedMalId(malId)}
            isDetailOpen={!!selectedMalId}
          />
        )}
      </main>
    </div>
  );
}

export default App;
