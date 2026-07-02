import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { SearchForm } from './components/SearchForm';
import { AnimeCard } from './components/AnimeCard';
import { AnimeModal } from './components/AnimeModal';
import { recommendAPI, fetchPostersAPI } from './api';
import type { AnimeItem, RecommendResponse } from './types';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // pool contains the raw, unmodified pool of recommendations from backend
  const [pool, setPool] = useState<RecommendResponse | null>(null);
  
  // Client-side filters state
  const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);
  const [selectedThemes, setSelectedThemes] = useState<string[]>([]);
  const [selectedStudios, setSelectedStudios] = useState<string[]>([]);
  const [minScore, setMinScore] = useState<number>(0);
  
  // Client-side display counts
  const [mainK, setMainK] = useState<number>(20);
  const [coldK, setColdK] = useState<number>(10);
  
  // Posters cache map and tracking
  const [posters, setPosters] = useState<Record<number, string | null>>({});
  const requestedIdsRef = useRef<Set<number>>(new Set());
  
  const [selectedMalId, setSelectedMalId] = useState<number | null>(null);

  const handleSearch = async (username: string) => {
    setIsLoading(true);
    setError(null);
    setPool(null);
    setPosters({});
    requestedIdsRef.current = new Set();
    
    // Clear all client filters and display size defaults
    setSelectedGenres([]);
    setSelectedTypes([]);
    setSelectedThemes([]);
    setSelectedStudios([]);
    setMinScore(0);
    setMainK(20);
    setColdK(10);

    try {
      // Backend returns full pool (main up to 500, cold up to 200)
      const response = await recommendAPI({
        username,
        top_k: 500,
        cold_k: 200,
        sfw: true
      });
      setPool(response);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (err: any) {
      if (err.response?.data?.detail) {
        setError(typeof err.response.data.detail === 'string' 
          ? err.response.data.detail 
          : JSON.stringify(err.response.data.detail));
      } else {
        setError('An unexpected error occurred while connecting to the server.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Extract facets (genres, themes, studios, types) from the current pool
  const facetOptions = useMemo(() => {
    if (!pool) return { genres: [], themes: [], studios: [], types: [] };

    const genresSet = new Set<string>();
    const themesSet = new Set<string>();
    const studiosSet = new Set<string>();
    const typesSet = new Set<string>();

    const allItems = [...pool.main, ...pool.cold];
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
  }, [pool]);

  // Client-side filtering check
  const matchFilters = useCallback((item: AnimeItem) => {
    // Genres: AND — item must contain ALL selected genres
    if (selectedGenres.length > 0) {
      if (!item.genres || !selectedGenres.every(g => item.genres.includes(g))) {
        return false;
      }
    }

    // Themes: AND — item must contain ALL selected themes
    if (selectedThemes.length > 0) {
      if (!item.themes || !selectedThemes.every(t => item.themes.includes(t))) {
        return false;
      }
    }

    // Studios: OR — item belongs to ANY of the selected studios
    if (selectedStudios.length > 0) {
      if (!item.studios || !selectedStudios.some(s => item.studios.includes(s))) {
        return false;
      }
    }

    // Types: AND between facets, OR within the same facet (item.type ∈ selectedTypes)
    if (selectedTypes.length > 0) {
      if (!selectedTypes.includes(item.type)) {
        return false;
      }
    }

    // Score filter: If minScore > 0, filter out items with mal_score == null or mal_score < minScore
    if (minScore > 0) {
      if (item.mal_score === null || item.mal_score < minScore) {
        return false;
      }
    }

    return true;
  }, [selectedGenres, selectedTypes, selectedThemes, selectedStudios, minScore]);

  // Get filtered lists
  const filteredMain = useMemo(() => {
    if (!pool) return [];
    return pool.main.filter(matchFilters);
  }, [pool, matchFilters]);

  const filteredCold = useMemo(() => {
    if (!pool) return [];
    return pool.cold.filter(matchFilters);
  }, [pool, matchFilters]);

  // Sliced display lists
  const displayedMain = useMemo(() => {
    return filteredMain.slice(0, mainK);
  }, [filteredMain, mainK]);

  const displayedCold = useMemo(() => {
    return filteredCold.slice(0, coldK);
  }, [filteredCold, coldK]);

  // Check if any client-side filters are actively selected
  const isFiltered = selectedGenres.length > 0 ||
    selectedTypes.length > 0 ||
    selectedThemes.length > 0 ||
    selectedStudios.length > 0 ||
    minScore > 0;

  // Poster lazy loading useEffect (debounced by 200ms)
  useEffect(() => {
    if (!pool) return;

    const displayedIds = [
      ...displayedMain.map(a => a.mal_id),
      ...displayedCold.map(a => a.mal_id)
    ];

    // Find IDs that are not in posters AND not already requested
    const missingIds = displayedIds.filter(
      id => !(id in posters) && !requestedIdsRef.current.has(id)
    );

    if (missingIds.length === 0) {
      return;
    }

    const timer = setTimeout(async () => {
      // Register when the API fetch action actually begins
      missingIds.forEach(id => requestedIdsRef.current.add(id));
      try {
        const postersData = await fetchPostersAPI(missingIds);
        setPosters(prev => ({
          ...prev,
          ...postersData
        }));
      } catch (err) {
        console.error("Failed to fetch batch posters", err);
        // fallback to null map so cards fall back to individual Jikan calls
        const fallbackData: Record<number, string | null> = {};
        missingIds.forEach(id => {
          fallbackData[id] = null;
        });
        setPosters(prev => ({
          ...prev,
          ...fallbackData
        }));
      }
    }, 200);

    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [displayedMain, displayedCold, pool]);

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
            isLoading={isLoading}
            facetOptions={facetOptions}
            selectedGenres={selectedGenres}
            setSelectedGenres={setSelectedGenres}
            selectedTypes={selectedTypes}
            setSelectedTypes={setSelectedTypes}
            selectedThemes={selectedThemes}
            setSelectedThemes={setSelectedThemes}
            selectedStudios={selectedStudios}
            setSelectedStudios={setSelectedStudios}
            minScore={minScore}
            setMinScore={setMinScore}
            mainK={mainK}
            setMainK={setMainK}
            coldK={coldK}
            setColdK={setColdK}
            hasPool={!!pool}
          />
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-12">
        {isLoading && (
          <div className="text-center py-20">
            <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-gray-900 border-r-transparent align-[-0.125em] motion-reduce:animate-[spin_1.5s_linear_infinite]" role="status">
              <span className="!absolute !-m-px !h-px !w-px !overflow-hidden !whitespace-nowrap !border-0 !p-0 ![clip:rect(0,0,0,0)]">Loading...</span>
            </div>
            <p className="mt-4 text-gray-500 font-medium">Analyzing your taste...</p>
          </div>
        )}

        {error && (
          <div className="max-w-2xl mx-auto p-4 border border-red-200 bg-red-50 text-red-700 text-center">
            <p className="font-semibold">Error</p>
            <p className="text-sm mt-1">{error}</p>
          </div>
        )}

        {pool && !isLoading && (
          <div className="space-y-16 animate-in fade-in duration-500">
            {/* Meta Info */}
            <div className="text-center text-sm text-gray-400">
              <p>Based on {pool.meta.total_entries} entries from your list</p>
            </div>

            {/* Main Recommendations */}
            <section>
              <div className="flex items-end justify-between mb-6 pb-2 border-b border-gray-100">
                <h2 className="text-2xl font-bold tracking-tight">Main Recommendations</h2>
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
                    />
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500 border border-dashed border-gray-200 bg-gray-50">
                  No matches
                </div>
              )}
            </section>

            {/* Cold Recommendations */}
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
                    />
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500 border border-dashed border-gray-200 bg-gray-50">
                  No matches
                </div>
              )}
            </section>

            {pool.main.length === 0 && pool.cold.length === 0 && (
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
      </main>
    </div>
  );
}

export default App;
