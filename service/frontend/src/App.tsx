import { useState, useEffect, useRef } from 'react';
import { SearchForm } from './components/SearchForm';
import { AnimeModal } from './components/AnimeModal';
import { ResultsSection } from './components/ResultsSection';
import { fetchPostersAPI, pingHealthAPI } from './api';
import { useTabPrefs } from './hooks/useTabPrefs';
import { useGuestState } from './hooks/useGuestState';
import { useRecommendations } from './hooks/useRecommendations';
import { useResultsPipeline } from './hooks/useResultsPipeline';
import { useUrlSync } from './hooks/useUrlSync';
import type { AnimeItem, PosterEntry, Tab } from './types';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('username');
  const hasInitializedRef = useRef(false);

  // Filter/sort/số hiển thị theo tab (client-side thuần)
  const { prefs, updatePrefs, updateTabPrefs } = useTabPrefs(activeTab);

  // Guest picks (Love) + watched set (Seen) + persist localStorage
  const { guestPicks, setGuestPicks, watchedSet, setWatchedSet, toggleSeen, toggleLove } =
    useGuestState(hasInitializedRef);

  // Posters cache map and tracking. Poster keyed theo mal_id là bất biến → GIỮ qua các lần search
  // (search user khác trùng phim → dùng lại ngay, không load lại). Chỉ tăng dần, không reset.
  const [posters, setPosters] = useState<Record<number, PosterEntry>>({});
  const requestedIdsRef = useRef<Set<number>>(new Set());

  const { tabPools, loadingStates, slowLoadingStates, error, searchedUsername, handleSearch, cancelSearch } =
    useRecommendations({ watchedSet, onPoolReset: () => {} });

  const currentPool = tabPools[activeTab];
  const { facetOptions, filteredMain, filteredCold, displayedMain, displayedCold, isFiltered } =
    useResultsPipeline(currentPool, prefs, updatePrefs);

  useUrlSync({
    hasInitializedRef,
    activeTab,
    setActiveTab,
    tabPools,
    loadingStates,
    searchedUsername,
    guestPicks,
    setGuestPicks,
    watchedSet,
    setWatchedSet,
    prefs,
    updateTabPrefs,
    handleSearch,
  });

  const [selectedAnime, setSelectedAnime] = useState<AnimeItem | null>(null);

  // Automatic search for guest mode when active tab, picks, or watched set changes
  useEffect(() => {
    if (!hasInitializedRef.current) return;
    if (activeTab !== 'guest') return;

    const pickIds = guestPicks.map(p => p.mal_id);
    handleSearch({ mal_ids: pickIds.length > 0 ? pickIds : undefined }, 'guest');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, guestPicks, watchedSet]);

  useEffect(() => {
    pingHealthAPI();
  }, []);

  // Fetch posters (batch, debounce 200ms) cho các card đang hiển thị
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
        const fallbackData: Record<number, PosterEntry> = {};
        missingIds.forEach(id => { fallbackData[id] = { poster: null, score: null, members: null }; });
        setPosters(prev => ({ ...prev, ...fallbackData }));
      }
    }, 200);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [displayedMain, displayedCold, currentPool]);

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
            onCancel={() => cancelSearch(activeTab)}
            isLoading={loadingStates[activeTab]}
            hasPool={!!currentPool}
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            guestPicks={guestPicks}
            setGuestPicks={setGuestPicks}
            facetOptions={facetOptions}
            prefs={prefs}
            updatePrefs={updatePrefs}
          />
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-12">
        {loadingStates[activeTab] && !currentPool && (
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

        {currentPool && (
          <div className={`space-y-16 animate-in fade-in duration-500 transition-opacity duration-300 ${loadingStates[activeTab] ? 'opacity-60' : ''}`}>
            <div className="text-center text-sm text-gray-400">
              {activeTab === 'guest' ? (
                guestPicks.length === 0 ? (
                  <p>Popular picks — tick anime you've watched or loved to personalize</p>
                ) : (
                  <p>Based on your {guestPicks.length} favorite anime</p>
                )
              ) : (
                <p>Based on {currentPool.meta.total_entries} entries from your list</p>
              )}
            </div>

            <ResultsSection
              variant="main"
              title="Main Recommendations"
              isLoading={loadingStates[activeTab]}
              totalCount={filteredMain.length}
              isFiltered={isFiltered}
              headerExtra={activeTab === 'guest' ? (
                <div className="hidden sm:flex items-center gap-3 mt-1">
                  <span className="text-xs text-gray-400">
                    Mark cards as seen or favorite to personalize
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
              ) : undefined}
              items={displayedMain}
              posters={posters}
              isGuestMode={activeTab === 'guest'}
              watchedSet={watchedSet}
              guestPicks={guestPicks}
              onSelect={setSelectedAnime}
              onSeen={toggleSeen}
              onLove={toggleLove}
            />

            <ResultsSection
              variant="cold"
              title="New & Trending (Cold)"
              isLoading={loadingStates[activeTab]}
              totalCount={filteredCold.length}
              isFiltered={isFiltered}
              items={displayedCold}
              posters={posters}
              isGuestMode={activeTab === 'guest'}
              watchedSet={watchedSet}
              guestPicks={guestPicks}
              onSelect={setSelectedAnime}
              onSeen={toggleSeen}
              onLove={toggleLove}
            />

            {currentPool.main.length === 0 && currentPool.cold.length === 0 && (
              <div className="text-center py-20 text-gray-500">
                No recommendations found.
              </div>
            )}
          </div>
        )}

        <AnimeModal
          anime={selectedAnime}
          entry={selectedAnime ? posters[selectedAnime.mal_id] : undefined}
          isOpen={!!selectedAnime}
          onClose={() => setSelectedAnime(null)}
        />
      </main>
    </div>
  );
}

export default App;
