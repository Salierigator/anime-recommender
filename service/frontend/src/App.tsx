import { useState } from 'react';
import { SearchForm } from './components/SearchForm';
import { AnimeCard } from './components/AnimeCard';
import { recommendAPI } from './api';
import type { RecommendRequest, RecommendResponse } from './types';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RecommendResponse | null>(null);

  const handleSearch = async (data: RecommendRequest) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await recommendAPI(data);
      setResult(response);
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
          
          <SearchForm onSubmit={handleSearch} isLoading={isLoading} />
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

        {result && !isLoading && (
          <div className="space-y-16 animate-in fade-in duration-500">
            {/* Meta Info */}
            <div className="text-center text-sm text-gray-400 space-y-1">
              <p>Based on {result.meta.history_count} items from history (Source: {result.meta.source})</p>
              <p>Mode: {result.meta.mode.toUpperCase()}</p>
            </div>

            {/* Main Recommendations */}
            {result.main.length > 0 && (
              <section>
                <div className="flex items-end justify-between mb-6 pb-2 border-b border-gray-100">
                  <h2 className="text-2xl font-bold tracking-tight">Main Recommendations</h2>
                  <span className="text-sm text-gray-500">{result.main.length} items</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {result.main.map((anime, idx) => (
                    <AnimeCard key={anime.mal_id} anime={anime} rank={idx + 1} />
                  ))}
                </div>
              </section>
            )}

            {/* Cold Recommendations */}
            {result.cold.length > 0 && (
              <section>
                <div className="flex items-end justify-between mb-6 pb-2 border-b border-gray-100">
                  <h2 className="text-2xl font-bold tracking-tight text-gray-600">New & Trending (Cold)</h2>
                  <span className="text-sm text-gray-400">{result.cold.length} items</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 opacity-90 hover:opacity-100 transition-opacity">
                  {result.cold.map((anime, idx) => (
                    <AnimeCard key={anime.mal_id} anime={anime} />
                  ))}
                </div>
              </section>
            )}

            {result.main.length === 0 && result.cold.length === 0 && (
              <div className="text-center py-20 text-gray-500">
                No recommendations found. Try adjusting your parameters.
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
