import { useState } from 'react';
import { ChevronDown, ChevronUp, Search, User, Hash } from 'lucide-react';
import type { RecommendRequest } from '../types';

interface Props {
  onSubmit: (data: RecommendRequest) => void;
  isLoading: boolean;
}

export function SearchForm({ onSubmit, isLoading }: Props) {
  const [username, setUsername] = useState('');
  const [anchorMalId, setAnchorMalId] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [malIds, setMalIds] = useState('');
  const [topK, setTopK] = useState(20);
  const [coldK, setColdK] = useState(10);
  
  // "live" is usually fixed to false or derived, but we can add it if needed. 
  // Contract: "live": false -> ép fetch MAL dù username có trong dataset
  // Let's hide it from the user to keep it simple, or add a checkbox if needed.
  // Actually, we'll just not send it, backend defaults apply.
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    let ids: number[] | null = null;
    if (malIds.trim()) {
      ids = malIds.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
    }
    
    const parsedAnchor = parseInt(anchorMalId.trim(), 10);
    
    onSubmit({
      username: username.trim() || null,
      mal_ids: ids && ids.length > 0 ? ids : null,
      top_k: topK,
      cold_k: coldK,
      anchor_mal_id: !isNaN(parsedAnchor) ? parsedAnchor : null,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto space-y-4">
      {/* Main Search Inputs */}
      <div className="flex flex-col md:flex-row gap-4">
        {/* Username Input */}
        <div className="relative flex-1">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <User className="h-5 w-5 text-gray-400" />
          </div>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Enter MAL Username..."
            className="w-full pl-11 pr-4 py-4 bg-white border border-gray-300 text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-900 focus:border-gray-900 sm:text-lg transition-shadow shadow-sm"
            disabled={isLoading}
          />
        </div>

        {/* Anchor MAL ID Input */}
        <div className="relative flex-1">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <Hash className="h-5 w-5 text-gray-400" />
          </div>
          <input
            type="number"
            value={anchorMalId}
            onChange={(e) => setAnchorMalId(e.target.value)}
            placeholder="Similar to Anime (MAL ID)..."
            className="w-full pl-11 pr-4 py-4 bg-white border border-gray-300 text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-900 focus:border-gray-900 sm:text-lg transition-shadow shadow-sm"
            disabled={isLoading}
          />
        </div>
      </div>

      <button
        type="submit"
        disabled={isLoading || (!username.trim() && !malIds.trim())}
        className="w-full px-6 py-4 bg-gray-900 text-white font-medium hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {isLoading ? 'Searching...' : 'Search'}
      </button>

      {/* Advanced Toggle */}
      <div className="flex justify-end">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center text-sm text-gray-500 hover:text-gray-900 transition-colors"
        >
          {showAdvanced ? (
            <><ChevronUp className="w-4 h-4 mr-1" /> Hide Advanced</>
          ) : (
            <><ChevronDown className="w-4 h-4 mr-1" /> Advanced Options</>
          )}
        </button>
      </div>

      {/* Advanced Options Panel */}
      {showAdvanced && (
        <div className="p-5 border border-gray-200 bg-gray-50 grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="md:col-span-2">
            <label htmlFor="malIds" className="block text-sm font-medium text-gray-700 mb-1">
              MAL IDs (comma-separated, ignores username)
            </label>
            <textarea
              id="malIds"
              value={malIds}
              onChange={(e) => setMalIds(e.target.value)}
              placeholder="e.g. 52991, 11061, 28977"
              rows={2}
              className="w-full p-2 border border-gray-300 focus:outline-none focus:border-gray-900 focus:ring-1 focus:ring-gray-900 text-sm"
            />
          </div>
          
          <div>
            <label htmlFor="topK" className="block text-sm font-medium text-gray-700 mb-1">
              Main Recommendations (Top K)
            </label>
            <input
              type="number"
              id="topK"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value) || 0)}
              min={1}
              max={100}
              className="w-full p-2 border border-gray-300 focus:outline-none focus:border-gray-900 focus:ring-1 focus:ring-gray-900 text-sm"
            />
          </div>

          <div>
            <label htmlFor="coldK" className="block text-sm font-medium text-gray-700 mb-1">
              New Anime (Cold K)
            </label>
            <input
              type="number"
              id="coldK"
              value={coldK}
              onChange={(e) => setColdK(parseInt(e.target.value) || 0)}
              min={0}
              max={100}
              className="w-full p-2 border border-gray-300 focus:outline-none focus:border-gray-900 focus:ring-1 focus:ring-gray-900 text-sm"
            />
          </div>
        </div>
      )}
    </form>
  );
}
