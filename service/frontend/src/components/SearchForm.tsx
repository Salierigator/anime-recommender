/* eslint-disable react-hooks/set-state-in-effect */
import { useState, useRef, useEffect } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import { Search, X, User, ExternalLink, HelpCircle, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { searchAnimeAPI, checkUsernameExistsAPI } from '../api';
import { FilterPanel } from './FilterPanel';
import type { FacetOptions, SearchResultItem, Tab, TabPrefs } from '../types';
import type { UpdatePrefs } from '../hooks/useTabPrefs';

interface Props {
  onSubmit: (params: { username?: string; mal_ids?: number[] }, tab: Tab) => void;
  onCancel: () => void;
  isLoading: boolean;
  hasPool: boolean;
  activeTab: Tab;
  setActiveTab: (tab: Tab) => void;
  guestPicks: SearchResultItem[];
  setGuestPicks: Dispatch<SetStateAction<SearchResultItem[]>>;
  facetOptions: FacetOptions;
  prefs: TabPrefs;
  updatePrefs: UpdatePrefs;
}

export function SearchForm({
  onSubmit,
  onCancel,
  isLoading,
  hasPool,
  activeTab,
  setActiveTab,
  guestPicks,
  setGuestPicks,
  facetOptions,
  prefs,
  updatePrefs
}: Props) {
  const [username, setUsername] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('u') || '';
  });
  const [lastSearchedUsername, setLastSearchedUsername] = useState(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get('u') || '';
  });

  const [usernameStatus, setUsernameStatus] = useState<'idle' | 'valid' | 'invalid'>('idle');
  const userAbortControllerRef = useRef<AbortController | null>(null);

  // Guest mode state
  const [guestQuery, setGuestQuery] = useState('');
  const [guestResults, setGuestResults] = useState<SearchResultItem[]>([]);
  const [guestSearching, setGuestSearching] = useState(false);
  const [showGuestDropdown, setShowGuestDropdown] = useState(false);
  const searchDropdownRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const queryCacheRef = useRef<Map<string, SearchResultItem[]>>(new Map());

  // Username validation
  useEffect(() => {
    if (activeTab !== 'username' || username.trim().length < 2) {
      setUsernameStatus('idle');
      if (userAbortControllerRef.current) userAbortControllerRef.current.abort();
      return;
    }

    if (userAbortControllerRef.current) userAbortControllerRef.current.abort();
    const abortController = new AbortController();
    userAbortControllerRef.current = abortController;

    const timer = setTimeout(async () => {
      try {
        const res = await checkUsernameExistsAPI(username, abortController.signal);
        if (res.exists) {
          setUsernameStatus('valid');
        } else {
          setUsernameStatus('invalid');
        }
      } catch (err: unknown) {
        const errorName = (err as { name?: string })?.name;
        if (errorName === 'CanceledError' || errorName === 'AbortError') return;
        setUsernameStatus('idle'); // Network error, status 502, stay silent
      }
    }, 600);
    return () => {
      clearTimeout(timer);
      abortController.abort();
    };
  }, [username, activeTab]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchDropdownRef.current && !searchDropdownRef.current.contains(event.target as Node)) {
        setShowGuestDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Guest Search
  useEffect(() => {
    if (activeTab !== 'guest' || guestQuery.length < 2) {
      if (!guestSearching) setGuestResults([]); // keep results if still loading
      setShowGuestDropdown(false);
      return;
    }

    const normalizedQuery = guestQuery.trim().toLowerCase();

    if (queryCacheRef.current.has(normalizedQuery)) {
      setGuestResults(queryCacheRef.current.get(normalizedQuery)!);
      setShowGuestDropdown(true);
      setGuestSearching(false);
      return;
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    const timer = setTimeout(async () => {
      setGuestSearching(true);
      setShowGuestDropdown(true);
      try {
        const data = await searchAnimeAPI(guestQuery, 10, abortController.signal);
        queryCacheRef.current.set(normalizedQuery, data.results);
        setGuestResults(data.results);
      } catch (err: unknown) {
        const errorName = (err as { name?: string })?.name;
        if (errorName !== 'CanceledError' && errorName !== 'AbortError') {
          console.error("Search error", err);
        }
      } finally {
        if (!abortController.signal.aborted) {
          setGuestSearching(false);
        }
      }
    }, 400);

    return () => {
      clearTimeout(timer);
      abortController.abort();
    };
  }, [guestQuery, activeTab]);

  const addGuestPick = (item: SearchResultItem) => {
    if (!item.in_corpus) return;
    if (!guestPicks.some(p => p.mal_id === item.mal_id)) {
      setGuestPicks([...guestPicks, item]);
    }
    setGuestQuery('');
    setShowGuestDropdown(false);
  };

  const removeGuestPick = (malId: number) => {
    setGuestPicks(guestPicks.filter(p => p.mal_id !== malId));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (activeTab === 'username') {
      if (username.trim()) {
        setLastSearchedUsername(username.trim());
        onSubmit({ username: username.trim() }, 'username');
      }
    } else {
      if (guestPicks.length > 0) {
        onSubmit({ mal_ids: guestPicks.map(p => p.mal_id) }, 'guest');
      }
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto space-y-6">
      {/* Tabs */}
      <div className="flex justify-center mb-6">
        <div className="inline-flex bg-gray-100 p-1 border border-gray-200">
          <button
            type="button"
            onClick={() => setActiveTab('username')}
            className={`px-6 py-2 text-sm font-semibold transition-colors rounded-none cursor-pointer ${
              activeTab === 'username' ? 'bg-white text-gray-900 shadow-sm border border-gray-200' : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            MAL Username
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('guest')}
            className={`px-6 py-2 text-sm font-semibold transition-colors rounded-none cursor-pointer ${
              activeTab === 'guest' ? 'bg-white text-gray-900 shadow-sm border border-gray-200' : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Pick Favorites
          </button>
        </div>
      </div>

      <div className="flex gap-2 max-w-xl mx-auto w-full">
        <div className="relative flex-1" ref={searchDropdownRef}>
          {activeTab === 'username' ? (
            <>
              <div className="absolute inset-y-0 left-0 flex items-center z-10 pointer-events-none">
                {username === lastSearchedUsername && lastSearchedUsername.length > 0 ? (
                  <a
                    href={`https://myanimelist.net/profile/${username}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="h-full flex items-center px-4 hover:bg-gray-100 transition-colors pointer-events-auto border-r border-transparent hover:border-gray-200 group"
                    title="View MAL Profile"
                  >
                    <ExternalLink className="h-5 w-5 text-gray-400 group-hover:text-gray-900 transition-colors" />
                  </a>
                ) : (
                  <div className="h-full flex items-center px-4">
                    <User className="h-5 w-5 text-gray-400" />
                  </div>
                )}
              </div>

              <input
                type="text"
                value={username}
                onChange={(e) => {
                  setUsername(e.target.value);
                  setUsernameStatus('idle');
                }}
                placeholder="Enter MAL Username..."
                className={`w-full ${(username === lastSearchedUsername && lastSearchedUsername.length > 0) ? 'pl-[3.25rem]' : 'pl-11'} pr-10 py-3 bg-white border border-gray-300 text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-900 focus:border-gray-900 sm:text-base transition-shadow shadow-sm rounded-none`}
              />

              <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                {usernameStatus === 'valid' && <CheckCircle2 className="w-5 h-5 text-green-500" />}
                {usernameStatus === 'invalid' && (
                  <div className="group relative">
                    <XCircle className="w-5 h-5 text-red-500 pointer-events-auto cursor-help" />
                    <div className="hidden group-hover:block absolute right-0 top-6 w-32 bg-gray-900 text-white text-[10px] p-1.5 z-10 text-center rounded-sm">
                      Username not found
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                value={guestQuery}
                onChange={(e) => {
                  setGuestQuery(e.target.value);
                  if (e.target.value.length >= 2) setShowGuestDropdown(true);
                }}
                onFocus={() => {
                  if (guestQuery.length >= 2 && guestResults.length > 0) setShowGuestDropdown(true);
                }}
                placeholder="Search anime to add..."
                className="w-full pl-11 pr-10 py-3 bg-white border border-gray-300 text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-900 focus:border-gray-900 sm:text-base transition-shadow shadow-sm rounded-none"
              />
              <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                {guestSearching && <Loader2 className="w-4 h-4 text-gray-400 animate-spin" />}
              </div>

              {/* Guest Search Dropdown */}
              {showGuestDropdown && (guestResults.length > 0 || guestSearching) && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-300 shadow-xl z-50 max-h-80 overflow-y-auto">
                  {guestResults.map(item => (
                    <button
                      key={item.mal_id}
                      type="button"
                      onClick={() => addGuestPick(item)}
                      disabled={!item.in_corpus}
                      className={`w-full flex items-center gap-3 p-2 text-left border-b border-gray-100 last:border-0 transition-colors ${
                        !item.in_corpus ? 'opacity-50 cursor-not-allowed bg-gray-50' : 'hover:bg-gray-50 cursor-pointer'
                      }`}
                    >
                      {item.image_url ? (
                        <img src={item.image_url} alt="" className="w-10 h-14 object-cover flex-shrink-0 bg-gray-200" />
                      ) : (
                        <div className="w-10 h-14 bg-gray-200 flex-shrink-0" />
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="font-semibold text-gray-900 truncate">{item.title}</div>
                        {item.title_english && <div className="text-xs text-gray-500 truncate">{item.title_english}</div>}
                        <div className="text-[10px] text-gray-400 mt-1 uppercase flex items-center gap-2">
                          <span>{item.type || '?'} {item.year ? `· ${item.year}` : ''}</span>
                          {item.mal_score && <span>★ {item.mal_score}</span>}
                        </div>
                      </div>
                      {!item.in_corpus && (
                        <div className="flex-shrink-0 px-2 group relative">
                          <HelpCircle className="w-4 h-4 text-gray-400" />
                          <div className="hidden group-hover:block absolute right-0 top-6 w-32 bg-gray-900 text-white text-[10px] p-1.5 z-10 text-center pointer-events-none">
                            Not in model yet
                          </div>
                        </div>
                      )}
                    </button>
                  ))}
                  {guestSearching && guestResults.length === 0 && (
                    <div className="p-4 text-center text-sm text-gray-500">Searching...</div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
        {isLoading ? (
          <button
            key="stop"
            type="button"
            onClick={(e) => {
              // preventDefault: khi state đổi giữa lúc dispatch, React morph node này thành
              // nút submit → default action của click sẽ submit form ngay sau cancel
              e.preventDefault();
              onCancel();
            }}
            title="Stop the current search"
            className="w-48 py-3 bg-white border border-gray-900 text-gray-900 font-medium hover:bg-gray-100 focus:outline-none focus:ring-1 focus:ring-gray-900 transition-colors cursor-pointer rounded-none whitespace-nowrap"
          >
            Stop
          </button>
        ) : (
          <button
            key="search"
            type="submit"
            disabled={activeTab === 'username' ? !username.trim() : guestPicks.length === 0}
            className="w-48 py-3 bg-gray-900 text-white font-medium hover:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-gray-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer rounded-none whitespace-nowrap"
          >
            {activeTab === 'username' ? 'Get Goods :)' : 'Git Gud :)'}
          </button>
        )}
      </div>

      {activeTab === 'guest' && (
        <div className="max-w-xl mx-auto w-full space-y-2">
          {guestPicks.length === 0 ? (
            <div className="text-center text-sm text-gray-400">Search and pick at least 1 anime. (Recommended: 5+)</div>
          ) : (
            <div className="flex flex-wrap w-full gap-2">
              {guestPicks.map(pick => (
                <div key={pick.mal_id} className="flex items-center bg-white border border-gray-300 pr-1 overflow-hidden h-10 group">
                  {pick.image_url ? (
                    <img src={pick.image_url} alt="" className="h-full w-7 object-cover mr-2" />
                  ) : (
                    <div className="h-full w-7 bg-gray-200 mr-2" />
                  )}
                  <span className="text-xs font-medium max-w-[120px] truncate mr-2" title={pick.title}>{pick.title}</span>
                  <button
                    type="button"
                    onClick={() => removeGuestPick(pick.mal_id)}
                    className="p-1 text-gray-400 hover:text-gray-900 hover:bg-gray-100 cursor-pointer transition-colors"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <FilterPanel
        hasPool={hasPool}
        facetOptions={facetOptions}
        prefs={prefs}
        updatePrefs={updatePrefs}
      />
    </form>
  );
}
