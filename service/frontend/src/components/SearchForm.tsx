/* eslint-disable react-hooks/set-state-in-effect */
import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Search, X, Check, User, ExternalLink, HelpCircle, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { searchAnimeAPI, checkUsernameExistsAPI } from '../api';
import type { SearchResultItem } from '../types';

interface FacetOptions {
  genres: string[];
  themes: string[];
  studios: string[];
  types: string[];
}

interface Props {
  onSubmit: (params: { username?: string; mal_ids?: number[] }, tab: 'username' | 'guest') => void;
  isLoading: boolean;
  facetOptions: FacetOptions;
  
  selectedGenres: string[];
  setSelectedGenres: (genres: string[]) => void;
  selectedTypes: string[];
  setSelectedTypes: (types: string[]) => void;
  selectedThemes: string[];
  setSelectedThemes: (themes: string[]) => void;
  selectedStudios: string[];
  setSelectedStudios: (studios: string[]) => void;
  minScore: number;
  setMinScore: (score: number) => void;
  
  mainK: number;
  setMainK: (k: number) => void;
  coldK: number;
  setColdK: (k: number) => void;
  
  hasPool: boolean;

  activeTab: 'username' | 'guest';
  setActiveTab: (tab: 'username' | 'guest') => void;
  guestPicks: SearchResultItem[];
  setGuestPicks: React.Dispatch<React.SetStateAction<SearchResultItem[]>>;
}

interface MultiSelectProps {
  label: string;
  options: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
  showSearch?: boolean;
  single?: boolean;
  onOpenStateChange?: (isOpen: boolean) => void;
  isCompact?: boolean;
}

function MultiSelectDropdown({
  label,
  options,
  selected,
  onChange,
  showSearch = false,
  single = false,
  onOpenStateChange,
  isCompact = false
}: MultiSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [orderedOptions, setOrderedOptions] = useState<string[]>(options);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleSetIsOpen = (open: boolean) => {
    setIsOpen(open);
    if (onOpenStateChange) {
      onOpenStateChange(open);
    }
    if (!open && closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  };

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        handleSetIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    return () => {
      if (closeTimerRef.current) {
        clearTimeout(closeTimerRef.current);
      }
    };
  }, []);

  const handleMouseEnter = () => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
  };

  const handleMouseLeave = () => {
    if (isOpen) {
      if (closeTimerRef.current) {
        clearTimeout(closeTimerRef.current);
      }
      closeTimerRef.current = setTimeout(() => {
        handleSetIsOpen(false);
      }, 150);
    }
  };

  useEffect(() => {
    if (isOpen) {
      const sel = options.filter(o => selected.includes(o));
      const rest = options.filter(o => !selected.includes(o));
      setOrderedOptions([...sel, ...rest]);
    }
  }, [isOpen]);

  useEffect(() => {
    setOrderedOptions(options);
  }, [options]);

  const toggleOption = (option: string) => {
    if (single) {
      if (label.startsWith('Show')) {
        onChange([option]);
      } else {
        if (selected.includes(option)) {
          onChange([]);
        } else {
          onChange([option]);
        }
      }
      handleSetIsOpen(false);
    } else {
      if (selected.includes(option)) {
        onChange(selected.filter(item => item !== option));
      } else {
        onChange([...selected, option]);
      }
    }
  };

  const getOptionLabel = (option: string) => {
    if (label === 'Type' && option === '?') return 'Unknown';
    if (label.startsWith('Show')) return `Top ${option}`;
    return option;
  };

  const getButtonText = () => {
    if (selected.length === 0) return `All ${label}s`;
    if (single) return getOptionLabel(selected[0]);
    return `${selected.length} selected`;
  };

  const filteredOptions = showSearch
    ? orderedOptions.filter(opt => opt.toLowerCase().includes(searchQuery.toLowerCase()))
    : orderedOptions;

  const showAllOption = single && !label.startsWith('Show');
  const isActive = selected.length > 0;

  return (
    <div 
      className={`relative flex-1 transition-all duration-300 ease-in-out motion-reduce:transition-none ${
        isCompact ? 'min-w-[100px] max-w-[200px] flex-shrink-0' : 'min-w-[150px]'
      }`} 
      ref={dropdownRef}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <span 
        className={`block font-semibold text-gray-500 uppercase tracking-wider transition-all duration-300 ease-in-out motion-reduce:transition-none ${
          isCompact 
            ? 'h-0 overflow-hidden opacity-0 mb-0 pointer-events-none text-[0px]' 
            : 'text-xs mb-1'
        }`}
      >
        {label}
      </span>
      <button
        type="button"
        onClick={() => handleSetIsOpen(!isOpen)}
        className={`w-full border flex items-center justify-between focus:outline-none focus:ring-1 focus:ring-gray-900 cursor-pointer transition-all duration-300 ease-in-out motion-reduce:transition-none ${
          isCompact ? 'px-2 py-1 text-xs' : 'px-3 py-2 text-sm'
        } ${
          isCompact
            ? isActive
              ? 'border-gray-900 text-gray-900 font-semibold bg-white'
              : 'border-gray-300 text-gray-500 bg-white hover:border-gray-400'
            : 'border-gray-300 text-gray-900 bg-white hover:border-gray-400'
        }`}
      >
        <span className="truncate flex items-center">
          {isCompact && (
            <span className={`text-[10px] uppercase tracking-wider mr-1.5 flex-shrink-0 ${
              isActive ? 'text-gray-900 font-bold' : 'text-gray-400 font-bold'
            }`}>
              {label}:
            </span>
          )}
          <span className={`truncate text-left ${isCompact && isActive ? 'font-semibold' : ''}`}>
            {getButtonText()}
          </span>
        </span>
        <ChevronDown className={`w-3.5 h-3.5 ml-1 flex-shrink-0 ${
          isCompact
            ? isActive ? 'text-gray-900' : 'text-gray-400'
            : 'text-gray-500'
        }`} />
      </button>

      {isOpen && (
        <div className="absolute left-0 right-0 mt-1 bg-white border border-gray-300 shadow-md z-40 max-h-60 overflow-y-auto animate-in fade-in duration-100">
          {showSearch && (
            <div className="p-2 border-b border-gray-100 flex items-center bg-gray-50 sticky top-0 z-10">
              <Search className="w-3.5 h-3.5 text-gray-400 mr-2 flex-shrink-0" />
              <input
                type="text"
                placeholder={`Search ${label}...`}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full text-xs bg-transparent focus:outline-none placeholder-gray-400 text-gray-900"
              />
              {searchQuery && (
                <button
                  type="button"
                  onClick={() => setSearchQuery('')}
                  className="p-0.5 hover:bg-gray-200 text-gray-500 rounded-full"
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>
          )}
          <div className="py-1">
            {showAllOption && (
              <button
                type="button"
                onClick={() => {
                  onChange([]);
                  handleSetIsOpen(false);
                }}
                className="w-full px-3 py-1.5 text-xs text-left text-gray-700 hover:bg-gray-100 flex items-center justify-between cursor-pointer transition-colors"
              >
                <span className="truncate">All {label}s</span>
                {selected.length === 0 && <Check className="w-3.5 h-3.5 text-gray-900 flex-shrink-0" />}
              </button>
            )}
            {filteredOptions.length === 0 ? (
              !showAllOption && <div className="px-3 py-2 text-xs text-gray-400">No options found</div>
            ) : (
              filteredOptions.map((option) => {
                const isSelected = selected.includes(option);
                const displayLabel = getOptionLabel(option);
                return (
                  <button
                    key={option}
                    type="button"
                    onClick={() => toggleOption(option)}
                    className="w-full px-3 py-1.5 text-xs text-left text-gray-700 hover:bg-gray-100 flex items-center justify-between cursor-pointer transition-colors"
                  >
                    <span className="truncate mr-2">{displayLabel}</span>
                    {isSelected && <Check className="w-3.5 h-3.5 text-gray-900 flex-shrink-0" />}
                  </button>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export function SearchForm({
  onSubmit,
  isLoading,
  facetOptions,
  selectedGenres,
  setSelectedGenres,
  selectedTypes,
  setSelectedTypes,
  selectedThemes,
  setSelectedThemes,
  selectedStudios,
  setSelectedStudios,
  minScore,
  setMinScore,
  mainK,
  setMainK,
  coldK,
  setColdK,
  hasPool,
  activeTab,
  setActiveTab,
  guestPicks,
  setGuestPicks
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

  const [isSticky, setIsSticky] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const [reservedHeight, setReservedHeight] = useState<number | null>(null);
  const [openDropdowns, setOpenDropdowns] = useState<Record<string, boolean>>({});
  const panelRef = useRef<HTMLDivElement>(null);

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
  
  const mouseCoordsRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    if (!isSticky) {
      setIsHovered(false);
      return;
    }

    const handlePointerMove = (e: PointerEvent) => {
      mouseCoordsRef.current = { x: e.clientX, y: e.clientY };
      if (panelRef.current) {
        const rect = panelRef.current.getBoundingClientRect();
        const isInside = (
          e.clientX >= rect.left &&
          e.clientX <= rect.right &&
          e.clientY >= rect.top &&
          e.clientY <= rect.bottom
        );
        setIsHovered(isInside);
      }
    };

    window.addEventListener('pointermove', handlePointerMove);
    return () => window.removeEventListener('pointermove', handlePointerMove);
  }, [isSticky]);

  useEffect(() => {
    if (!hasPool) {
      setIsSticky(false);
      return;
    }

    const handleScroll = () => {
      setIsSticky(window.scrollY > 240);
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); 

    return () => window.removeEventListener('scroll', handleScroll);
  }, [hasPool]);

  useEffect(() => {
    if (!hasPool || isSticky) return;

    const measureHeight = () => {
      if (panelRef.current) {
        setReservedHeight(panelRef.current.offsetHeight);
      }
    };

    measureHeight();
    window.addEventListener('resize', measureHeight);
    return () => window.removeEventListener('resize', measureHeight);
  }, [hasPool, isSticky, facetOptions]);

  const handleDropdownOpenChange = (label: string, isOpen: boolean) => {
    setOpenDropdowns(prev => ({ ...prev, [label]: isOpen }));
  };

  const isAnyDropdownOpen = Object.values(openDropdowns).some(Boolean);
  const prevAnyDropdownOpen = useRef(false);

  useEffect(() => {
    const wasOpen = prevAnyDropdownOpen.current;
    const isOpen = isAnyDropdownOpen;
    prevAnyDropdownOpen.current = isOpen;

    if (wasOpen && !isOpen && isSticky) {
      requestAnimationFrame(() => {
        if (panelRef.current) {
          const rect = panelRef.current.getBoundingClientRect();
          const { x, y } = mouseCoordsRef.current;
          const isInside = (
            x >= rect.left &&
            x <= rect.right &&
            y >= rect.top &&
            y <= rect.bottom
          );
          setIsHovered(isInside);
        }
      });
    }
  }, [isAnyDropdownOpen, isSticky]);

  const isCompact = isSticky && !isHovered && !isFocused && !isAnyDropdownOpen;

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
      <style dangerouslySetInnerHTML={{__html: `
        .no-scrollbar::-webkit-scrollbar {
          display: none;
        }
      `}} />

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
                disabled={isLoading}
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
                disabled={isLoading}
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
        <button
          type="submit"
          disabled={isLoading || (activeTab === 'username' ? !username.trim() : guestPicks.length === 0)}
          className="w-48 py-3 bg-gray-900 text-white font-medium hover:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-gray-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer rounded-none whitespace-nowrap"
        >
          {isLoading ? 'Loading...' : activeTab === 'username' ? 'Search' : 'Get Nut :)'}
        </button>
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

      {hasPool && (
        <div
          style={{ height: (isSticky && reservedHeight) ? `${reservedHeight}px` : 'auto' }}
          className="transition-[height] duration-300 ease-in-out motion-reduce:transition-none"
        >
          <div
            ref={panelRef}
            onMouseEnter={() => {
              if (!isSticky) setIsHovered(true);
            }}
            onMouseLeave={() => {
              if (!isSticky) setIsHovered(false);
            }}
            onFocus={() => setIsFocused(true)}
            onBlur={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget)) setIsFocused(false);
            }}
            className={`
              transition-all duration-300 ease-in-out motion-reduce:transition-none rounded-none
              ${isSticky 
                ? 'fixed top-0 left-0 right-0 w-full bg-white shadow-md z-40 border-0 border-b border-gray-200' 
                : 'bg-gray-50 max-w-3xl mx-auto border border-gray-200'
              }
              ${isCompact ? 'py-1.5' : 'p-5'}
            `}
          >
            <div
              className={`w-full transition-all duration-300 ease-in-out motion-reduce:transition-none
                ${isSticky ? 'max-w-3xl mx-auto px-4' : ''}
                ${isCompact 
                  ? 'flex flex-row items-center gap-2 flex-nowrap overflow-x-auto no-scrollbar' 
                  : 'flex flex-col gap-6'
                }
              `}
              style={{
                scrollbarWidth: isCompact ? 'none' : 'auto',
                msOverflowStyle: isCompact ? 'none' : 'auto',
              }}
            >
              <div className={`transition-all duration-300 ease-in-out motion-reduce:transition-none ${
                isCompact 
                  ? 'flex flex-row items-center gap-2 flex-nowrap flex-shrink-0' 
                  : 'grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 w-full'
              }`}>
                <MultiSelectDropdown
                  label="Genre"
                  options={facetOptions.genres}
                  selected={selectedGenres}
                  onChange={setSelectedGenres}
                  onOpenStateChange={(open) => handleDropdownOpenChange("Genre", open)}
                  isCompact={isCompact}
                />
                <MultiSelectDropdown
                  label="Type"
                  options={facetOptions.types}
                  selected={selectedTypes}
                  onChange={setSelectedTypes}
                  single={true}
                  onOpenStateChange={(open) => handleDropdownOpenChange("Type", open)}
                  isCompact={isCompact}
                />
                <MultiSelectDropdown
                  label="Theme"
                  options={facetOptions.themes}
                  selected={selectedThemes}
                  onChange={setSelectedThemes}
                  onOpenStateChange={(open) => handleDropdownOpenChange("Theme", open)}
                  isCompact={isCompact}
                />
                <MultiSelectDropdown
                  label="Studio"
                  options={facetOptions.studios}
                  selected={selectedStudios}
                  onChange={setSelectedStudios}
                  showSearch={true}
                  onOpenStateChange={(open) => handleDropdownOpenChange("Studio", open)}
                  isCompact={isCompact}
                />
              </div>

              <div className={`transition-all duration-300 ease-in-out motion-reduce:transition-none ${
                isCompact 
                  ? 'flex flex-row items-center gap-2 flex-nowrap flex-shrink-0 pt-0' 
                  : 'grid grid-cols-1 md:grid-cols-2 gap-6 pt-2 w-full'
              }`}>
                <div className={`transition-all duration-300 ease-in-out motion-reduce:transition-none ${
                  isCompact ? 'flex items-center gap-1.5 flex-shrink-0 w-auto' : 'w-full'
                }`}>
                  <div className={`transition-all duration-300 ease-in-out motion-reduce:transition-none ${
                    isCompact ? 'flex items-center gap-1 mb-0' : 'flex justify-between items-center mb-1.5'
                  }`}>
                    <span className={`font-semibold uppercase tracking-wider flex-shrink-0 transition-all duration-300 ease-in-out motion-reduce:transition-none ${
                      isCompact 
                        ? minScore > 0 ? 'text-gray-900 font-semibold text-[10px] mr-0.5' : 'text-gray-400 text-[10px] mr-0.5'
                        : 'text-xs text-gray-500'
                    }`}>
                      Score ≥
                    </span>
                    <span className={`font-mono font-semibold bg-white border rounded-none transition-all duration-300 ease-in-out motion-reduce:transition-none ${
                      isCompact ? 'px-1 py-0 text-[10px]' : 'px-2 py-0.5 text-xs'
                    } ${
                      isCompact
                        ? minScore > 0 ? 'border-gray-900 text-gray-900 font-semibold' : 'border-gray-300 text-gray-500'
                        : 'border-gray-200 text-gray-900'
                    }`}>
                      {minScore.toFixed(1)}
                    </span>
                  </div>
                  <div className={`flex items-center transition-all duration-300 ease-in-out motion-reduce:transition-none ${
                    isCompact ? 'h-[26px] w-16 px-1' : 'h-[38px] w-full'
                  }`}>
                    <input
                      type="range"
                      min="0"
                      max="10"
                      step="0.1"
                      value={minScore}
                      onChange={(e) => setMinScore(parseFloat(e.target.value))}
                      className="w-full h-1 bg-gray-200 appearance-none cursor-pointer accent-gray-900 focus:outline-none"
                    />
                  </div>
                </div>

                {!isCompact && (
                  <div className="grid grid-cols-2 gap-4 w-full transition-all duration-300 ease-in-out motion-reduce:transition-none">
                    <MultiSelectDropdown
                      label="Show Main"
                      options={["10", "20", "50", "100", "200"]}
                      selected={[String(mainK)]}
                      onChange={(val) => {
                        if (val.length > 0) setMainK(Number(val[0]));
                      }}
                      single={true}
                      onOpenStateChange={(open) => handleDropdownOpenChange("Show Main", open)}
                      isCompact={isCompact}
                    />
                    <MultiSelectDropdown
                      label="Show Cold"
                      options={["5", "10", "20", "50", "100"]}
                      selected={[String(coldK)]}
                      onChange={(val) => {
                        if (val.length > 0) setColdK(Number(val[0]));
                      }}
                      single={true}
                      onOpenStateChange={(open) => handleDropdownOpenChange("Show Cold", open)}
                      isCompact={isCompact}
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </form>
  );
}
