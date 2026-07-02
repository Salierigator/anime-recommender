/* eslint-disable react-hooks/set-state-in-effect */
import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Search, X, Check, User } from 'lucide-react';

interface FacetOptions {
  genres: string[];
  themes: string[];
  studios: string[];
  types: string[];
}

interface Props {
  onSubmit: (username: string) => void;
  isLoading: boolean;
  facetOptions: FacetOptions;
  
  // Genres
  selectedGenres: string[];
  setSelectedGenres: (genres: string[]) => void;
  
  // Types
  selectedTypes: string[];
  setSelectedTypes: (types: string[]) => void;
  
  // Themes
  selectedThemes: string[];
  setSelectedThemes: (themes: string[]) => void;
  
  // Studios
  selectedStudios: string[];
  setSelectedStudios: (studios: string[]) => void;
  
  // Score
  minScore: number;
  setMinScore: (score: number) => void;
  
  // Display counts
  mainK: number;
  setMainK: (k: number) => void;
  coldK: number;
  setColdK: (k: number) => void;
  
  hasPool: boolean;
}

interface MultiSelectProps {
  label: string;
  options: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
  showSearch?: boolean;
  single?: boolean;
  isCompact?: boolean;
}

function MultiSelectDropdown({
  label,
  options,
  selected,
  onChange,
  showSearch = false,
  single = false,
  isCompact = false
}: MultiSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [orderedOptions, setOrderedOptions] = useState<string[]>(options);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Snapshot order when dropdown opens: selected items first, rest after (both groups keep A→Z)
  useEffect(() => {
    if (isOpen) {
      const sel = options.filter(o => selected.includes(o));
      const rest = options.filter(o => !selected.includes(o));
      setOrderedOptions([...sel, ...rest]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  // Sync if options list changes (new pool loaded)
  useEffect(() => {
    setOrderedOptions(options);
  }, [options]);

  const toggleOption = (option: string) => {
    if (single) {
      if (label.startsWith('Show')) {
        // Enforce at least one selection for size selectors
        onChange([option]);
      } else {
        if (selected.includes(option)) {
          onChange([]);
        } else {
          onChange([option]);
        }
      }
      setIsOpen(false);
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

  return (
    <div className={`relative flex-1 ${isCompact ? 'min-w-[120px]' : 'min-w-[150px]'}`} ref={dropdownRef}>
      {!isCompact && (
        <span className="block text-xs font-semibold text-gray-500 uppercase tracking-wider mb-1">
          {label}
        </span>
      )}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full bg-white border border-gray-300 text-gray-900 flex items-center justify-between hover:border-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-900 cursor-pointer transition-colors ${
          isCompact ? 'px-2 py-1 text-xs' : 'px-3 py-2 text-sm'
        }`}
      >
        <span className="truncate">
          {isCompact ? `${label}: ${getButtonText()}` : getButtonText()}
        </span>
        <ChevronDown className={`${isCompact ? 'w-3.5 h-3.5' : 'w-4 h-4'} text-gray-500 ml-1.5 flex-shrink-0`} />
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
                  setIsOpen(false);
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
  hasPool
}: Props) {
  const [username, setUsername] = useState('');
  const [isSticky, setIsSticky] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  useEffect(() => {
    if (!hasPool) {
      setIsSticky(false);
      return;
    }

    const handleScroll = () => {
      // Threshold represents scrolling past the title & search input row
      setIsSticky(window.scrollY > 240);
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Check immediately in case page is already scrolled

    return () => window.removeEventListener('scroll', handleScroll);
  }, [hasPool]);

  const isCompact = isSticky && !isHovered;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (username.trim()) {
      onSubmit(username.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto space-y-6">
      {/* Search Bar Row */}
      <div className="flex gap-2 max-w-xl mx-auto w-full">
        <div className="relative flex-1">
          <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
            <User className="h-5 w-5 text-gray-400" />
          </div>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Enter MAL Username..."
            className="w-full pl-11 pr-4 py-3 bg-white border border-gray-300 text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-900 focus:border-gray-900 sm:text-base transition-shadow shadow-sm"
            disabled={isLoading}
          />
        </div>
        <button
          type="submit"
          disabled={isLoading || !username.trim()}
          className="w-32 py-3 bg-gray-900 text-white font-medium hover:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-gray-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
        >
          {isLoading ? 'Searching...' : 'Search'}
        </button>
      </div>

      {/* Client-side Filter Panel */}
      {hasPool && (
        <div className={isSticky ? 'h-[184px]' : ''}>
          <div
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
            className={`
              transition-all duration-300 ease-in-out
              ${isSticky 
                ? 'fixed top-0 left-0 right-0 bg-white border-b border-gray-200 shadow-md z-40' 
                : 'p-5 border border-gray-200 bg-gray-50'
              }
            `}
          >
            <div className={`max-w-6xl mx-auto transition-all duration-300 ${isSticky ? (isCompact ? 'py-2 px-4' : 'p-5') : ''} ${isCompact ? 'space-y-0' : 'space-y-6'}`}>
              {isCompact ? (
                <div className="flex flex-col md:flex-row md:items-center gap-4 w-full animate-in fade-in duration-200">
                  {/* 4 main dropdowns */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 flex-1">
                    <MultiSelectDropdown
                      label="Genre"
                      options={facetOptions.genres}
                      selected={selectedGenres}
                      onChange={setSelectedGenres}
                      isCompact={true}
                    />
                    <MultiSelectDropdown
                      label="Type"
                      options={facetOptions.types}
                      selected={selectedTypes}
                      onChange={setSelectedTypes}
                      single={true}
                      isCompact={true}
                    />
                    <MultiSelectDropdown
                      label="Theme"
                      options={facetOptions.themes}
                      selected={selectedThemes}
                      onChange={setSelectedThemes}
                      isCompact={true}
                    />
                    <MultiSelectDropdown
                      label="Studio"
                      options={facetOptions.studios}
                      selected={selectedStudios}
                      onChange={setSelectedStudios}
                      showSearch={true}
                      isCompact={true}
                    />
                  </div>
                  
                  {/* Score Slider (Compact) */}
                  <div className="flex items-center gap-2 flex-1 max-w-xs min-w-[150px]">
                    <div className="flex items-center gap-1.5 flex-shrink-0">
                      <span className="block text-[10px] font-semibold text-gray-500 uppercase tracking-wider">
                        Score ≥
                      </span>
                      <span className="text-[10px] font-mono font-semibold text-gray-900 bg-white border border-gray-200 px-1 py-0.5 rounded-sm">
                        {minScore.toFixed(1)}
                      </span>
                    </div>
                    <div className="h-6 flex items-center flex-grow">
                      <input
                        type="range"
                        min="0"
                        max="10"
                        step="0.5"
                        value={minScore}
                        onChange={(e) => setMinScore(parseFloat(e.target.value))}
                        className="w-full h-1 bg-gray-200 appearance-none cursor-pointer accent-gray-900 focus:outline-none"
                      />
                    </div>
                  </div>

                  {/* Show Main & Show Cold */}
                  <div className="grid grid-cols-2 gap-2 w-56 flex-shrink-0">
                    <MultiSelectDropdown
                      label="Show Main"
                      options={["10", "20", "50", "100", "250", "500"]}
                      selected={[String(mainK)]}
                      onChange={(val) => {
                        if (val.length > 0) setMainK(Number(val[0]));
                      }}
                      single={true}
                      isCompact={true}
                    />
                    <MultiSelectDropdown
                      label="Show Cold"
                      options={["5", "10", "20", "50", "100", "200"]}
                      selected={[String(coldK)]}
                      onChange={(val) => {
                        if (val.length > 0) setColdK(Number(val[0]));
                      }}
                      single={true}
                      isCompact={true}
                    />
                  </div>
                </div>
              ) : (
                <div className="space-y-6 animate-in fade-in duration-200 w-full">
                  {/* Multi-select Row */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                    <MultiSelectDropdown
                      label="Genre"
                      options={facetOptions.genres}
                      selected={selectedGenres}
                      onChange={setSelectedGenres}
                    />
                    <MultiSelectDropdown
                      label="Type"
                      options={facetOptions.types}
                      selected={selectedTypes}
                      onChange={setSelectedTypes}
                      single={true}
                    />
                    <MultiSelectDropdown
                      label="Theme"
                      options={facetOptions.themes}
                      selected={selectedThemes}
                      onChange={setSelectedThemes}
                    />
                    <MultiSelectDropdown
                      label="Studio"
                      options={facetOptions.studios}
                      selected={selectedStudios}
                      onChange={setSelectedStudios}
                      showSearch={true}
                    />
                  </div>

                  {/* Slider and Size Selectors Row */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2">
                    {/* Score Slider */}
                    <div className="w-full">
                      <div className="flex justify-between items-center mb-1.5">
                        <span className="block text-xs font-semibold text-gray-500 uppercase tracking-wider">
                          Score ≥
                        </span>
                        <span className="text-xs font-mono font-semibold text-gray-900 bg-white border border-gray-200 px-2 py-0.5 rounded-sm">
                          {minScore.toFixed(1)}
                        </span>
                      </div>
                      <div className="h-[38px] flex items-center">
                        <input
                          type="range"
                          min="0"
                          max="10"
                          step="0.5"
                          value={minScore}
                          onChange={(e) => setMinScore(parseFloat(e.target.value))}
                          className="w-full h-1 bg-gray-200 appearance-none cursor-pointer accent-gray-900 focus:outline-none"
                        />
                      </div>
                    </div>

                    {/* Sizes Selectors */}
                    <div className="grid grid-cols-2 gap-4 w-full">
                      <MultiSelectDropdown
                        label="Show Main"
                        options={["10", "20", "50", "100", "250", "500"]}
                        selected={[String(mainK)]}
                        onChange={(val) => {
                          if (val.length > 0) setMainK(Number(val[0]));
                        }}
                        single={true}
                      />
                      <MultiSelectDropdown
                        label="Show Cold"
                        options={["5", "10", "20", "50", "100", "200"]}
                        selected={[String(coldK)]}
                        onChange={(val) => {
                          if (val.length > 0) setColdK(Number(val[0]));
                        }}
                        single={true}
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </form>
  );
}
