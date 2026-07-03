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

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        handleSetIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Cleanup close timer on unmount
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
  hasPool
}: Props) {
  const [username, setUsername] = useState('');
  const [isSticky, setIsSticky] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const [reservedHeight, setReservedHeight] = useState<number | null>(null);
  const [openDropdowns, setOpenDropdowns] = useState<Record<string, boolean>>({});
  const panelRef = useRef<HTMLDivElement>(null);
  
  // Bug 1: Cursor coordinates tracking ref
  const mouseCoordsRef = useRef({ x: 0, y: 0 });

  // Pointermove listener active when isSticky is true
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
      // Threshold represents scrolling past the title & search input row
      setIsSticky(window.scrollY > 240);
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Check immediately in case page is already scrolled

    return () => window.removeEventListener('scroll', handleScroll);
  }, [hasPool]);

  // Measure expanded height when in-flow (not sticky)
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
    setOpenDropdowns(prev => ({
      ...prev,
      [label]: isOpen
    }));
  };

  const isAnyDropdownOpen = Object.values(openDropdowns).some(Boolean);

  // Bug 1: Unmount fallback check when dropdown closes
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

  const handleFocus = () => {
    setIsFocused(true);
  };

  const handleBlur = (e: React.FocusEvent) => {
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsFocused(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (username.trim()) {
      onSubmit(username.trim());
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-3xl mx-auto space-y-6">
      <style dangerouslySetInnerHTML={{__html: `
        .no-scrollbar::-webkit-scrollbar {
          display: none;
        }
      `}} />

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
            className="w-full pl-11 pr-4 py-3 bg-white border border-gray-300 text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-1 focus:ring-gray-900 focus:border-gray-900 sm:text-base transition-shadow shadow-sm rounded-none"
            disabled={isLoading}
          />
        </div>
        <button
          type="submit"
          disabled={isLoading || !username.trim()}
          className="w-32 py-3 bg-gray-900 text-white font-medium hover:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-gray-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer rounded-none"
        >
          {isLoading ? 'Searching...' : 'Search'}
        </button>
      </div>

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
            onFocus={handleFocus}
            onBlur={handleBlur}
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
              {/* Row 1: The 4 dropdowns (Genre, Type, Theme, Studio) */}
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

              {/* Row 2: Score slider on left, list sizes on right */}
              <div className={`transition-all duration-300 ease-in-out motion-reduce:transition-none ${
                isCompact 
                  ? 'flex flex-row items-center gap-2 flex-nowrap flex-shrink-0 pt-0' 
                  : 'grid grid-cols-1 md:grid-cols-2 gap-6 pt-2 w-full'
              }`}>
                {/* Score Slider */}
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

                {/* Sizes Selectors */}
                {!isCompact && (
                  <div className="grid grid-cols-2 gap-4 w-full transition-all duration-300 ease-in-out motion-reduce:transition-none">
                    <MultiSelectDropdown
                      label="Show Main"
                      options={["10", "20", "50", "100", "250", "500"]}
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
                      options={["5", "10", "20", "50", "100", "200"]}
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
