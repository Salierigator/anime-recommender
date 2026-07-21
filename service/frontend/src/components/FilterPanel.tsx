/* eslint-disable react-hooks/set-state-in-effect */
import { useState, useRef, useEffect } from 'react';
import { ArrowUp, ArrowDown } from 'lucide-react';
import { MultiSelectDropdown } from './MultiSelectDropdown';
import type { FacetOptions, SortKey, TabPrefs } from '../types';
import type { UpdatePrefs } from '../hooks/useTabPrefs';

interface Props {
  hasPool: boolean;
  facetOptions: FacetOptions;
  prefs: TabPrefs;
  updatePrefs: UpdatePrefs;
}

const sortOptionsMap: Record<SortKey, string> = {
  relevance: 'Relevance',
  score: 'MAL Score',
  popularity: 'Popularity',
  date: 'Release Date',
};

function getSortDirectionLabel(key: SortKey, isAsc: boolean) {
  switch (key) {
    case 'relevance':
      return isAsc ? 'Reversed Order' : 'Original Order';
    case 'score':
      return isAsc ? 'Lowest Score' : 'Highest Score';
    case 'popularity':
      return isAsc ? 'Least Popular' : 'Most Popular';
    case 'date':
      return isAsc ? 'Oldest First' : 'Newest First';
    default:
      return isAsc ? 'Ascending' : 'Descending';
  }
}

/**
 * Panel filter/sort/số hiển thị dưới form search. Bình thường nằm trong flow;
 * cuộn quá 240px thì ghim lên đỉnh (sticky) và thu gọn (compact) khi không hover/focus.
 */
export function FilterPanel({ hasPool, facetOptions, prefs, updatePrefs }: Props) {
  const [isSticky, setIsSticky] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const [reservedHeight, setReservedHeight] = useState<number | null>(null);
  const [openDropdowns, setOpenDropdowns] = useState<Record<string, boolean>>({});
  const panelRef = useRef<HTMLDivElement>(null);
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

  const handleSortByChange = (newSortBy: SortKey) => {
    updatePrefs({
      sortBy: newSortBy,
      sortAsc: false,
    });
  };

  const handleSortLabelSelect = (val: string[]) => {
    if (val.length > 0) {
      const selectedLabel = val[0];
      const selectedKey = Object.keys(sortOptionsMap).find(
        key => sortOptionsMap[key as SortKey] === selectedLabel
      ) as SortKey;
      if (selectedKey) handleSortByChange(selectedKey);
    }
  };

  if (!hasPool) return null;

  return (
    <div
      style={{ height: (isSticky && reservedHeight) ? `${reservedHeight}px` : 'auto' }}
      className="transition-[height] duration-300 ease-in-out motion-reduce:transition-none"
    >
      <style dangerouslySetInnerHTML={{__html: `
        .no-scrollbar::-webkit-scrollbar {
          display: none;
        }
      `}} />
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
              selected={prefs.genres}
              onChange={(val) => updatePrefs({ genres: val })}
              onOpenStateChange={(open) => handleDropdownOpenChange("Genre", open)}
              isCompact={isCompact}
            />
            <MultiSelectDropdown
              label="Type"
              options={facetOptions.types}
              selected={prefs.types}
              onChange={(val) => updatePrefs({ types: val })}
              single={true}
              onOpenStateChange={(open) => handleDropdownOpenChange("Type", open)}
              isCompact={isCompact}
            />
            <MultiSelectDropdown
              label="Theme"
              options={facetOptions.themes}
              selected={prefs.themes}
              onChange={(val) => updatePrefs({ themes: val })}
              onOpenStateChange={(open) => handleDropdownOpenChange("Theme", open)}
              isCompact={isCompact}
            />
            <MultiSelectDropdown
              label="Studio"
              options={facetOptions.studios}
              selected={prefs.studios}
              onChange={(val) => updatePrefs({ studios: val })}
              showSearch={true}
              onOpenStateChange={(open) => handleDropdownOpenChange("Studio", open)}
              isCompact={isCompact}
            />
          </div>

          <div className={`transition-all duration-300 ease-in-out motion-reduce:transition-none ${
            isCompact
              ? 'flex flex-row items-center gap-2 flex-nowrap flex-shrink-0 pt-0'
              : 'grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 w-full'
          }`}>
            <div className={`transition-all duration-300 ease-in-out motion-reduce:transition-none ${
              isCompact ? 'flex items-center gap-1.5 flex-shrink-0 w-auto' : 'w-full'
            }`}>
              <div className={`transition-all duration-300 ease-in-out motion-reduce:transition-none ${
                isCompact ? 'flex items-center gap-1 mb-0' : 'flex justify-between items-center mb-1.5'
              }`}>
                <span className={`font-semibold uppercase tracking-wider flex-shrink-0 transition-all duration-300 ease-in-out motion-reduce:transition-none ${
                  isCompact
                    ? prefs.minScore > 0 ? 'text-gray-900 font-semibold text-[10px] mr-0.5' : 'text-gray-400 text-[10px] mr-0.5'
                    : 'text-xs text-gray-500'
                }`}>
                  Score ≥
                </span>
                <span className={`font-mono font-semibold bg-white border rounded-none transition-all duration-300 ease-in-out motion-reduce:transition-none ${
                  isCompact ? 'px-1 py-0 text-[10px]' : 'px-2 py-0.5 text-xs'
                } ${
                  isCompact
                    ? prefs.minScore > 0 ? 'border-gray-900 text-gray-900 font-semibold' : 'border-gray-300 text-gray-500'
                    : 'border-gray-200 text-gray-900'
                }`}>
                  {prefs.minScore.toFixed(1)}
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
                  value={prefs.minScore}
                  onChange={(e) => updatePrefs({ minScore: parseFloat(e.target.value) })}
                  className="w-full h-1 bg-gray-200 appearance-none cursor-pointer accent-gray-900 focus:outline-none"
                />
              </div>
            </div>

            {isCompact && (
              <>
                <MultiSelectDropdown
                  label="Sort"
                  options={["Relevance", "MAL Score", "Popularity", "Release Date"]}
                  selected={[sortOptionsMap[prefs.sortBy]]}
                  onChange={handleSortLabelSelect}
                  single={true}
                  onOpenStateChange={(open) => handleDropdownOpenChange("Sort", open)}
                  isCompact={isCompact}
                  hideAllOption={true}
                />
                <button
                  type="button"
                  onClick={() => updatePrefs({ sortAsc: !prefs.sortAsc })}
                  className="border flex items-center justify-between focus:outline-none focus:ring-1 focus:ring-gray-900 cursor-pointer transition-all duration-300 ease-in-out px-2 py-1 text-xs border-gray-300 text-gray-500 hover:border-gray-400 bg-white whitespace-nowrap h-[26px] flex-shrink-0"
                >
                  <span className="text-[10px] uppercase tracking-wider mr-1.5 flex-shrink-0 text-gray-400 font-bold">
                    Order:
                  </span>
                  <span className="font-semibold text-gray-900">{getSortDirectionLabel(prefs.sortBy, prefs.sortAsc)}</span>
                </button>
              </>
            )}

            {!isCompact && (
              <>
                <div className="flex flex-col w-full">
                  <span className="block font-semibold text-gray-500 uppercase tracking-wider text-xs mb-1">
                    Sort
                  </span>
                  <div className="flex items-start gap-2">
                    <div className="flex-1 min-w-0">
                      <MultiSelectDropdown
                        label="Sort"
                        options={["Relevance", "MAL Score", "Popularity", "Release Date"]}
                        selected={[sortOptionsMap[prefs.sortBy]]}
                        onChange={handleSortLabelSelect}
                        single={true}
                        onOpenStateChange={(open) => handleDropdownOpenChange("Sort", open)}
                        isCompact={isCompact}
                        hideLabel={true}
                        hideAllOption={true}
                      />
                    </div>
                    <button
                      type="button"
                      onClick={() => updatePrefs({ sortAsc: !prefs.sortAsc })}
                      title={getSortDirectionLabel(prefs.sortBy, prefs.sortAsc)}
                      aria-label={getSortDirectionLabel(prefs.sortBy, prefs.sortAsc)}
                      className="h-[38px] w-[38px] flex-shrink-0 border border-gray-300 bg-white text-gray-500 hover:border-gray-400 hover:text-gray-900 focus:outline-none focus:ring-1 focus:ring-gray-900 flex items-center justify-center cursor-pointer transition-colors"
                    >
                      {prefs.sortAsc ? <ArrowUp className="w-4 h-4" /> : <ArrowDown className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <MultiSelectDropdown
                  label="Show Main"
                  options={["10", "20", "50", "100", "200"]}
                  selected={[String(prefs.mainK)]}
                  onChange={(val) => {
                    if (val.length > 0) updatePrefs({ mainK: Number(val[0]) });
                  }}
                  single={true}
                  onOpenStateChange={(open) => handleDropdownOpenChange("Show Main", open)}
                  isCompact={isCompact}
                />
                <MultiSelectDropdown
                  label="Show Cold"
                  options={["5", "10", "20", "50", "100"]}
                  selected={[String(prefs.coldK)]}
                  onChange={(val) => {
                    if (val.length > 0) updatePrefs({ coldK: Number(val[0]) });
                  }}
                  single={true}
                  onOpenStateChange={(open) => handleDropdownOpenChange("Show Cold", open)}
                  isCompact={isCompact}
                />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
