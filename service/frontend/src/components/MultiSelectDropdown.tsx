/* eslint-disable react-hooks/set-state-in-effect */
import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Search, X, Check } from 'lucide-react';

interface MultiSelectProps {
  label: string;
  options: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
  showSearch?: boolean;
  single?: boolean;
  onOpenStateChange?: (isOpen: boolean) => void;
  isCompact?: boolean;
  hideLabel?: boolean;
  hideAllOption?: boolean;
}

export function MultiSelectDropdown({
  label,
  options,
  selected,
  onChange,
  showSearch = false,
  single = false,
  onOpenStateChange,
  isCompact = false,
  hideLabel = false,
  hideAllOption = false
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

  const showAllOption = single && !label.startsWith('Show') && !hideAllOption;
  const isActive = selected.length > 0;

  return (
    <div
      className={`relative flex-1 transition-all duration-300 ease-in-out motion-reduce:transition-none ${
        isCompact ? 'min-w-[100px] max-w-[200px] flex-shrink-0' : 'min-w-0'
      }`}
      ref={dropdownRef}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {!hideLabel && (
        <span
          className={`block font-semibold text-gray-500 uppercase tracking-wider transition-all duration-300 ease-in-out motion-reduce:transition-none ${
            isCompact
              ? 'h-0 overflow-hidden opacity-0 mb-0 pointer-events-none text-[0px]'
              : 'text-xs mb-1'
          }`}
        >
          {label}
        </span>
      )}
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
