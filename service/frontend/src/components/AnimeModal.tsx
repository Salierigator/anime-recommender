import { useEffect, useState } from 'react';
import { X, ExternalLink, AlertCircle } from 'lucide-react';
import { fetchAnimeDetail } from '../utils/jikanQueue';

interface AnimeModalProps {
  malId: number | null;
  isOpen: boolean;
  onClose: () => void;
}

export function AnimeModal({ malId, isOpen, onClose }: AnimeModalProps) {
  const [data, setData] = useState<any | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<boolean>(false);

  useEffect(() => {
    if (!isOpen || !malId) {
      setData(null);
      setError(false);
      return;
    }

    let isMounted = true;
    setIsLoading(true);
    setError(false);

    fetchAnimeDetail(malId)
      .then((res) => {
        if (isMounted) {
          if (res) {
            setData(res);
          } else {
            setError(true);
          }
          setIsLoading(false);
        }
      })
      .catch(() => {
        if (isMounted) {
          setError(true);
          setIsLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [malId, isOpen]);

  // Handle escape key
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      window.addEventListener('keydown', handleEsc);
    }
    return () => window.removeEventListener('keydown', handleEsc);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const imageUrl = data?.images?.jpg?.large_image_url || data?.images?.jpg?.image_url;
  const year = data?.year || data?.aired?.prop?.from?.year || data?.aired?.string?.split(' to ')[0]?.trim();

  // Combine tags
  const tags = [
    ...(data?.genres || []),
    ...(data?.themes || []),
    ...(data?.demographics || [])
  ].map((t: any) => t.name);

  const studios = data?.studios?.map((s: any) => s.name).join(', ');

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm transition-opacity"
      onClick={handleOverlayClick}
    >
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-white rounded-xl shadow-2xl overflow-hidden flex flex-col md:flex-row animate-in fade-in zoom-in-95 duration-200">

        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-10 p-2 bg-white/80 hover:bg-white text-gray-700 rounded-full shadow-sm transition-colors"
          aria-label="Close modal"
        >
          <X className="w-5 h-5" />
        </button>

        {isLoading && (
          <div className="w-full h-96 flex flex-col items-center justify-center text-gray-500">
            <div className="h-8 w-8 rounded-full border-4 border-gray-200 border-t-gray-600 animate-spin mb-4"></div>
            <p className="font-medium">Loading details...</p>
          </div>
        )}

        {error && !isLoading && (
          <div className="w-full h-96 flex flex-col items-center justify-center text-gray-500 space-y-4">
            <AlertCircle className="w-12 h-12 text-red-400" />
            <p className="font-medium text-lg">Failed to load details</p>
            <button
              onClick={() => {
                setIsLoading(true);
                setError(false);
                fetchAnimeDetail(malId!).then(res => {
                  if (res) setData(res);
                  else setError(true);
                  setIsLoading(false);
                });
              }}
              className="px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors text-sm"
            >
              Retry
            </button>
          </div>
        )}

        {!isLoading && !error && data && (
          <>
            {/* Left: Poster */}
            <div className="md:w-1/3 bg-gray-100 flex-shrink-0 hidden md:flex items-center justify-center overflow-hidden relative">
              {imageUrl ? (
                <>
                  <img
                    src={imageUrl}
                    alt=""
                    aria-hidden="true"
                    className="absolute inset-0 w-full h-full object-cover blur-2xl scale-110 opacity-30"
                  />
                  <div className="absolute inset-0 bg-white/20"></div>
                  <img
                    src={imageUrl}
                    alt={data.title}
                    className="relative w-full aspect-[2/3] object-cover shadow-lg"
                  />
                </>
              ) : (
                <div className="w-full aspect-[2/3] flex items-center justify-center text-gray-400 relative">
                  No Image
                </div>
              )}
            </div>

            {/* Right: Content */}
            <div className="flex-1 flex flex-col min-h-0 min-w-0 overflow-hidden">
              {/* Header */}
              <div className="p-6 pb-4 border-b border-gray-100 flex-shrink-0">
                <div className="pr-10">
                  <h2 className="text-2xl md:text-3xl font-bold text-gray-900 leading-tight">
                    {data.title}
                  </h2>
                  {data.title_english && data.title_english !== data.title && (
                    <p className="text-gray-500 mt-1 text-sm">{data.title_english}</p>
                  )}
                </div>

                {/* Stats Row */}
                <div className="flex flex-wrap items-center gap-4 mt-4 text-sm">
                  {data.score && (
                    <div className="flex flex-col">
                      <span className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">MAL Score</span>
                      <span className="font-bold text-gray-900">{data.score}</span>
                    </div>
                  )}
                  {data.rank && (
                    <div className="flex flex-col">
                      <span className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">Rank</span>
                      <span className="font-bold text-gray-900">#{data.rank}</span>
                    </div>
                  )}
                  {data.popularity && (
                    <div className="flex flex-col">
                      <span className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold">Popularity</span>
                      <span className="font-bold text-gray-900">#{data.popularity}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Scrollable Body */}
              <div className="p-6 overflow-y-auto flex-1 min-h-0">
                {/* Meta Info */}
                <div className="flex flex-wrap items-center gap-y-2 gap-x-4 text-sm text-gray-700 mb-6 bg-gray-50 p-3 rounded-lg border border-gray-100">
                  {data.type && <div><span className="font-medium text-gray-500 mr-1">Type:</span> {data.type}</div>}
                  {year && <div><span className="font-medium text-gray-500 mr-1">Year:</span> {year}</div>}
                  {data.episodes && <div><span className="font-medium text-gray-500 mr-1">Episodes:</span> {data.episodes}</div>}
                  {data.status && <div><span className="font-medium text-gray-500 mr-1">Status:</span> {data.status}</div>}
                  {studios && <div className="w-full mt-1"><span className="font-medium text-gray-500 mr-1">Studio:</span> {studios}</div>}
                </div>

                {/* Tags */}
                {tags.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-6">
                    {tags.map((tag, idx) => (
                      <span key={idx} className="px-2.5 py-1 bg-gray-100 text-gray-700 text-xs rounded-full border border-gray-200">
                        {tag}
                      </span>
                    ))}
                  </div>
                )}

                {/* Synopsis */}
                {data.synopsis && (
                  <div>
                    <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider mb-2">Synopsis</h3>
                    <p className="text-gray-600 text-sm leading-relaxed whitespace-pre-line">
                      {data.synopsis}
                    </p>
                  </div>
                )}
              </div>

              {/* Footer */}
              <div className="p-4 border-t border-gray-100 bg-gray-50 flex justify-end flex-shrink-0">
                <a
                  href={`https://myanimelist.net/anime/${malId}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-4 py-2 bg-gray-900 text-white text-sm font-medium rounded-lg hover:bg-gray-800 transition-colors shadow-sm"
                >
                  View on MAL
                  <ExternalLink className="w-4 h-4" />
                </a>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
