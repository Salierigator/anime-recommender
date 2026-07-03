import { useEffect, useRef, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ZoomIn, ZoomOut, Home } from 'lucide-react';
import type { MapResponse, AnimeItem } from '../types';
import { API_URL } from '../api';

interface MapExplorerProps {
  isOpen: boolean;
  onClose: () => void;
  mapData: MapResponse;
  mapXy: [number, number] | null;
  mainRecs: AnimeItem[];
  coldRecs: AnimeItem[];
  onSelectAnime: (malId: number) => void;
}

export function MapExplorer({
  isOpen,
  onClose,
  mapData,
  mapXy,
  mainRecs,
  coldRecs,
  onSelectAnime,
}: MapExplorerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Background image state
  const [bgImage, setBgImage] = useState<HTMLImageElement | null>(null);
  const [imageError, setImageError] = useState(false);

  // View state refs (mutable, accessed inside rAF loop for high performance)
  const [x0, x1, y0, y1] = mapData.meta.extent;
  const initialCx = (x0 + x1) / 2;
  const initialCy = (y0 + y1) / 2;

  const viewRef = useRef({
    cx: initialCx,
    cy: initialCy,
    zoom: 1.0,
  });

  // Animation ref for ease-fly to user
  const animRef = useRef({
    startTime: 0,
    duration: 1200,
    startCx: initialCx,
    startCy: initialCy,
    startZoom: 1.0,
    targetCx: mapXy ? mapXy[0] : initialCx,
    targetCy: mapXy ? mapXy[1] : initialCy,
    targetZoom: mapXy ? 3.0 : 1.0,
    active: false,
  });

  // Dragging and touch state refs
  const dragRef = useRef({
    isDragging: false,
    lastMouseX: 0,
    lastMouseY: 0,
    pinchDistance: 0,
    isPinching: false,
  });

  // Hovered point state (React state for tooltip rendering)
  const [hoveredPoint, setHoveredPoint] = useState<{
    malId: number;
    title: string;
    clusterName: string;
    isRec: boolean;
    isCold: boolean;
    rank?: number;
    screenX: number;
    screenY: number;
  } | null>(null);

  const { points, clusters, meta } = mapData;
  const k = meta.k || 28;
  const bg = meta.bg || '#0b1020';

  // Load background image
  useEffect(() => {
    if (!meta.territory_url) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = `${API_URL}${meta.territory_url}`;
    img.onload = () => setBgImage(img);
    img.onerror = () => setImageError(true);
  }, [meta.territory_url]);

  // Precomputed maps for fast lookup
  const pointsIndexMap = useMemo(() => {
    const m = new Map<number, number>();
    points.mal_id.forEach((id, idx) => m.set(id, idx));
    return m;
  }, [points.mal_id]);

  const recLookupMap = useMemo(() => {
    const m = new Map<number, { isCold: boolean; rank?: number }>();
    mainRecs.forEach((item, idx) => {
      m.set(item.mal_id, { isCold: false, rank: idx + 1 });
    });
    coldRecs.forEach((item) => {
      m.set(item.mal_id, { isCold: true });
    });
    return m;
  }, [mainRecs, coldRecs]);

  const clusterNameMap = useMemo(() => {
    const m = new Map<number, string>();
    clusters.forEach(c => m.set(c.label, c.name));
    return m;
  }, [clusters]);

  // Largest clusters first (for text LOD)
  const sortedClusters = useMemo(() => {
    return [...clusters].sort((a, b) => b.size - a.size);
  }, [clusters]);

  // Escape key handler to close
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Trigger ease-fly animation to user on open
  useEffect(() => {
    if (isOpen) {
      animRef.current = {
        startTime: Date.now(),
        duration: 1200,
        startCx: initialCx,
        startCy: initialCy,
        startZoom: 1.0,
        targetCx: mapXy ? mapXy[0] : initialCx,
        targetCy: mapXy ? mapXy[1] : initialCy,
        targetZoom: mapXy ? 3.0 : 1.0,
        active: mapXy !== null,
      };
      
      if (!mapXy) {
        viewRef.current = {
          cx: initialCx,
          cy: initialCy,
          zoom: 1.0,
        };
      }
    }
  }, [isOpen, mapXy, initialCx, initialCy]);

  // Main canvas render loop
  useEffect(() => {
    if (!isOpen) return;

    let animationId: number;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeAndRender = () => {
      const container = containerRef.current;
      if (!container) return;

      const rect = container.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;

      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      ctx.resetTransform();
      ctx.scale(dpr, dpr);

      const renderLoop = () => {
        const container = containerRef.current;
        if (!container) return;
        const rect = container.getBoundingClientRect();
        const w = rect.width;
        const h = rect.height;

        // 1. Process Fly Animation
        if (animRef.current.active) {
          const elapsed = Date.now() - animRef.current.startTime;
          const t = Math.min(elapsed / animRef.current.duration, 1);
          // Cubic ease-in-out
          const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

          viewRef.current.cx = animRef.current.startCx + (animRef.current.targetCx - animRef.current.startCx) * ease;
          viewRef.current.cy = animRef.current.startCy + (animRef.current.targetCy - animRef.current.startCy) * ease;
          viewRef.current.zoom = animRef.current.startZoom + (animRef.current.targetZoom - animRef.current.startZoom) * ease;

          if (t >= 1) {
            animRef.current.active = false;
          }
        }

        const { cx, cy, zoom } = viewRef.current;
        const baseScaleX = w / (x1 - x0);
        const baseScaleY = h / (y1 - y0);
        const scaleX = baseScaleX * zoom;
        const scaleY = baseScaleY * zoom;

        // Extent viewport calculations
        const visibleWidth = w / scaleX;
        const visibleHeight = h / scaleY;
        const visibleX0 = cx - visibleWidth / 2;
        const visibleX1 = cx + visibleWidth / 2;
        const visibleY0 = cy - visibleHeight / 2;
        const visibleY1 = cy + visibleHeight / 2;

        // Clear Canvas
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, w, h);

        // 2. Layer 1: Background Image (territory.png)
        if (bgImage && !imageError) {
          // Opacity fades out: 1.0 at zoom=1.0 down to 0.0 at zoom>=3.0
          const opacity = Math.max(0, Math.min(1.0, 1.0 - (zoom - 1.0) / 2.0));
          if (opacity > 0) {
            ctx.save();
            ctx.globalAlpha = opacity;
            const imgLeft = (x0 - visibleX0) * scaleX;
            const imgRight = (x1 - visibleX0) * scaleX;
            const imgTop = h - (y1 - visibleY0) * scaleY;
            const imgBottom = h - (y0 - visibleY0) * scaleY;
            ctx.drawImage(bgImage, imgLeft, imgTop, imgRight - imgLeft, imgBottom - imgTop);
            ctx.restore();
          }
        }

        // 3. Layer 2: Points (filtered by popularity zoom LOD)
        // Zoom LOD thresholds
        let popularityThreshold = 3000;
        if (zoom > 1.2) {
          const t = Math.min(1, (zoom - 1.2) / 2.8); // reach all points at zoom=4.0
          popularityThreshold = 3000 + t * 997000;
        }

        const pointRadius = Math.min(3.0, Math.max(0.7, 0.7 + (zoom - 1.0) * 0.25));

        const numPoints = points.x.length;
        for (let i = 0; i < numPoints; i++) {
          const px = points.x[i];
          const py = points.y[i];

          // Clip points outside the viewport
          if (px < visibleX0 || px > visibleX1 || py < visibleY0 || py > visibleY1) {
            continue;
          }

          const malId = points.mal_id[i];
          const isRec = recLookupMap.has(malId);
          const pop = points.popularity[i];

          // Skip drawing if popular/lod condition fails AND it is not a recommended highlight point
          if (pop >= popularityThreshold && !isRec) {
            continue;
          }

          // Render normal point (recs are rendered highlighted in next layer for visual prominence)
          if (!isRec) {
            const sx = (px - visibleX0) * scaleX;
            const sy = h - (py - visibleY0) * scaleY;

            const label = points.label[i];
            const hue = (label % k) / k * 360;

            ctx.fillStyle = `hsla(${hue}, 65%, 62%, 0.75)`;
            ctx.beginPath();
            ctx.arc(sx, sy, pointRadius, 0, 2 * Math.PI);
            ctx.fill();
          }
        }

        // 4. Layer 3: Highlight Recommendations
        recLookupMap.forEach((recInfo, recMalId) => {
          const pIdx = pointsIndexMap.get(recMalId);
          if (pIdx === undefined) return; // skip items not present on the SFW-only map

          const px = points.x[pIdx];
          const py = points.y[pIdx];

          // Clip points outside the viewport
          if (px < visibleX0 || px > visibleX1 || py < visibleY0 || py > visibleY1) {
            return;
          }

          const sx = (px - visibleX0) * scaleX;
          const sy = h - (py - visibleY0) * scaleY;

          const label = points.label[pIdx];
          const hue = (label % k) / k * 360;

          const drawRadius = Math.max(3.5, pointRadius + 2.0);

          ctx.fillStyle = `hsla(${hue}, 85%, 65%, 0.95)`;
          ctx.beginPath();
          ctx.arc(sx, sy, drawRadius, 0, 2 * Math.PI);
          ctx.fill();

          ctx.strokeStyle = recInfo.isCold ? '#00ffff' : '#ffffff';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        });

        // 5. Layer 4: Cluster Labels
        // Zoom LOD for clusters: Zoom far -> top 10 size; Zoom close -> all 28.
        const clusterCount = zoom < 1.8 ? 10 : k;
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 11px Inter, system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        for (let i = 0; i < Math.min(clusterCount, sortedClusters.length); i++) {
          const c = sortedClusters[i];
          
          // Clip label outside viewport
          if (c.cx < visibleX0 || c.cx > visibleX1 || c.cy < visibleY0 || c.cy > visibleY1) {
            continue;
          }

          const sx = (c.cx - visibleX0) * scaleX;
          const sy = h - (c.cy - visibleY0) * scaleY;

          // Shadow/halo text
          ctx.strokeStyle = '#020617';
          ctx.lineWidth = 3.5;
          ctx.strokeText(c.name, sx, sy);
          ctx.fillText(c.name, sx, sy);
        }

        // 6. Layer 5: User Marker at map_xy
        if (mapXy !== null) {
          const mx = mapXy[0];
          const my = mapXy[1];

          if (mx >= visibleX0 && mx <= visibleX1 && my >= visibleY0 && my <= visibleY1) {
            const sx = (mx - visibleX0) * scaleX;
            const sy = h - (my - visibleY0) * scaleY;

            const now = Date.now();
            const pulse = (now % 1500) / 1500; // 0 to 1

            // Outer pulsing ring
            const outerRadius = 7 + pulse * 18;
            const outerOpacity = 0.8 * (1 - pulse);
            ctx.strokeStyle = `rgba(255, 255, 255, ${outerOpacity})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(sx, sy, outerRadius, 0, 2 * Math.PI);
            ctx.stroke();

            // Inner solid dot
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.arc(sx, sy, 4.5, 0, 2 * Math.PI);
            ctx.fill();

            // Marker label
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 10px Inter, system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';

            ctx.strokeStyle = '#020617';
            ctx.lineWidth = 2.5;
            ctx.strokeText('You are here', sx, sy + 8);
            ctx.fillText('You are here', sx, sy + 8);
          }
        }

        animationId = requestAnimationFrame(renderLoop);
      };

      renderLoop();
    };

    resizeAndRender();

    const resizeObserver = new ResizeObserver(() => {
      if (animationId) cancelAnimationFrame(animationId);
      resizeAndRender();
    });

    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      if (animationId) cancelAnimationFrame(animationId);
      resizeObserver.disconnect();
    };
  }, [isOpen, bgImage, imageError, pointsIndexMap, recLookupMap, sortedClusters, mapXy, x0, x1, y0, y1, k, bg, points]);

  // Zoom helpers
  const handleZoom = (direction: 'in' | 'out') => {
    animRef.current.active = false; // interrupt animation

    const zoomFactor = direction === 'in' ? 1.4 : 1 / 1.4;
    const currentZoom = viewRef.current.zoom;
    const nextZoom = Math.min(40.0, Math.max(0.8, currentZoom * zoomFactor));

    viewRef.current.zoom = nextZoom;
  };

  const handleReset = () => {
    animRef.current = {
      startTime: Date.now(),
      duration: 800,
      startCx: viewRef.current.cx,
      startCy: viewRef.current.cy,
      startZoom: viewRef.current.zoom,
      targetCx: initialCx,
      targetCy: initialCy,
      targetZoom: 1.0,
      active: true,
    };
  };

  // Mouse pan handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return; // Left click only
    animRef.current.active = false; // stop any active transitions

    dragRef.current.isDragging = true;
    dragRef.current.lastMouseX = e.clientX;
    dragRef.current.lastMouseY = e.clientY;
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // 1. Pan if dragging
    if (dragRef.current.isDragging) {
      const dx = e.clientX - dragRef.current.lastMouseX;
      const dy = e.clientY - dragRef.current.lastMouseY;

      dragRef.current.lastMouseX = e.clientX;
      dragRef.current.lastMouseY = e.clientY;

      const w = rect.width;
      const h = rect.height;
      const scaleX = (w / (x1 - x0)) * viewRef.current.zoom;
      const scaleY = (h / (y1 - y0)) * viewRef.current.zoom;

      viewRef.current.cx -= dx / scaleX;
      viewRef.current.cy += dy / scaleY; // screen y is inverted compared to world y
      
      setHoveredPoint(null);
      return;
    }

    // 2. Update Tooltip / find nearest point
    const w = rect.width;
    const h = rect.height;
    const { cx, cy, zoom } = viewRef.current;
    const scaleX = (w / (x1 - x0)) * zoom;
    const scaleY = (h / (y1 - y0)) * zoom;

    const visibleWidth = w / scaleX;
    const visibleHeight = h / scaleY;
    const visibleX0 = cx - visibleWidth / 2;
    const visibleX1 = cx + visibleWidth / 2;
    const visibleY0 = cy - visibleHeight / 2;
    const visibleY1 = cy + visibleHeight / 2;

    let popularityThreshold = 3000;
    if (zoom > 1.2) {
      const t = Math.min(1, (zoom - 1.2) / 2.8);
      popularityThreshold = 3000 + t * 997000;
    }

    let closestIdx = -1;
    let minDistance = 8.0; // 8px radius

    const numPoints = points.x.length;
    for (let i = 0; i < numPoints; i++) {
      const px = points.x[i];
      const py = points.y[i];

      // Clip bounds check
      if (px < visibleX0 || px > visibleX1 || py < visibleY0 || py > visibleY1) {
        continue;
      }

      const malId = points.mal_id[i];
      const isRec = recLookupMap.has(malId);
      const pop = points.popularity[i];

      if (pop >= popularityThreshold && !isRec) {
        continue;
      }

      const sx = (px - visibleX0) * scaleX;
      const sy = h - (py - visibleY0) * scaleY;

      const d = Math.hypot(sx - mouseX, sy - mouseY);
      if (d < minDistance) {
        minDistance = d;
        closestIdx = i;
      }
    }

    if (closestIdx !== -1) {
      const malId = points.mal_id[closestIdx];
      const title = points.title[closestIdx];
      const clusterLabel = points.label[closestIdx];
      const clusterName = clusterNameMap.get(clusterLabel) || 'Unknown';
      const rec = recLookupMap.get(malId);

      setHoveredPoint({
        malId,
        title,
        clusterName,
        isRec: !!rec,
        isCold: rec?.isCold ?? false,
        rank: rec?.rank,
        screenX: mouseX,
        screenY: mouseY,
      });
    } else {
      setHoveredPoint(null);
    }
  };

  const handleMouseUp = () => {
    dragRef.current.isDragging = false;
  };

  const handleMouseLeave = () => {
    dragRef.current.isDragging = false;
    setHoveredPoint(null);
  };

  // Zoom to cursor handler
  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    animRef.current.active = false;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const w = rect.width;
    const h = rect.height;
    const { cx, cy, zoom } = viewRef.current;

    const baseScaleX = w / (x1 - x0);
    const baseScaleY = h / (y1 - y0);
    let scaleX = baseScaleX * zoom;
    let scaleY = baseScaleY * zoom;

    // Calculate mouse position in world space before zoom
    const wx = (mouseX - w / 2) / scaleX + cx;
    const wy = -(mouseY - h / 2) / scaleY + cy;

    // Zoom multiplier
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const nextZoom = Math.min(40.0, Math.max(0.8, zoom * factor));

    // Recompute scale with nextZoom
    scaleX = baseScaleX * nextZoom;
    scaleY = baseScaleY * nextZoom;

    // Shift center to keep mouse target constant
    viewRef.current.zoom = nextZoom;
    viewRef.current.cx = wx - (mouseX - w / 2) / scaleX;
    viewRef.current.cy = wy + (mouseY - h / 2) / scaleY;

    // Force recalculating hover item under updated coordinates
    handleMouseMove(e as any);
  };

  // Click to open recommendation modal
  const handleCanvasClick = () => {
    if (hoveredPoint && hoveredPoint.isRec) {
      onSelectAnime(hoveredPoint.malId);
    }
  };

  // Touch handlers for panning & pinching
  const handleTouchStart = (e: React.TouchEvent<HTMLCanvasElement>) => {
    animRef.current.active = false;
    if (e.touches.length === 1) {
      dragRef.current.isDragging = true;
      dragRef.current.lastMouseX = e.touches[0].clientX;
      dragRef.current.lastMouseY = e.touches[0].clientY;
      dragRef.current.isPinching = false;
    } else if (e.touches.length === 2) {
      dragRef.current.isDragging = false;
      dragRef.current.isPinching = true;
      const d = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      dragRef.current.pinchDistance = d;
      dragRef.current.lastMouseX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
      dragRef.current.lastMouseY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
    }
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    const baseScaleX = w / (x1 - x0);
    const baseScaleY = h / (y1 - y0);

    if (dragRef.current.isDragging && e.touches.length === 1) {
      const dx = e.touches[0].clientX - dragRef.current.lastMouseX;
      const dy = e.touches[0].clientY - dragRef.current.lastMouseY;

      dragRef.current.lastMouseX = e.touches[0].clientX;
      dragRef.current.lastMouseY = e.touches[0].clientY;

      const scaleX = baseScaleX * viewRef.current.zoom;
      const scaleY = baseScaleY * viewRef.current.zoom;

      viewRef.current.cx -= dx / scaleX;
      viewRef.current.cy += dy / scaleY;
    } else if (dragRef.current.isPinching && e.touches.length === 2) {
      const d = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      const factor = d / dragRef.current.pinchDistance;
      dragRef.current.pinchDistance = d;

      // Pinch center in client screen
      const clientCenterX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
      const clientCenterY = (e.touches[0].clientY + e.touches[1].clientY) / 2;

      const canvasCenterX = clientCenterX - rect.left;
      const canvasCenterY = clientCenterY - rect.top;

      const currentZoom = viewRef.current.zoom;
      const nextZoom = Math.min(40.0, Math.max(0.8, currentZoom * factor));

      let scaleX = baseScaleX * currentZoom;
      let scaleY = baseScaleY * currentZoom;
      const wx = (canvasCenterX - w / 2) / scaleX + viewRef.current.cx;
      const wy = -(canvasCenterY - h / 2) / scaleY + viewRef.current.cy;

      scaleX = baseScaleX * nextZoom;
      scaleY = baseScaleY * nextZoom;

      viewRef.current.zoom = nextZoom;
      viewRef.current.cx = wx - (canvasCenterX - w / 2) / scaleX;
      viewRef.current.cy = wy + (canvasCenterY - h / 2) / scaleY;
    }
  };

  const handleTouchEnd = () => {
    dragRef.current.isDragging = false;
    dragRef.current.isPinching = false;
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          layoutId="taste-map-container"
          className="fixed inset-0 z-50 overflow-hidden flex flex-col justify-end"
          style={{ backgroundColor: bg }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          {/* Main Map canvas */}
          <div ref={containerRef} className="absolute inset-0 w-full h-full">
            <canvas
              ref={canvasRef}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseLeave}
              onWheel={handleWheel}
              onClick={handleCanvasClick}
              onTouchStart={handleTouchStart}
              onTouchMove={handleTouchMove}
              onTouchEnd={handleTouchEnd}
              className="absolute inset-0 block cursor-grab active:cursor-grabbing select-none"
            />
          </div>

          {/* Close Button overlay */}
          <button
            onClick={onClose}
            className="absolute top-6 right-6 z-50 p-2.5 bg-black/60 hover:bg-black/80 text-white/90 border border-white/10 rounded-full shadow-lg transition-colors cursor-pointer"
            aria-label="Close Map Explorer"
          >
            <X className="w-5 h-5" />
          </button>

          {/* Controls HUD */}
          <div className="absolute top-6 left-6 z-50 flex gap-2">
            <button
              onClick={() => handleZoom('in')}
              className="p-2.5 bg-black/60 hover:bg-black/80 text-white border border-white/10 rounded-full shadow-lg transition-colors cursor-pointer"
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={() => handleZoom('out')}
              className="p-2.5 bg-black/60 hover:bg-black/80 text-white border border-white/10 rounded-full shadow-lg transition-colors cursor-pointer"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <button
              onClick={handleReset}
              className="p-2.5 bg-black/60 hover:bg-black/80 text-white border border-white/10 rounded-full shadow-lg transition-colors cursor-pointer"
              title="Reset View"
            >
              <Home className="w-4 h-4" />
            </button>
          </div>

          {/* Legend Overlay at Bottom-Left */}
          <div className="absolute bottom-6 left-6 z-50 pointer-events-none select-none">
            <div className="flex items-center gap-3 bg-[#0b1020]/95 px-4 py-2 border border-white/10 rounded-full text-[10px] uppercase font-bold tracking-wider text-white/70 shadow-2xl backdrop-blur">
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-2.5 h-2.5 rounded-full bg-indigo-400" />
                Your taste map
              </span>
              <span className="text-white/20">·</span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-2.5 h-2.5 rounded-full border border-white bg-indigo-500" />
                Recommended
              </span>
              <span className="text-white/20">·</span>
              <span className="flex items-center gap-1.5">
                <span className="text-[10px] text-white">✦</span>
                You
              </span>
            </div>
          </div>

          {/* Hover Tooltip Overlay (Absolute DOM elements on top of Canvas) */}
          {hoveredPoint && (
            <div
              className="absolute z-40 pointer-events-none select-none bg-slate-950/90 backdrop-blur-md border border-white/10 text-white rounded-lg px-3 py-2.5 shadow-2xl flex flex-col gap-1 min-w-[200px]"
              style={{
                left: `${hoveredPoint.screenX + 16}px`,
                top: `${hoveredPoint.screenY + 16}px`,
              }}
            >
              <span className="font-bold text-sm leading-tight text-white/95">{hoveredPoint.title}</span>
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wide">
                Cluster: {hoveredPoint.clusterName}
              </span>
              
              {hoveredPoint.isRec && (
                <div className="mt-1 flex items-center gap-1.5">
                  <span className={`px-1.5 py-0.5 rounded text-[9px] uppercase font-extrabold tracking-wider ${
                    hoveredPoint.isCold 
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' 
                      : 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                  }`}>
                    {hoveredPoint.isCold ? 'New & Trending (Cold)' : `Recommended #${hoveredPoint.rank}`}
                  </span>
                  <span className="text-[9px] text-slate-400 font-medium">Click to view details</span>
                </div>
              )}
            </div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
