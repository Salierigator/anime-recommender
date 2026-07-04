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
  isDetailOpen?: boolean;
}

export function MapExplorer({
  isOpen,
  onClose,
  mapData,
  mapXy,
  mainRecs,
  coldRecs: _coldRecs,
  onSelectAnime,
  isDetailOpen = false,
}: MapExplorerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Background image state
  const [bgImage, setBgImage] = useState<HTMLImageElement | null>(null);
  const [imageError, setImageError] = useState(false);

  // CSS dimensions of container
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  // View state refs (mutable, accessed inside rAF loop for high performance)
  const [x0, x1, y0, y1] = mapData.meta.extent;
  const initialCx = (x0 + x1) / 2;
  const initialCy = (y0 + y1) / 2;

  const [showRecommended, setShowRecommended] = useState(true);
  const [highlightTopN, setHighlightTopN] = useState(50);

  const viewRef = useRef({
    cx: initialCx,
    cy: initialCy,
    scale: 1.0,
  });

  const fitScaleRef = useRef(1.0);
  const isInitialFitDone = useRef(false);

  // Animation ref for ease-fly to user
  const animRef = useRef({
    startTime: 0,
    duration: 1200,
    startCx: initialCx,
    startCy: initialCy,
    startScale: 1.0,
    targetCx: mapXy ? mapXy[0] : initialCx,
    targetCy: mapXy ? mapXy[1] : initialCy,
    targetScale: 1.0,
    active: false,
  });

  const setView = ({ cx, cy, scale }: { cx: number; cy: number; scale: number }) => {
    const fitScale = fitScaleRef.current;
    const clampedScale = Math.max(fitScale, Math.min(fitScale * 40, scale));
    const clampedCx = Math.max(x0, Math.min(x1, cx));
    const clampedCy = Math.max(y0, Math.min(y1, cy));
    viewRef.current = {
      cx: clampedCx,
      cy: clampedCy,
      scale: clampedScale,
    };
  };

  // Dragging, pinching and backdrop click tracking
  const dragRef = useRef({
    isDragging: false,
    lastMouseX: 0,
    lastMouseY: 0,
    pinchDistance: 0,
    isPinching: false,
  });

  const backdropPointerDownRef = useRef(false);

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
    if (!showRecommended) return m;

    mainRecs.slice(0, highlightTopN).forEach((item, idx) => {
      m.set(item.mal_id, { isCold: false, rank: idx + 1 });
    });
    // According to requirements: "chỉ warm/main, KHÔNG cần cold" in highlight,
    // so we do not include cold recommendations in the map highlights.
    return m;
  }, [mainRecs, highlightTopN, showRecommended]);

  const clusterNameMap = useMemo(() => {
    const m = new Map<number, string>();
    clusters.forEach(c => m.set(c.label, c.name));
    return m;
  }, [clusters]);

  // Compute radiusWorld for each cluster (percentile 80 distance to center)
  const clustersWithRadius = useMemo(() => {
    const clusterPoints = new Map<number, { x: number; y: number }[]>();
    const n = points.x.length;
    for (let i = 0; i < n; i++) {
      const lbl = points.label[i];
      const pt = { x: points.x[i], y: points.y[i] };
      let pts = clusterPoints.get(lbl);
      if (!pts) {
        pts = [];
        clusterPoints.set(lbl, pts);
      }
      pts.push(pt);
    }

    return clusters.map((cluster) => {
      const pts = clusterPoints.get(cluster.label) || [];
      if (pts.length === 0) {
        return { ...cluster, radiusWorld: 0 };
      }
      const distances = pts.map(pt => Math.hypot(pt.x - cluster.cx, pt.y - cluster.cy));
      distances.sort((a, b) => a - b);
      const percentileIndex = Math.min(distances.length - 1, Math.floor(distances.length * 0.8));
      const radiusWorld = distances[percentileIndex];
      return {
        ...cluster,
        radiusWorld,
      };
    });
  }, [clusters, points]);

  // Escape key handler to close (blocks closing when detail view is open)
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (isDetailOpen) return;
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, isDetailOpen]);

  // ResizeObserver to track container sizes
  useEffect(() => {
    if (!isOpen) return;
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      if (!entries || entries.length === 0) return;
      const { width, height } = entries[0].contentRect;
      if (width > 0 && height > 0) {
        const oldFitScale = fitScaleRef.current;
        const extentW = x1 - x0;
        const extentH = y1 - y0;
        const fitScale = Math.min(width / extentW, height / extentH) * 0.9;
        fitScaleRef.current = fitScale;
        setDimensions({ width, height });

        // Perform initial centering and ease-fly setup on first layout
        if (!isInitialFitDone.current) {
          setView({
            cx: initialCx,
            cy: initialCy,
            scale: fitScale,
          });

          if (mapXy !== null) {
            animRef.current = {
              startTime: Date.now(),
              duration: 1200,
              startCx: initialCx,
              startCy: initialCy,
              startScale: fitScale,
              targetCx: mapXy[0],
              targetCy: mapXy[1],
              targetScale: fitScale * 3.0,
              active: true,
            };
          }
          isInitialFitDone.current = true;
        } else {
          // Adjust current scale to maintain same zoom level relative to fitScale
          const currentZoom = viewRef.current.scale / oldFitScale;
          setView({
            cx: viewRef.current.cx,
            cy: viewRef.current.cy,
            scale: fitScale * currentZoom,
          });
        }
      }
    });

    observer.observe(container);
    return () => {
      observer.disconnect();
      isInitialFitDone.current = false;
    };
  }, [isOpen, initialCx, initialCy, mapXy, x0, x1, y0, y1]);

  // Resize and scale canvas context matching devicePixelRatio
  useEffect(() => {
    if (!isOpen) return;
    const canvas = canvasRef.current;
    const { width, height } = dimensions;
    if (!canvas || width === 0 || height === 0) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
  }, [isOpen, dimensions]);

  // Render Loop
  useEffect(() => {
    if (!isOpen) return;
    const canvas = canvasRef.current;
    const { width, height } = dimensions;
    if (!canvas || width === 0 || height === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;

    const renderLoop = () => {
      // 1. Process Fly Animation
      if (animRef.current.active) {
        const elapsed = Date.now() - animRef.current.startTime;
        const t = Math.min(elapsed / animRef.current.duration, 1);
        const ease = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

        const nextCx = animRef.current.startCx + (animRef.current.targetCx - animRef.current.startCx) * ease;
        const nextCy = animRef.current.startCy + (animRef.current.targetCy - animRef.current.startCy) * ease;
        const nextScale = animRef.current.startScale + (animRef.current.targetScale - animRef.current.startScale) * ease;

        setView({ cx: nextCx, cy: nextCy, scale: nextScale });

        if (t >= 1) {
          animRef.current.active = false;
        }
      }

      const { cx, cy, scale } = viewRef.current;
      const fitScale = fitScaleRef.current;
      const zoom = scale / fitScale;

      // Viewport bounds in world coords
      const visibleWidth = width / scale;
      const visibleHeight = height / scale;
      const visibleX0 = cx - visibleWidth / 2;
      const visibleX1 = cx + visibleWidth / 2;
      const visibleY0 = cy - visibleHeight / 2;
      const visibleY1 = cy + visibleHeight / 2;

      // Clear Canvas
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);

      // Layer 1: Background Image (territory.png)
      if (bgImage && !imageError) {
        const opacity = Math.max(0, Math.min(1.0, 1.0 - (zoom - 1.0) / 2.0));
        if (opacity > 0) {
          ctx.save();
          ctx.globalAlpha = opacity;
          const imgLeft = (x0 - cx) * scale + width / 2;
          const imgRight = (x1 - cx) * scale + width / 2;
          const imgTop = -(y1 - cy) * scale + height / 2;
          const imgBottom = -(y0 - cy) * scale + height / 2;
          ctx.drawImage(bgImage, imgLeft, imgTop, imgRight - imgLeft, imgBottom - imgTop);
          ctx.restore();
        }
      }

      // Layer 2: Points
      let popularityThreshold = 3000;
      if (zoom > 1.2) {
        const t = Math.min(1, (zoom - 1.2) / 2.8);
        popularityThreshold = 3000 + t * 997000;
      }

      const pointRadius = Math.min(3.0, Math.max(0.7, 0.7 + (zoom - 1.0) * 0.25));

      const numPoints = points.x.length;
      for (let i = 0; i < numPoints; i++) {
        const px = points.x[i];
        const py = points.y[i];

        if (px < visibleX0 || px > visibleX1 || py < visibleY0 || py > visibleY1) {
          continue;
        }

        const malId = points.mal_id[i];
        const isRec = recLookupMap.has(malId);
        const pop = points.popularity[i];

        if (pop >= popularityThreshold && !isRec) {
          continue;
        }

        if (!isRec) {
          const sx = (px - cx) * scale + width / 2;
          const sy = -(py - cy) * scale + height / 2;

          const label = points.label[i];
          const hue = (label % k) / k * 360;

          ctx.fillStyle = `hsla(${hue}, 65%, 62%, 0.75)`;
          ctx.beginPath();
          ctx.arc(sx, sy, pointRadius, 0, 2 * Math.PI);
          ctx.fill();
        }
      }

      // Layer 3: Highlight Recommendations
      recLookupMap.forEach((recInfo, recMalId) => {
        const pIdx = pointsIndexMap.get(recMalId);
        if (pIdx === undefined) return;

        const px = points.x[pIdx];
        const py = points.y[pIdx];

        if (px < visibleX0 || px > visibleX1 || py < visibleY0 || py > visibleY1) {
          return;
        }

        const sx = (px - cx) * scale + width / 2;
        const sy = -(py - cy) * scale + height / 2;

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

      // Layer 4: Cluster Labels
      const marginW = visibleWidth * 0.1;
      const marginH = visibleHeight * 0.1;

      const activeClusters = clustersWithRadius
        .map(c => {
          const pixelRadius = c.radiusWorld * scale;
          const isInside = 
            c.cx >= cx - visibleWidth / 2 - marginW &&
            c.cx <= cx + visibleWidth / 2 + marginW &&
            c.cy >= cy - visibleHeight / 2 - marginH &&
            c.cy <= cy + visibleHeight / 2 + marginH;

          if (pixelRadius > 60 && isInside) {
            let alpha = 1.0;
            if (pixelRadius < 90) {
              alpha = (pixelRadius - 60) / 30;
            }
            return { c, alpha };
          }
          return null;
        })
        .filter((item): item is { c: typeof clustersWithRadius[0]; alpha: number } => item !== null);

      let renderedClusters = activeClusters;
      if (activeClusters.length > 12) {
        activeClusters.sort((a, b) => b.c.size - a.c.size);
        renderedClusters = activeClusters.slice(0, 12);
      }

      ctx.font = 'bold 11px Inter, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      renderedClusters.forEach(({ c, alpha }) => {
        const sx = (c.cx - cx) * scale + width / 2;
        const sy = -(c.cy - cy) * scale + height / 2;

        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = bg;
        ctx.lineWidth = 3.5;
        ctx.strokeText(c.name, sx, sy);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(c.name, sx, sy);
        ctx.restore();
      });

      // Layer 5: User Marker at map_xy
      if (mapXy !== null) {
        const mx = mapXy[0];
        const my = mapXy[1];

        if (mx >= visibleX0 && mx <= visibleX1 && my >= visibleY0 && my <= visibleY1) {
          const sx = (mx - cx) * scale + width / 2;
          const sy = -(my - cy) * scale + height / 2;

          const now = Date.now();
          const pulse = (now % 1500) / 1500;

          const outerRadius = 7 + pulse * 18;
          const outerOpacity = 0.8 * (1 - pulse);
          ctx.strokeStyle = `rgba(255, 255, 255, ${outerOpacity})`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(sx, sy, outerRadius, 0, 2 * Math.PI);
          ctx.stroke();

          ctx.fillStyle = '#ffffff';
          ctx.beginPath();
          ctx.arc(sx, sy, 4.5, 0, 2 * Math.PI);
          ctx.fill();

          ctx.fillStyle = '#ffffff';
          ctx.font = 'bold 10px Inter, system-ui, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';

          ctx.strokeStyle = bg;
          ctx.lineWidth = 2.5;
          ctx.strokeText('You are here', sx, sy + 8);
          ctx.fillText('You are here', sx, sy + 8);
        }
      }

      animationId = requestAnimationFrame(renderLoop);
    };

    renderLoop();
    return () => cancelAnimationFrame(animationId);
  }, [isOpen, dimensions, bgImage, imageError, pointsIndexMap, recLookupMap, clustersWithRadius, mapXy, x0, x1, y0, y1, k, bg, points]);

  // Bind custom wheel listener to support trackpad pinching and 2-finger panning
  useEffect(() => {
    if (!isOpen) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      animRef.current.active = false;

      const { width, height } = dimensions;
      if (width === 0 || height === 0) return;

      const { cx, cy, scale } = viewRef.current;

      if (e.ctrlKey) {
        // Pinch Zoom
        const factor = Math.exp(-e.deltaY * 0.01);
        const nextScale = scale * factor;

        const wx = (e.clientX - canvas.getBoundingClientRect().left - width / 2) / scale + cx;
        const wy = -(e.clientY - canvas.getBoundingClientRect().top - height / 2) / scale + cy;

        setView({
          cx: wx - (e.clientX - canvas.getBoundingClientRect().left - width / 2) / nextScale,
          cy: wy + (e.clientY - canvas.getBoundingClientRect().top - height / 2) / nextScale,
          scale: nextScale,
        });
      } else {
        // Two-finger Pan
        setView({
          cx: cx + e.deltaX / scale,
          cy: cy - e.deltaY / scale,
          scale: scale,
        });
      }
    };

    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      canvas.removeEventListener('wheel', handleWheel);
    };
  }, [isOpen, dimensions]);

  // Zoom HUD actions
  const handleZoom = (direction: 'in' | 'out') => {
    animRef.current.active = false;
    const zoomFactor = direction === 'in' ? 1.4 : 1 / 1.4;
    setView({
      cx: viewRef.current.cx,
      cy: viewRef.current.cy,
      scale: viewRef.current.scale * zoomFactor,
    });
  };

  const handleReset = () => {
    animRef.current = {
      startTime: Date.now(),
      duration: 800,
      startCx: viewRef.current.cx,
      startCy: viewRef.current.cy,
      startScale: viewRef.current.scale,
      targetCx: initialCx,
      targetCy: initialCy,
      targetScale: fitScaleRef.current,
      active: true,
    };
  };

  const handleFlyToUser = () => {
    if (!mapXy) return;
    animRef.current = {
      startTime: Date.now(),
      duration: 600,
      startCx: viewRef.current.cx,
      startCy: viewRef.current.cy,
      startScale: viewRef.current.scale,
      targetCx: mapXy[0],
      targetCy: mapXy[1],
      targetScale: fitScaleRef.current * 6.0,
      active: true,
    };
  };

  // Pointer panning handlers
  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.setPointerCapture(e.pointerId);
    }
    animRef.current.active = false;
    dragRef.current.isDragging = true;
    dragRef.current.lastMouseX = e.clientX;
    dragRef.current.lastMouseY = e.clientY;
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const { width, height } = dimensions;
    if (width === 0 || height === 0) return;

    // 1. Panning
    if (dragRef.current.isDragging) {
      const dx = e.clientX - dragRef.current.lastMouseX;
      const dy = e.clientY - dragRef.current.lastMouseY;
      dragRef.current.lastMouseX = e.clientX;
      dragRef.current.lastMouseY = e.clientY;

      setView({
        cx: viewRef.current.cx - dx / viewRef.current.scale,
        cy: viewRef.current.cy + dy / viewRef.current.scale,
        scale: viewRef.current.scale,
      });

      setHoveredPoint(null);
      return;
    }

    // 2. Tooltip nearest point search
    const { cx, cy, scale } = viewRef.current;
    const fitScale = fitScaleRef.current;
    const zoom = scale / fitScale;

    const visibleWidth = width / scale;
    const visibleHeight = height / scale;
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
    let minDistance = 8.0;

    const numPoints = points.x.length;
    for (let i = 0; i < numPoints; i++) {
      const px = points.x[i];
      const py = points.y[i];

      if (px < visibleX0 || px > visibleX1 || py < visibleY0 || py > visibleY1) {
        continue;
      }

      const malId = points.mal_id[i];
      const isRec = recLookupMap.has(malId);
      const pop = points.popularity[i];

      if (pop >= popularityThreshold && !isRec) {
        continue;
      }

      const sx = (px - cx) * scale + width / 2;
      const sy = -(py - cy) * scale + height / 2;

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

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.releasePointerCapture(e.pointerId);
    }
    dragRef.current.isDragging = false;
  };

  const handlePointerLeave = () => {
    dragRef.current.isDragging = false;
    setHoveredPoint(null);
  };

  const handleCanvasClick = () => {
    if (hoveredPoint) {
      onSelectAnime(hoveredPoint.malId);
    }
  };

  // Pinch zoom touch handlers for mobile
  const handleTouchStart = (e: React.TouchEvent<HTMLCanvasElement>) => {
    if (e.touches.length === 2) {
      animRef.current.active = false;
      dragRef.current.isPinching = true;
      const d = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      dragRef.current.pinchDistance = d;
    }
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLCanvasElement>) => {
    if (dragRef.current.isPinching && e.touches.length === 2) {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const { width, height } = dimensions;
      if (width === 0 || height === 0) return;

      const d = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY
      );
      const factor = d / dragRef.current.pinchDistance;
      dragRef.current.pinchDistance = d;

      const clientCenterX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
      const clientCenterY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
      const canvasCenterX = clientCenterX - rect.left;
      const canvasCenterY = clientCenterY - rect.top;

      const { cx, cy, scale } = viewRef.current;
      const oldScale = scale;
      const nextScale = scale * factor;

      const wx = (canvasCenterX - width / 2) / oldScale + cx;
      const wy = -(canvasCenterY - height / 2) / oldScale + cy;

      setView({
        cx: wx - (canvasCenterX - width / 2) / nextScale,
        cy: wy + (canvasCenterY - height / 2) / nextScale,
        scale: nextScale,
      });
    }
  };

  const handleTouchEnd = () => {
    dragRef.current.isPinching = false;
  };

  // Backdrop click validation: drag starting inside must not trigger close.
  const handleBackdropPointerDown = (e: React.PointerEvent) => {
    if (e.target === e.currentTarget) {
      backdropPointerDownRef.current = true;
    } else {
      backdropPointerDownRef.current = false;
    }
  };

  const handleBackdropPointerUp = (e: React.PointerEvent) => {
    if (e.target === e.currentTarget && backdropPointerDownRef.current) {
      onClose();
    }
    backdropPointerDownRef.current = false;
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-50 overflow-hidden flex items-center justify-center bg-black/60"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          onPointerDown={handleBackdropPointerDown}
          onPointerUp={handleBackdropPointerUp}
        >
          <motion.div
            layoutId="taste-map-container"
            className="relative w-[92vw] h-[84vh] max-w-[1500px] bg-[#0b1020] rounded-3xl overflow-hidden shadow-2xl flex flex-col"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
            onPointerDown={(e) => e.stopPropagation()}
            onPointerUp={(e) => e.stopPropagation()}
          >
            {/* Main Map canvas */}
            <div ref={containerRef} className="absolute inset-0 w-full h-full">
              <canvas
                ref={canvasRef}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerLeave={handlePointerLeave}
                onClick={handleCanvasClick}
                onTouchStart={handleTouchStart}
                onTouchMove={handleTouchMove}
                onTouchEnd={handleTouchEnd}
                className={`absolute inset-0 block select-none ${
                  hoveredPoint 
                    ? 'cursor-pointer' 
                    : dragRef.current.isDragging 
                      ? 'cursor-grabbing' 
                      : 'cursor-grab'
                }`}
              />
            </div>

            {/* Close Button overlay */}
            <button
              onClick={onClose}
              className="absolute top-6 right-6 z-50 p-2.5 bg-black/40 hover:bg-black/60 text-white/90 border border-white/10 rounded-full shadow-lg transition-colors cursor-pointer"
              aria-label="Close Map Explorer"
            >
              <X className="w-5 h-5" />
            </button>

            {/* Controls HUD */}
            <div className="absolute top-6 left-6 z-50 flex gap-2">
              <button
                onClick={() => handleZoom('in')}
                className="p-2.5 bg-black/40 hover:bg-black/60 text-white border border-white/10 rounded-full shadow-lg transition-colors cursor-pointer"
                title="Zoom In"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleZoom('out')}
                className="p-2.5 bg-black/40 hover:bg-black/60 text-white border border-white/10 rounded-full shadow-lg transition-colors cursor-pointer"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <button
                onClick={handleReset}
                className="p-2.5 bg-black/40 hover:bg-black/60 text-white border border-white/10 rounded-full shadow-lg transition-colors cursor-pointer"
                title="Reset View"
              >
                <Home className="w-4 h-4" />
              </button>
            </div>

            {/* Controls panel at Bottom-Left */}
            <div className="absolute bottom-6 left-6 z-50 flex flex-col sm:flex-row items-start sm:items-center gap-3 select-none">
              {/* Legend Overlay */}
              <div className="flex items-center gap-3 bg-[#0b1020]/95 px-4 py-2 border border-white/10 rounded-full text-[10px] uppercase font-bold tracking-wider text-white/70 shadow-2xl backdrop-blur">
                <span className="flex items-center gap-1.5">
                  <span className="inline-block w-2.5 h-2.5 rounded-full bg-indigo-400" />
                  Your taste map
                </span>
                <span className="text-white/20">·</span>
                <button
                  onClick={() => setShowRecommended(!showRecommended)}
                  className={`flex items-center gap-1.5 cursor-pointer transition-colors ${
                    showRecommended ? 'text-white hover:text-white/80' : 'text-white/30 hover:text-white/50'
                  }`}
                  title="Toggle recommended highlights"
                >
                  <span className={`inline-block w-2.5 h-2.5 rounded-full border border-white transition-opacity ${
                    showRecommended ? 'bg-indigo-500' : 'bg-transparent opacity-30'
                  }`} />
                  Recommended
                </button>
                <span className="text-white/20">·</span>
                <button
                  onClick={handleFlyToUser}
                  disabled={!mapXy}
                  className={`flex items-center gap-1.5 transition-colors ${
                    mapXy 
                      ? 'cursor-pointer text-white hover:text-white/80' 
                      : 'cursor-not-allowed text-white/20 opacity-40'
                  }`}
                  title={mapXy ? 'Fly to your position' : 'Your position unavailable'}
                >
                  <span className={`text-[10px] ${mapXy ? 'text-white animate-pulse' : 'text-white/20'}`}>✦</span>
                  You
                </button>
              </div>

              {/* Highlight Slider */}
              {showRecommended && (
                <div className="flex items-center gap-3 bg-[#0b1020]/95 px-4 py-2 border border-white/10 rounded-full text-[10px] uppercase font-bold tracking-wider text-white/70 shadow-2xl backdrop-blur">
                  <span className="text-white/50">Highlight top:</span>
                  <input
                    type="range"
                    min="1"
                    max="500"
                    value={highlightTopN}
                    onChange={(e) => setHighlightTopN(Number(e.target.value))}
                    className="w-28 sm:w-36 accent-white bg-white/20 h-1 cursor-pointer outline-none"
                  />
                  <span className="text-white font-mono min-w-[24px] text-right">{highlightTopN}</span>
                </div>
              )}
            </div>

            {/* Hover Tooltip Overlay */}
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
                
                <div className="mt-1 flex items-center gap-1.5 flex-wrap">
                  {hoveredPoint.isRec ? (
                    <span className={`px-1.5 py-0.5 rounded text-[9px] uppercase font-extrabold tracking-wider ${
                      hoveredPoint.isCold 
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' 
                        : 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                    }`}>
                      {hoveredPoint.isCold ? 'New & Trending (Cold)' : `Recommended #${hoveredPoint.rank}`}
                    </span>
                  ) : (
                    <span className="px-1.5 py-0.5 rounded text-[9px] uppercase font-extrabold tracking-wider bg-slate-800 text-slate-300 border border-slate-700">
                      Standard
                    </span>
                  )}
                  <span className="text-[9px] text-slate-400 font-medium">Click to view details</span>
                </div>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
