import { useEffect, useRef, useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import type { MapResponse } from '../types';
import { API_URL } from '../api';

interface MapPreviewProps {
  mapData: MapResponse;
  mapXy: [number, number] | null;
  onClick: () => void;
}

export function MapPreview({ mapData, mapXy, onClick }: MapPreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [bgImage, setBgImage] = useState<HTMLImageElement | null>(null);
  const [imageError, setImageError] = useState(false);

  const { points, meta } = mapData;
  const k = meta.k || 28;
  const bg = meta.bg || '#0b1020';

  // Load territory image if available
  useEffect(() => {
    if (!meta.territory_url) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = `${API_URL}${meta.territory_url}`;
    img.onload = () => setBgImage(img);
    img.onerror = () => setImageError(true);
  }, [meta.territory_url]);

  // Pre-filter points for preview performance and clarity (popularity < 5000)
  const previewPoints = useMemo(() => {
    const indices: number[] = [];
    const numPoints = points.mal_id.length;
    for (let i = 0; i < numPoints; i++) {
      if (points.popularity[i] < 5000) {
        indices.push(i);
      }
    }

    return {
      x: indices.map(i => points.x[i]),
      y: indices.map(i => points.y[i]),
      label: indices.map(i => points.label[i]),
    };
  }, [points]);

  useEffect(() => {
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

      // 1. Calculate Viewport Bounds
      const [x0, x1, y0, y1] = meta.extent;
      let visible_x0 = x0;
      let visible_x1 = x1;
      let visible_y0 = y0;
      let visible_y1 = y1;

      if (mapXy !== null) {
        // Zoomed view: centered on mapXy showing 1/4th of the extent size
        const ex_w = x1 - x0;
        const ex_h = y1 - y0;
        const zoomWidth = ex_w / 4;
        const zoomHeight = ex_h / 4;

        visible_x0 = mapXy[0] - zoomWidth / 2;
        visible_x1 = mapXy[0] + zoomWidth / 2;
        visible_y0 = mapXy[1] - zoomHeight / 2;
        visible_y1 = mapXy[1] + zoomHeight / 2;
      }

      const scaleX = width / (visible_x1 - visible_x0);
      const scaleY = height / (visible_y1 - visible_y0);

      const render = () => {
        // Clear background
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, width, height);

        // 2. Draw background image (territory.png) stretched to full extent coords
        if (bgImage && !imageError) {
          const imgLeft = (x0 - visible_x0) * scaleX;
          const imgRight = (x1 - visible_x0) * scaleX;
          const imgTop = height - (y1 - visible_y0) * scaleY;
          const imgBottom = height - (y0 - visible_y0) * scaleY;

          ctx.drawImage(bgImage, imgLeft, imgTop, imgRight - imgLeft, imgBottom - imgTop);
        }

        // 3. Draw points
        const numPoints = previewPoints.x.length;
        for (let i = 0; i < numPoints; i++) {
          const px = previewPoints.x[i];
          const py = previewPoints.y[i];
          
          // Clip points outside the viewport
          if (px < visible_x0 || px > visible_x1 || py < visible_y0 || py > visible_y1) {
            continue;
          }

          const sx = (px - visible_x0) * scaleX;
          const sy = height - (py - visible_y0) * scaleY;

          const label = previewPoints.label[i];
          const hue = (label % k) / k * 360;

          ctx.fillStyle = `hsla(${hue}, 65%, 62%, 0.75)`;
          ctx.beginPath();
          ctx.arc(sx, sy, 2, 0, 2 * Math.PI);
          ctx.fill();
        }

        // 4. Draw pulsing user marker at the center
        if (mapXy !== null) {
          const cx = width / 2;
          const cy = height / 2;

          const now = Date.now();
          const pulse = (now % 1500) / 1500; // 0 to 1

          // Outer pulsing ring
          const outerRadius = 6 + pulse * 14;
          const outerOpacity = 0.8 * (1 - pulse);
          ctx.strokeStyle = `rgba(255, 255, 255, ${outerOpacity})`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(cx, cy, outerRadius, 0, 2 * Math.PI);
          ctx.stroke();

          // Inner solid dot
          ctx.fillStyle = '#ffffff';
          ctx.beginPath();
          ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
          ctx.fill();
        }

        animationId = requestAnimationFrame(render);
      };

      render();
    };

    resizeAndRender();
    
    // Resize listener
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
  }, [previewPoints, bgImage, imageError, mapXy, meta.extent, k, bg]);

  const bannerText = mapXy !== null 
    ? 'YOUR TASTE MAP — Click to explore ⤢' 
    : 'ANIME MAP — Click to explore ⤢';

  return (
    <motion.div
      ref={containerRef}
      layoutId="taste-map-container"
      onClick={onClick}
      className="relative w-full h-[160px] rounded-2xl overflow-hidden cursor-pointer select-none"
      style={{ backgroundColor: bg }}
      whileHover={{ scale: 1.01, filter: 'brightness(1.1)' }}
      transition={{ type: 'spring', stiffness: 300, damping: 25 }}
    >
      <canvas ref={canvasRef} className="absolute inset-0 block" />
      
      {/* Corner text banner overlay */}
      <div className="absolute bottom-4 left-6 pointer-events-none">
        <span className="text-xs uppercase font-medium tracking-[0.2em] text-white/70 drop-shadow-sm">
          {bannerText}
        </span>
      </div>
    </motion.div>
  );
}
