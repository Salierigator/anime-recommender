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

      // 1. Calculate Center and Uniform Scale
      const [x0, x1, y0, y1] = meta.extent;
      const centerX = mapXy ? mapXy[0] : (x0 + x1) / 2;
      const centerY = mapXy ? mapXy[1] : (y0 + y1) / 2;

      const viewH = (y1 - y0) / 3;
      const scale = height / viewH;
      const viewW = width / scale;

      const visible_x0 = centerX - viewW / 2;
      const visible_x1 = centerX + viewW / 2;
      const visible_y0 = centerY - viewH / 2;
      const visible_y1 = centerY + viewH / 2;

      const render = () => {
        // Clear background
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, width, height);

        // 2. Draw background image (territory.png) using uniform scaling
        if (bgImage && !imageError) {
          const imgLeft = (x0 - centerX) * scale + width / 2;
          const imgRight = (x1 - centerX) * scale + width / 2;
          const imgTop = -(y1 - centerY) * scale + height / 2;
          const imgBottom = -(y0 - centerY) * scale + height / 2;

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

          const sx = (px - centerX) * scale + width / 2;
          const sy = -(py - centerY) * scale + height / 2;

          const label = previewPoints.label[i];
          const hue = (label % k) / k * 360;

          ctx.fillStyle = `hsla(${hue}, 65%, 62%, 0.75)`;
          ctx.beginPath();
          ctx.arc(sx, sy, 2, 0, 2 * Math.PI);
          ctx.fill();
        }

        // 4. Draw Vignette & Bottom Text Gradient
        const vignetteGrad = ctx.createRadialGradient(
          width / 2, height / 2, Math.min(width, height) * 0.4,
          width / 2, height / 2, Math.max(width, height) * 0.7
        );
        vignetteGrad.addColorStop(0, 'rgba(11, 16, 32, 0)');
        vignetteGrad.addColorStop(1, 'rgba(11, 16, 32, 0.75)');
        ctx.fillStyle = vignetteGrad;
        ctx.fillRect(0, 0, width, height);

        const bottomGrad = ctx.createLinearGradient(0, height - 60, 0, height);
        bottomGrad.addColorStop(0, 'rgba(11, 16, 32, 0)');
        bottomGrad.addColorStop(1, 'rgba(11, 16, 32, 0.85)');
        ctx.fillStyle = bottomGrad;
        ctx.fillRect(0, height - 60, width, 60);

        // 5. Draw pulsing user marker at the center
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
      className="relative w-full h-[200px] rounded-2xl overflow-hidden cursor-pointer select-none"
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
