import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  // GitHub Pages serve ở https://<user>.github.io/anime-recommender/ nên bundle build cần prefix
  // đó trong asset path. Dev server vẫn ở '/' như cũ.
  base: command === 'build' ? '/anime-recommender/' : '/',
  plugins: [react(), tailwindcss()],
}))
