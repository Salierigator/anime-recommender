# Frontend — Anime Recommender

React 19 + Vite + TypeScript + Tailwind CSS 4 single-page app. Enter a MyAnimeList username
(or hand-pick favorite anime in guest mode) and get personalized recommendations from the
backend — with client-side filtering/sorting, shareable URLs, a taste map, and detail modals.

## Run

```bash
npm install
npm run dev        # http://localhost:5173 — needs backend on :8000 (mock mode is enough)
npm run build      # tsc -b + vite build → dist/
npm run lint
```

Backend base URL comes from `VITE_API_URL` (default `http://localhost:8000`).
Request/response shapes: [`../API_CONTRACT.md`](../API_CONTRACT.md).
Visual conventions: [`DESIGN.md`](DESIGN.md).

## Source layout

```
src/
├── main.tsx / index.css / App.css   # entry + Tailwind theme tokens
├── App.tsx                # wiring: hooks + page layout (header, results, modal, map)
├── api.ts                 # axios client, one function per endpoint
├── types.ts               # API shapes + UI state types (Tab, TabPrefs, FacetOptions)
├── components/
│   ├── SearchForm.tsx           # tabs (username/guest), inputs, guest picker
│   ├── FilterPanel.tsx          # filter/sort/show-K panel, sticky + compact on scroll
│   ├── MultiSelectDropdown.tsx  # generic dropdown used by FilterPanel
│   ├── ResultsSection.tsx       # one results grid (rendered twice: Main and Cold)
│   ├── AnimeCard.tsx            # card + poster loading/fallback
│   ├── AnimeModal.tsx           # detail modal (Jikan primary, backend fallback)
│   ├── MapPreview.tsx           # small taste-map banner
│   └── MapExplorer.tsx          # fullscreen canvas map (pan/zoom/hover)
├── hooks/
│   ├── useRecommendations.ts    # per-tab result pools + handleSearch (dedupe/abort)
│   ├── useResultsPipeline.ts    # pool → facets → filter → sort → slice
│   ├── useTabPrefs.ts           # per-tab filter/sort/display prefs (one object per tab)
│   ├── useGuestState.ts         # guest picks + watched set + localStorage persistence
│   └── useUrlSync.ts            # URL ?u=/?ids=/?watched=/?sort= ↔ state (share links)
└── utils/
    ├── sortAnime.ts             # client-side sort (relevance/score/popularity/date)
    └── jikanQueue.ts            # rate-limited Jikan queue (posters/details fallback)
```

Convention: components render + hold local UI state only; cross-cutting state and effects
live in `hooks/`; `App.tsx` only wires them together. New backend calls go in `api.ts`,
new shared types in `types.ts` — keep `App.tsx`/`SearchForm.tsx` from growing back into
monoliths.
