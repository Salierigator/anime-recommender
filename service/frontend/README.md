# service/frontend/

React 19 + Vite + TypeScript + Tailwind CSS 4 single-page app. Enter a MyAnimeList username (or hand-pick favorite anime in guest mode) and get personalized recommendations from the backend — with client-side filtering/sorting, shareable URLs, and detail modals.

## Run

```bash
npm install
npm run dev        # http://localhost:5173 — needs backend on :8000 (mock mode is enough)
npm run build      # tsc -b + vite build → dist/
npm run lint
```

Backend base URL comes from `VITE_API_URL` (default `http://localhost:8000`). The API it calls is documented in [`../backend/`](../backend/) — live request/response shapes are in the backend's Swagger UI at `/docs`.

## Source layout

```
src/
├── main.tsx / index.css / App.css   # entry + Tailwind theme tokens
├── App.tsx                # wiring: hooks + page layout (header, results, modal)
├── api.ts                 # axios client, one function per endpoint
├── types.ts               # API shapes + UI state types (Tab, TabPrefs, FacetOptions)
├── components/
│   ├── SearchForm.tsx           # tabs (username/guest), inputs, guest picker
│   ├── FilterPanel.tsx          # filter/sort/show-K panel, sticky + compact on scroll
│   ├── MultiSelectDropdown.tsx  # generic dropdown used by FilterPanel
│   ├── ResultsSection.tsx       # one results grid (rendered twice: Main and Cold)
│   ├── AnimeCard.tsx            # card + poster loading/fallback
│   └── AnimeModal.tsx           # detail modal (Jikan primary, backend fallback)
├── hooks/
│   ├── useRecommendations.ts    # per-tab result pools + handleSearch (dedupe/abort)
│   ├── useResultsPipeline.ts    # pool → facets → filter → sort → slice
│   ├── useTabPrefs.ts           # per-tab filter/sort/display prefs (one object per tab)
│   ├── useGuestState.ts         # guest picks + watched set + localStorage persistence
│   └── useUrlSync.ts            # URL ?u=/?ids=/?watched=/?sort= ↔ state (share links)
└── utils/
    ├── sortAnime.ts             # client-side sort (relevance/score/popularity/date)
    └── jikanDetail.ts           # client-side Jikan fetch for the modal (cached, backend fallback)
```
