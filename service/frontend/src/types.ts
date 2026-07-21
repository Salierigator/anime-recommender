export interface AnimeItem {
  mal_id: number;
  title: string;
  type: string;
  year: number | null;
  mal_score: number | null;
  pred?: number | null;
  cos?: number | null;
  image_url: string | null;
  genres: string[];
  themes: string[];
  studios: string[];
  popularity: number | null;
  members: number | null;
  start_date: string | null;
}

export interface RecommendMeta {
  source: string;
  split: string;
  history_count: number;
  alpha: number | null;
  k_retrieve: number | null;
  mode: string;
  total_entries: number | null;
  map_xy?: [number, number] | null;
}

export interface RecommendRequest {
  username?: string;
  mal_ids?: number[];
  exclude_ids?: number[];
  top_k: number;
  cold_k: number;
  sfw?: boolean;
}

export interface RecommendResponse {
  main: AnimeItem[];
  cold: AnimeItem[];
  meta: RecommendMeta;
}

export interface MapPoints {
  mal_id: number[];
  title: string[];
  x: number[];
  y: number[];
  label: number[];
  popularity: number[];
  is_cold: boolean[];
}

export interface MapCluster {
  label: number;
  name: string;
  size: number;
  examples: string;
  cx: number;
  cy: number;
}

export interface MapResponse {
  points: MapPoints;
  clusters: MapCluster[];
  meta: {
    k: number;
    n_points: number;
    extent: [number, number, number, number];
    generated: string;
    territory_url: string;
    bg: string;
  };
}

/** 1 entry từ POST /api/posters — poster + score/members MỚI (MAL v2), theo mal_id. */
export interface PosterEntry {
  poster: string | null;
  score: number | null;
  members: number | null;
}

export interface SearchResultItem {
  mal_id: number;
  title: string;
  title_english: string | null;
  type: string | null;
  year: number | null;
  mal_score: number | null;
  image_url: string | null;
  in_corpus: boolean;
}

export interface SearchResponse {
  results: SearchResultItem[];
}

// ---- UI state types (client-side thuần, không phải shape API) ----

export type Tab = 'username' | 'guest';

export type SortKey = 'relevance' | 'score' | 'popularity' | 'date';

/** Filter / sort / số hiển thị của MỖI tab — đổi các giá trị này KHÔNG gọi lại backend. */
export interface TabPrefs {
  genres: string[];
  themes: string[];
  studios: string[];
  types: string[];
  minScore: number;
  mainK: number;
  coldK: number;
  sortBy: SortKey;
  sortAsc: boolean;
}

/** Các lựa chọn filter khả dụng, gom từ pool hiện tại. */
export interface FacetOptions {
  genres: string[];
  themes: string[];
  studios: string[];
  types: string[];
}
