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
}

export interface RecommendMeta {
  source: string;
  split: string;
  history_count: number;
  alpha: number | null;
  k_retrieve: number | null;
  mode: string;
  total_entries: number | null;
}

export interface RecommendRequest {
  username: string;
  top_k: number;
  cold_k: number;
  sfw?: boolean;
}

export interface RecommendResponse {
  main: AnimeItem[];
  cold: AnimeItem[];
  meta: RecommendMeta;
}
