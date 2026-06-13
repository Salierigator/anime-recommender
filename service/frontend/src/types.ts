export interface AnimeItem {
  mal_id: number;
  title: string;
  type: string;
  year: number | null;
  mal_score: number | null;
  pred?: number | null;
  cos?: number | null;
  image_url: string | null;
}

export interface RecommendMeta {
  source: string;
  split: string;
  history_count: number;
  alpha: number | null;
  k_retrieve: number | null;
  mode: string;
}

export interface RecommendRequest {
  username?: string | null;
  mal_ids?: number[] | null;
  top_k?: number;
  cold_k?: number;
  live?: boolean;
}

export interface RecommendResponse {
  main: AnimeItem[];
  cold: AnimeItem[];
  meta: RecommendMeta;
}
