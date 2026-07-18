import axios from 'axios';
import type { RecommendRequest, RecommendResponse, MapResponse } from './types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Fire-and-forget: đánh thức backend free-tier (spin down khi idle) ngay khi trang mở
export const pingHealthAPI = (): void => {
  api.get('/api/health').catch(() => {});
};

export const recommendAPI = async (data: RecommendRequest): Promise<RecommendResponse> => {
  const response = await api.post<RecommendResponse>('/api/recommend', data);
  return response.data;
};

export const fetchPostersAPI = async (ids: number[]): Promise<Record<number, string | null>> => {
  const response = await api.post<{ posters: Record<string, string | null> }>('/api/posters', { ids });
  
  // Convert string keys to number keys
  const numberKeyedPosters: Record<number, string | null> = {};
  for (const [key, value] of Object.entries(response.data.posters)) {
    numberKeyedPosters[Number(key)] = value;
  }
  
  return numberKeyedPosters;
};

let cachedMapData: MapResponse | null = null;

export const fetchMapAPI = async (): Promise<MapResponse> => {
  if (cachedMapData) {
    return cachedMapData;
  }
  const response = await api.get<MapResponse>('/api/map');
  cachedMapData = response.data;
  return response.data;
};

export { API_URL };
