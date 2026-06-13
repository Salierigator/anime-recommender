import axios from 'axios';
import type { RecommendRequest, RecommendResponse } from './types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const recommendAPI = async (data: RecommendRequest): Promise<RecommendResponse> => {
  const response = await api.post<RecommendResponse>('/api/recommend', data);
  return response.data;
};
