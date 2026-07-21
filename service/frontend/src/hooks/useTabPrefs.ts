import { useCallback, useState } from 'react';
import type { Tab, TabPrefs } from '../types';

export type PrefsPatch = Partial<TabPrefs> | ((prev: TabPrefs) => Partial<TabPrefs>);
export type UpdatePrefs = (patch: PrefsPatch) => void;

const DEFAULT_PREFS: TabPrefs = {
  genres: [],
  themes: [],
  studios: [],
  types: [],
  minScore: 0,
  mainK: 20,
  coldK: 10,
  sortBy: 'relevance',
  sortAsc: false,
};

/** Gom toàn bộ filter/sort/số hiển thị theo tab vào 1 object mỗi tab. */
export function useTabPrefs(activeTab: Tab) {
  const [prefsByTab, setPrefsByTab] = useState<Record<Tab, TabPrefs>>({
    username: DEFAULT_PREFS,
    guest: DEFAULT_PREFS,
  });

  // Patch prefs của tab đang mở — nhận object hoặc updater (đọc prev, tránh stale).
  const updatePrefs = useCallback<UpdatePrefs>((patch) => {
    setPrefsByTab(prev => {
      const cur = prev[activeTab];
      const p = typeof patch === 'function' ? patch(cur) : patch;
      return { ...prev, [activeTab]: { ...cur, ...p } };
    });
  }, [activeTab]);

  // Patch prefs của 1 tab chỉ định (dùng khi khôi phục sort từ URL lúc mount).
  const updateTabPrefs = useCallback((tab: Tab, patch: Partial<TabPrefs>) => {
    setPrefsByTab(prev => ({ ...prev, [tab]: { ...prev[tab], ...patch } }));
  }, []);

  return { prefs: prefsByTab[activeTab], updatePrefs, updateTabPrefs };
}
