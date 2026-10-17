import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export type SpecificNovel = 'noli' | 'fili';

interface NovelState {
    selectedNovel: SpecificNovel;
    setSelectedNovel: (novel: SpecificNovel) => void;
}

export const useNovelStore = create<NovelState>()(
    persist(
        (set) => ({
            selectedNovel: 'noli',
            setSelectedNovel: (novel) => {
                if (novel === 'noli' || novel === 'fili') {
                    set({ selectedNovel: novel });
                    if (typeof document !== 'undefined') {
                        document.documentElement.setAttribute('data-novel', novel);
                    }
                }
            },
        }),
        {
            name: 'rizal-novel-store',
            storage: createJSONStorage(() => localStorage),
        }
    )
);
