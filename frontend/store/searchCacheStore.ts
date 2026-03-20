import { create } from 'zustand';

interface ThemeResult {
    label: string;
    confidence: number;
    evidence?: string;
}

interface PaksaData {
    has_theme: boolean;
    themes: ThemeResult[];
}

interface SanggunianData {
    has_reference: boolean;
    reference_text: string;
    alignment_status?: string;
    matched_characters?: string[];
    score: number;
}

interface SearchCacheState {
    paksaCache: Record<number, PaksaData>;
    sanggunianCache: Record<number, SanggunianData>;
    setPaksaBatch: (data: Record<number, PaksaData>) => void;
    setSanggunianBatch: (data: Record<number, SanggunianData>) => void;
}

export const useSearchCacheStore = create<SearchCacheState>((set) => ({
    paksaCache: {},
    sanggunianCache: {},
    setPaksaBatch: (data) => set((state) => ({ 
        paksaCache: { ...state.paksaCache, ...data } 
    })),
    setSanggunianBatch: (data) => set((state) => ({ 
        sanggunianCache: { ...state.sanggunianCache, ...data } 
    })),
}));
