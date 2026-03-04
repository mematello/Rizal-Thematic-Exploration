import { create } from 'zustand';

type Mode = 'buod' | 'full';

interface ModeState {
    mode: Mode;
    setMode: (mode: Mode) => void;
}

export const useModeStore = create<ModeState>((set) => ({
    mode: 'buod',
    setMode: (mode) => set({ mode }),
}));
