import { useMemo } from 'react';

/**
 * Hook to get the background image and overlay style for a specific novel.
 * @param novel - 'noli' or 'fili' (or 'both'/null for fallback)
 * @returns React.CSSProperties object with the background styling
 */
export function useNovelBackground(novel: 'noli' | 'fili' | 'both' | string | null | undefined) {
    return useMemo(() => {
        const isFili = novel === 'fili' || novel === 'elfili' || novel === 'El Filibusterismo';
        const bgImage = isFili ? 'elfili_background.jpg' : 'noli_background.jpg';

        // Using the same semi-transparent overlay to ensure text contrast:
        // Fili: rgba(242, 240, 237, opacity)
        // Noli: rgba(245, 241, 233, opacity)
        const overlayColor = isFili ? '242, 240, 237' : '245, 241, 233';

        return {
            backgroundImage: `linear-gradient(to bottom, rgba(${overlayColor}, 0.85), rgba(${overlayColor}, 0.98)), url('/images/backgrounds/${bgImage}')`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            backgroundAttachment: 'fixed',
        };
    }, [novel]);
}
