"use client";

import { useEffect, useState } from "react";
import { useIsFetching } from "@tanstack/react-query";

/**
 * ServerWarmupBanner
 *
 * Displays a non-intrusive status notification when a React Query request
 * is taking longer than 3 seconds (indicating a cold start / backend wake-up).
 * Automatically dismisses when queries finish (success or error).
 *
 * Requirements:
 * - Driven strictly by React Query layer (useIsFetching)
 * - Timer >= 3s before display; immediate dismissal on isFetching === 0
 * - Does NOT block interaction (pointer-events-none)
 * - Zero layout shift (fixed position)
 * - Accessible: role="status" and aria-live="polite"
 * - WCAG AA compliant contrast in light and dark modes
 * - Respects mobile safe-area-inset-top
 */
export function ServerWarmupBanner() {
    const isFetching = useIsFetching();
    const [visible, setVisible] = useState(false);

    useEffect(() => {
        let timer: NodeJS.Timeout | null = null;
        if (isFetching > 0) {
            timer = setTimeout(() => {
                setVisible(true);
            }, 3000);
        } else {
            setVisible(false);
        }

        return () => {
            if (timer) {
                clearTimeout(timer);
            }
        };
    }, [isFetching]);

    if (!visible) return null;

    return (
        <div
            role="status"
            aria-live="polite"
            className="fixed top-[max(0.75rem,env(safe-area-inset-top))] left-1/2 -translate-x-1/2 z-50 pointer-events-none transition-all duration-300"
            data-testid="server-warmup-banner"
        >
            <div className="flex items-center gap-2.5 px-4 py-2 rounded-full shadow-lg border border-[#C5A065]/40 bg-[#FCFAF7]/95 text-[#1B263B] dark:bg-[#131B2A]/95 dark:text-[#F0F4F8] dark:border-[#D4AF37]/30 backdrop-blur-md text-xs sm:text-sm font-medium tracking-wide">
                <span className="relative flex h-2 w-2" aria-hidden="true">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#C5A065] opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-[#C5A065]"></span>
                </span>
                <span>Ginigising ang server... Sandali lamang habang inihahanda ang datos.</span>
            </div>
        </div>
    );
}
