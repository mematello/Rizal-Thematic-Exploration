"use client";

import { useEffect, useState } from "react";
import { useIsFetching } from "@tanstack/react-query";

/**
 * ServerWarmupBanner
 *
 * Displays a non-intrusive status notification at bottom center when an initial
 * user-facing React Query request is taking longer than 3 seconds (indicating a cold start).
 * Automatically dismisses when queries finish (success or error).
 *
 * Requirements:
 * - Positioned at bottom center with safe-area lift to clear mobile navigation bars
 * - High z-index (z-[70]) to render above modal overlays (z-50)
 * - Excludes "Sa Itaas" button (which sits at bottom-right)
 * - Counts ONLY user-facing initial loads (queries observed by active UI with data === undefined)
 * - IGNORES background prefetches (query.getObserversCount() === 0)
 * - IGNORES background refetches (query.state.data !== undefined)
 * - Timer >= 3s before display; immediate dismissal on completion
 * - Does NOT block interaction (pointer-events-none)
 * - Zero layout shift (fixed position)
 * - Accessible: role="status" and aria-live="polite"
 * - WCAG AA/AAA compliant contrast in light and dark modes
 */
export function ServerWarmupBanner() {
    const isFetching = useIsFetching({
        predicate: (query) => {
            // Must have active component observing this query (excludes unobserved prefetches)
            const isObserved = query.getObserversCount() > 0;
            // Must be an initial load with no data yet (excludes background refetches)
            const hasNoData = query.state.data === undefined;
            // Must be actively fetching
            const inFlight = query.state.fetchStatus === "fetching";
            return isObserved && hasNoData && inFlight;
        },
    });

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
            className="fixed bottom-[max(1.25rem,calc(env(safe-area-inset-bottom)+0.75rem))] sm:bottom-6 left-1/2 -translate-x-1/2 z-[70] pointer-events-none transition-all duration-300 w-auto max-w-[92vw] sm:max-w-max animate-in fade-in slide-in-from-bottom-2"
            data-testid="server-warmup-banner"
        >
            <div className="flex items-center gap-2.5 px-4 py-2.5 rounded-full shadow-2xl border border-[#C5A065]/40 bg-[#FCFAF7]/95 text-[#1B263B] dark:bg-[#131B2A]/95 dark:text-[#F0F4F8] dark:border-[#D4AF37]/30 backdrop-blur-md text-xs sm:text-sm font-medium tracking-wide">
                <span className="relative flex h-2 w-2 shrink-0" aria-hidden="true">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#C5A065] opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-[#C5A065]"></span>
                </span>
                <span className="truncate">Ginigising ang server... Sandali lamang habang inihahanda ang datos.</span>
            </div>
        </div>
    );
}
