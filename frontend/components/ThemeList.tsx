"use client";

import { useState, useRef, useEffect, useLayoutEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, BookOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import { ItemModal } from "@/components/ItemModal";

const useIsomorphicLayoutEffect = typeof window !== "undefined" ? useLayoutEffect : useEffect;

/**
 * Fit helper used only by theme titles:
 * After layout, if the title's longest unbreakable word is wider than the container's
 * content width, scale font-size down by the minimum needed to fit, with a floor of 85%
 * of the original 24px (text-2xl). Recomputes on resize via ResizeObserver and avoids
 * a visible flash on first paint by measuring in useLayoutEffect before paint.
 */
function useThemeTitleFit(title: string) {
    const headingRef = useRef<HTMLHeadingElement>(null);
    const [fontSizeStyle, setFontSizeStyle] = useState<string | undefined>(undefined);

    useIsomorphicLayoutEffect(() => {
        const el = headingRef.current;
        if (!el) return;

        const measureAndFit = () => {
            const availableWidth = el.clientWidth;
            if (availableWidth <= 0) return;

            // Extract unbreakable segments (words with hyphens wrap naturally at '-')
            const words = title.split(/\s+/);
            const segments: string[] = [];
            for (const w of words) {
                if (w.includes("-")) {
                    segments.push(...w.split("-"));
                } else {
                    segments.push(w);
                }
            }

            // Create offscreen measurement span with base text-2xl font-serif font-bold
            const testSpan = document.createElement("span");
            testSpan.className = "text-2xl font-serif font-bold";
            testSpan.style.position = "absolute";
            testSpan.style.visibility = "hidden";
            testSpan.style.whiteSpace = "nowrap";
            testSpan.style.pointerEvents = "none";
            document.body.appendChild(testSpan);

            let maxSegmentWidth = 0;
            for (const seg of segments) {
                testSpan.textContent = seg;
                const w = testSpan.getBoundingClientRect().width;
                if (w > maxSegmentWidth) maxSegmentWidth = w;
            }
            testSpan.remove();

            // If longest unbreakable segment exceeds available width, scale down (floor: 85%)
            if (maxSegmentWidth > availableWidth) {
                const neededRatio = availableWidth / maxSegmentWidth;
                const scale = Math.max(0.85, Math.min(1, neededRatio * 0.99));
                setFontSizeStyle(`${(24 * scale).toFixed(2)}px`);
            } else {
                setFontSizeStyle(undefined);
            }
        };

        measureAndFit();

        if (typeof document !== "undefined" && document.fonts) {
            document.fonts.ready.then(measureAndFit);
        }

        const ro = new ResizeObserver(measureAndFit);
        ro.observe(el);
        return () => ro.disconnect();
    }, [title]);

    return { headingRef, fontSizeStyle };
}

export interface ThemeContext {
    book: string;
    chapter_number: number;
    chapter_title: string;
    sentence_text: string;
    sentence_index: number;
}

export interface Theme {
    book?: string;
    tagalog_title: string;
    meaning: string;
    best_match: ThemeContext | null;
}

export async function fetchThemes(): Promise<Theme[]> {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
    const apiUrl = baseUrl.endsWith("/") ? baseUrl.slice(0, -1) : baseUrl;
    const res = await fetch(`${apiUrl}/api/v1/themes`);
    if (!res.ok) throw new Error("Failed to fetch themes");
    return res.json();
}

interface ThemeListProps {
    onChapterSelect?: (book: string, chapter: number, title?: string, sentenceIndex?: number) => void;
    selectedNovel: "noli" | "fili";
}

export function ThemeList({ onChapterSelect, selectedNovel }: ThemeListProps) {
    const { data: themes = [], isLoading } = useQuery<Theme[]>({
        queryKey: ["themes"],
        queryFn: fetchThemes,
        staleTime: 1000 * 60 * 60 * 24, // 24 hours
        gcTime: 1000 * 60 * 60 * 24,
    });

    // Modal
    const [selectedTheme, setSelectedTheme] = useState<Theme | null>(null);
    const [isModalOpen, setIsModalOpen] = useState(false);

    const handleThemeClick = (theme: Theme) => {
        setSelectedTheme(theme);
        setIsModalOpen(true);
    };

    const handleClose = () => {
        setIsModalOpen(false);
        setTimeout(() => setSelectedTheme(null), 300);
    };

    if (isLoading) {
        return (
            <div className="max-w-7xl mx-auto px-4 pb-20">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                    {Array.from({ length: 8 }).map((_, i) => (
                        <div
                            key={i}
                            className="bg-brand-paper p-8 rounded-sm relative overflow-hidden border border-brand-gold/10 animate-pulse"
                        >
                            <div className="w-14 h-4 bg-brand-gold/15 rounded-sm mb-4" />
                            <div className="w-3/4 h-7 bg-brand-navy/15 rounded-sm mb-3" />
                            <div className="w-12 h-0.5 bg-brand-gold/20 mb-4" />
                            <div className="space-y-2 mb-6">
                                <div className="w-full h-3 bg-brand-text-light/10 rounded-sm" />
                                <div className="w-5/6 h-3 bg-brand-text-light/10 rounded-sm" />
                                <div className="w-2/3 h-3 bg-brand-text-light/10 rounded-sm" />
                            </div>
                            <div className="w-24 h-3 bg-brand-navy/10 rounded-sm" />
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    return (
        <div className="max-w-7xl mx-auto px-4 pb-20">
            <motion.div
                key={selectedNovel}
                className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6"
                initial="hidden"
                animate="show"
                variants={{
                    hidden: { opacity: 0 },
                    show: {
                        opacity: 1,
                        transition: { staggerChildren: 0.1 }
                    }
                }}
            >
                {themes
                    .filter(theme => {
                        if (!theme.best_match) return true;
                        const themeBook = theme.best_match.book === 'elfili' ? 'fili' : theme.best_match.book;
                        return themeBook === selectedNovel;
                    })
                    .map((theme, idx) => (
                        <ThemeCard key={idx} theme={theme} onClick={() => handleThemeClick(theme)} />
                    ))}
            </motion.div>

            {selectedTheme && (
                <ItemModal
                    isOpen={isModalOpen}
                    onClose={handleClose}
                    title={selectedTheme.tagalog_title}
                    type="theme"
                    meaning={selectedTheme.meaning}
                    themeContext={selectedTheme.best_match || undefined}
                    isLoading={false} // Data is already loaded
                    selectedNovel={selectedNovel} // Pass the prop
                    onNavigate={(book, chapter, sentenceIndex) => {
                        console.log('Theme context clicked:', book, chapter, sentenceIndex);
                        handleClose();
                        onChapterSelect?.(book, chapter, undefined, sentenceIndex);
                    }}
                />
            )}
        </div>
    );
}

function ThemeCard({ theme, onClick }: { theme: Theme; onClick: () => void }) {
    const { headingRef, fontSizeStyle } = useThemeTitleFit(theme.tagalog_title);

    return (
        <motion.div
            variants={{
                hidden: { opacity: 0, y: 20 },
                show: { opacity: 1, y: 0 }
            }}
            whileHover={{ y: -5, boxShadow: "0 20px 40px -15px rgba(0,0,0,0.1)" }}
            onClick={onClick}
            className="bg-brand-paper dark:bg-[#18202C] p-8 rounded-sm relative overflow-hidden group border border-brand-gold/10 dark:border-brand-gold/20 hover:border-brand-gold/40 transition-all duration-300 cursor-pointer"
        >
            {/* Background Icon */}
            <div aria-hidden="true" className="absolute -right-6 -bottom-6 opacity-[0.03] group-hover:opacity-[0.08] transition-opacity duration-500 transform group-hover:rotate-12 group-hover:scale-110">
                <BookOpen size={140} className="text-brand-navy dark:text-brand-gold" />
            </div>

            <div className="relative z-10">
                <div className="flex items-center space-x-2 mb-4">
                    <span className={cn(
                        "text-[10px] font-bold tracking-[0.2em] uppercase px-2 py-1 rounded-sm",
                        "bg-brand-gold/10 text-brand-gold"
                    )}>
                        Paksa
                    </span>
                </div>

                <h3
                    ref={headingRef}
                    style={fontSizeStyle ? { fontSize: fontSizeStyle } : undefined}
                    className="text-2xl font-serif text-brand-navy dark:text-foreground font-bold leading-tight mb-3 group-hover:text-brand-gold transition-colors duration-300"
                >
                    {theme.tagalog_title}
                </h3>

                <div className="w-12 h-0.5 bg-brand-gold/30 mb-4 group-hover:w-full transition-all duration-500 ease-out" />

                <p className="text-brand-text-light dark:text-brand-text-muted font-body leading-relaxed text-sm line-clamp-3">
                    {theme.meaning}
                </p>

                <div className="mt-4 text-[10px] font-bold text-brand-navy/40 dark:text-brand-gold/60 uppercase tracking-[0.2em] group-hover:text-brand-navy dark:group-hover:text-brand-gold transition-colors">
                    Tingnan ang Konteksto
                </div>
            </div>
        </motion.div>
    );
}
