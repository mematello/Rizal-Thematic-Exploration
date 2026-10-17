"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { cn, formatChapterTitle } from "@/lib/utils";
import { useModeStore } from "@/store/modeStore";
import { ChapterContent } from "@/types";

type Novel = "noli" | "fili";

export interface Chapter {
    book: string;
    chapter_number: number;
    chapter_title: string;
}

export async function fetchChapters(mode: string = "buod"): Promise<Chapter[]> {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
    const apiUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
    const res = await fetch(`${apiUrl}/api/v1/chapters?mode=${mode}`);
    if (!res.ok) throw new Error("Failed to fetch chapters");
    return res.json();
}

export async function fetchChapterContent(
    book: string,
    chapter: number,
    mode: string = "buod"
): Promise<ChapterContent[]> {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
    const apiUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
    const res = await fetch(`${apiUrl}/api/v1/chapters/${book}/${chapter}?mode=${mode}`);
    if (!res.ok) throw new Error(`Failed to fetch chapter ${chapter}`);
    return res.json();
}

export async function fetchChapterSanggunian(
    book: string,
    chapter: number,
    mode: string = "buod"
) {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
    const apiUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
    const res = await fetch(`${apiUrl}/api/v1/chapters/${book}/${chapter}/sanggunian?mode=${mode}`);
    if (!res.ok) throw new Error(`Failed to fetch sanggunian for chapter ${chapter}`);
    return res.json();
}

interface ChapterGridProps {
    selectedNovel: Novel;
    onChapterSelect: (book: string, chapter: number, title: string) => void;
}

export function ChapterGrid({ selectedNovel, onChapterSelect }: ChapterGridProps) {
    const { mode } = useModeStore();
    const { data: chapters = [], isLoading } = useQuery<Chapter[]>({
        queryKey: ["chapters", mode],
        queryFn: () => fetchChapters(mode),
        staleTime: 1000 * 60 * 60 * 24, // 24 hours
    });

    const getFilteredChapters = () => {
        const noli = chapters
            .filter(c => c.book.toLowerCase().includes("noli"))
            .map(c => ({ ...c, book: "noli", chapter_title: formatChapterTitle(c.chapter_title) }))
            .sort((a, b) => a.chapter_number - b.chapter_number);
        const fili = chapters
            .filter(c => c.book.toLowerCase().includes("fili"))
            .map(c => ({ ...c, book: "elfili", chapter_title: formatChapterTitle(c.chapter_title) }))
            .sort((a, b) => a.chapter_number - b.chapter_number);

        if (selectedNovel === "noli") return noli;
        return fili;
    };

    const items = getFilteredChapters();

    if (isLoading) {
        return (
            <div className="w-full pb-20 space-y-6">
                <div className="grid gap-2.5 sm:gap-6 grid-cols-2 lg:grid-cols-4 animate-pulse">
                    {Array.from({ length: 8 }).map((_, idx) => (
                        <div
                            key={idx}
                            className="p-3.5 sm:p-6 min-h-[9.5rem] sm:h-56 flex flex-col justify-between bg-white/60 dark:bg-brand-paper/60 backdrop-blur-md rounded-xl sm:rounded-sm border border-brand-gold/15"
                        >
                            <div className="space-y-2.5">
                                <div className="h-3 w-20 bg-brand-gold/20 rounded" />
                                <div className="h-5 sm:h-7 w-28 bg-brand-navy/15 dark:bg-brand-gold/15 rounded" />
                                <div className="h-3 w-36 bg-brand-text/15 rounded" />
                            </div>
                            <div className="h-3 w-16 bg-brand-gold/20 rounded pt-2" />
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    const COLS = 4; // matches lg:grid-cols-4

    // Group into 4-chapter sections (matching Kabanata navigation groups)
    const groups: { start: number; end: number; chapters: Chapter[] }[] = [];
    for (let i = 0; i < items.length; i += 4) {
        const slice = items.slice(i, i + 4);
        groups.push({
            start: slice[0].chapter_number,
            end: slice[slice.length - 1].chapter_number,
            chapters: slice,
        });
    }

    return (
        <div className="w-full pb-20 space-y-6">
            <motion.div
                key={selectedNovel}
                className="space-y-6"
                initial="hidden"
                animate="show"
            >
                {groups.map((group) => (
                    <section
                        key={`${selectedNovel}-group-${group.start}`}
                        id={`kabanata-${group.start}`}
                        data-group-start={group.start}
                        className="scroll-mt-[180px] lg:scroll-mt-[104px]"
                        style={{ scrollMarginTop: "var(--chapter-scroll-margin, 180px)" }}
                    >
                        <div className="grid gap-2.5 sm:gap-6 grid-cols-2 lg:grid-cols-4">
                            {group.chapters.map((chapter, idx) => (
                                <ChapterCard
                                    key={`${chapter.book}-${chapter.chapter_number}`}
                                    chapter={chapter}
                                    isGroupStart={chapter.chapter_number === group.start}
                                    colIndex={idx % COLS}
                                    onClick={() => onChapterSelect(chapter.book, chapter.chapter_number, chapter.chapter_title)}
                                />
                            ))}
                        </div>
                    </section>
                ))}
            </motion.div>
        </div>
    );
}

function ChapterCard({
    chapter,
    onClick,
    colIndex,
    isGroupStart,
}: {
    chapter: Chapter;
    onClick: () => void;
    colIndex: number;
    isGroupStart?: boolean;
}) {
    const isNoli = chapter.book.toLowerCase().includes("noli");

    return (
        <motion.div
            id={!isGroupStart ? `kabanata-${chapter.chapter_number}` : `kabanata-card-${chapter.chapter_number}`}
            role="button"
            tabIndex={0}
            onClick={onClick}
            onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onClick();
                }
            }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: "easeOut", delay: colIndex * 0.05 }}
            whileHover={{ y: -6, boxShadow: "0 20px 40px -12px rgba(0,0,0,0.12)" }}
            whileTap={{ scale: 0.98 }}
            className={cn(
                "group relative p-3.5 sm:p-6 min-h-[9.5rem] sm:h-56 flex flex-col justify-between overflow-hidden cursor-pointer select-none",
                "bg-white/75 dark:bg-brand-paper/85 backdrop-blur-md shadow-sm transition-all duration-300",
                "border border-brand-gold/15 hover:border-brand-gold/40 active:border-brand-gold/60 rounded-xl sm:rounded-sm",
                "scroll-mt-32 sm:scroll-mt-40 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-gold"
            )}
            aria-label={`Basahin ang Kabanata ${chapter.chapter_number}: ${chapter.chapter_title}`}
        >
            {/* Decorative Background Number */}
            <span className={cn(
                "absolute -right-2 -top-4 sm:-right-3 sm:-top-6 text-6xl sm:text-9xl font-serif font-black opacity-[0.03] select-none transition-opacity group-hover:opacity-[0.07]",
                "text-brand-navy dark:text-brand-gold pointer-events-none"
            )}>
                {chapter.chapter_number}
            </span>

            {/* Top Accent Line */}
            <div className={cn(
                "absolute top-0 left-0 w-full h-1 transform origin-left scale-x-0 group-hover:scale-x-100 transition-transform duration-500 ease-out",
                isNoli ? "bg-noli-accent" : "bg-fili-accent"
            )} />

            <div>
                <span className={cn(
                    "text-[9px] sm:text-[10px] font-bold tracking-[0.16em] sm:tracking-[0.2em] uppercase truncate block",
                    isNoli ? "text-noli-accent" : "text-fili-accent"
                )}>
                    {isNoli ? "Noli Me Tangere" : "El Filibusterismo"}
                </span>
                <h3 className="text-base sm:text-2xl font-serif text-brand-navy dark:text-foreground mt-1 sm:mt-3 group-hover:text-brand-gold transition-colors duration-300 leading-tight font-bold">
                    Kabanata {chapter.chapter_number}
                </h3>
                <p className="text-[11px] sm:text-sm font-serif text-brand-text/70 mt-1 sm:mt-2 italic line-clamp-1 sm:line-clamp-2">
                    {chapter.chapter_title}
                </p>
            </div>

            <div className="flex items-center text-[9px] sm:text-xs font-bold tracking-widest text-brand-text-light group-hover:text-brand-gold transition-colors duration-300 pt-1.5 sm:pt-2">
                <span className="mr-1 sm:mr-2 truncate">BASAHIN</span>
                <svg className="w-3.5 h-3.5 sm:w-4 sm:h-4 transform group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
            </div>
        </motion.div>
    );
}

