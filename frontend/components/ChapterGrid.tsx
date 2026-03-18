"use client";

import { useState, useEffect } from "react";

import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

type Novel = "noli" | "fili";

interface Chapter {
    book: string;
    chapter_number: number;
    chapter_title: string;
}

interface ChapterGridProps {
    selectedNovel: Novel;
    onChapterSelect: (book: string, chapter: number, title: string) => void;
    highlightChapterNumber?: number;
    highlightQuery?: string;
}

interface ChapterContent {
    sentence_index: number;
    sentence_text: string;
}

export function ChapterGrid({ selectedNovel, onChapterSelect, highlightChapterNumber, highlightQuery }: ChapterGridProps) {
    const [chapters, setChapters] = useState<Chapter[]>([]);
    const [loading, setLoading] = useState(true);


    useEffect(() => {
        async function fetchChapters() {
            try {
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
                const res = await fetch(`${apiUrl}/api/v1/chapters`);
                if (!res.ok) throw new Error("Failed to fetch");
                const data = await res.json();
                setChapters(data);
            } catch (error) {
                console.error("Error fetching chapters:", error);
            } finally {
                setLoading(false);
            }
        }
        fetchChapters();
    }, []);


    const getFilteredChapters = () => {
        const noli = chapters.filter(c => c.book === "noli").sort((a, b) => a.chapter_number - b.chapter_number);
        // DB uses 'elfili', frontend uses 'fili' convention in toggle
        const fili = chapters.filter(c => c.book === "elfili").sort((a, b) => a.chapter_number - b.chapter_number);

        if (selectedNovel === "noli") return noli;
        return fili;
    };

    const items = getFilteredChapters();

    if (loading) {
        return <div className="text-center py-20 text-brand-gold font-serif animate-pulse">Nagkakargang mga Tala...</div>;
    }

    const COLS = 4; // matches lg:grid-cols-4

    return (
        <div className="w-full pb-20">
            <motion.div
                key={selectedNovel}
                className="grid gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4"
                initial="hidden"
                animate="show"
            >
                {items.map((chapter, idx) => (
                    <ChapterCard
                        key={`${chapter.book}-${chapter.chapter_number}`}
                        chapter={chapter}
                        colIndex={idx % COLS}
                        onClick={() => onChapterSelect(chapter.book, chapter.chapter_number, chapter.chapter_title)}
                        isHighlighted={chapter.chapter_number === highlightChapterNumber}
                        highlightQuery={highlightQuery}
                    />
                ))}
            </motion.div>
        </div>
    );
}

function ChapterCard({
    chapter,
    onClick,
    colIndex,
    isHighlighted,
    highlightQuery
}: {
    chapter: Chapter;
    onClick: () => void;
    colIndex: number;
    isHighlighted?: boolean;
    highlightQuery?: string;
}) {
    const isNoli = chapter.book === "noli";
    const normalize = (s: string) => s.toLowerCase().replace(/\s+/g, " ").trim();
    const isTitleExactMatch =
        highlightQuery && normalize(highlightQuery) === normalize(chapter.chapter_title);

    return (
        <motion.div
            id={`kabanata-${chapter.chapter_number}`}
            onClick={onClick}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: "easeOut", delay: colIndex * 0.1 }}
            whileHover={{ y: -8, boxShadow: "0 25px 50px -12px rgba(0,0,0,0.15)" }}
            whileTap={{ scale: 0.99 }}
            className={cn(
                "group relative p-6 h-56 flex flex-col justify-between overflow-hidden cursor-pointer",
                "bg-white/70 backdrop-blur-md shadow-sm transition-all duration-500",
                "border border-brand-gold/10 hover:border-brand-gold/40 rounded-sm",
                "scroll-mt-40",
                isHighlighted && [
                    "ring-2 ring-offset-2 ring-offset-transparent",
                    isNoli ? "ring-noli-accent/60" : "ring-fili-accent/60",
                    "shadow-lg shadow-black/10"
                ]
            )}
        >
            {/* Spotlight highlight */}
            {isHighlighted && (
                <motion.div
                    aria-hidden="true"
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.25, ease: "easeOut" }}
                    className={cn(
                        "pointer-events-none absolute inset-0",
                        "bg-gradient-to-br from-white/35 via-transparent to-transparent"
                    )}
                />
            )}

            {/* Decorative Background Number */}
            <span className={cn(
                "absolute -right-4 -top-6 text-9xl font-serif font-black opacity-[0.03] select-none transition-opacity group-hover:opacity-[0.07]",
                "text-brand-navy",
                isHighlighted && "opacity-[0.06]"
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
                    "text-[10px] font-bold tracking-[0.2em] uppercase",
                    isNoli ? "text-noli-accent" : "text-fili-accent"
                )}>
                    {isNoli ? "Noli Me Tangere" : "El Filibusterismo"}
                </span>
                <h3 className="text-2xl font-serif text-brand-navy mt-3 group-hover:text-brand-gold transition-colors duration-300 leading-tight">
                    Kabanata {chapter.chapter_number}
                </h3>
                <p className="text-sm font-serif text-brand-text/60 mt-2 italic line-clamp-2">
                    {isTitleExactMatch ? (
                        <span className={cn(
                            "inline-flex items-center rounded-sm px-1.5 py-0.5",
                            isNoli ? "bg-noli-accent/15 text-brand-navy" : "bg-fili-accent/15 text-brand-navy"
                        )}>
                            {chapter.chapter_title}
                        </span>
                    ) : (
                        chapter.chapter_title
                    )}
                </p>
            </div>

            <div className="flex items-center text-xs font-bold tracking-widest text-brand-text-light group-hover:text-brand-gold transition-colors duration-300">
                <span className="mr-2">BASAHIN ANG KABANATA</span>
                <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
            </div>
        </motion.div>
    );
}

