"use client";

import { useState, useEffect } from "react";

import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

type Novel = "noli" | "fili" | "both";

interface Chapter {
    book: string;
    chapter_number: number;
    chapter_title: string;
}

interface ChapterGridProps {
    selectedNovel: Novel;
    onChapterSelect: (book: string, chapter: number, title: string) => void;
}

interface ChapterContent {
    sentence_index: number;
    sentence_text: string;
}

export function ChapterGrid({ selectedNovel, onChapterSelect }: ChapterGridProps) {
    const [chapters, setChapters] = useState<Chapter[]>([]);
    const [loading, setLoading] = useState(true);


    useEffect(() => {
        async function fetchChapters() {
            try {
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
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
        if (selectedNovel === "fili") return fili;

        // Interleave for "Both"
        const combined: Chapter[] = [];
        const maxLen = Math.max(noli.length, fili.length);

        for (let i = 0; i < maxLen; i += 2) {
            if (i < noli.length) combined.push(noli[i]);
            if (i + 1 < noli.length) combined.push(noli[i + 1]);
            if (i < fili.length) combined.push(fili[i]);
            if (i + 1 < fili.length) combined.push(fili[i + 1]);
        }
        return combined;
    };

    const items = getFilteredChapters();

    if (loading) {
        return <div className="text-center py-20 text-brand-gold font-serif animate-pulse">Loading Chronicles...</div>;
    }

    return (
        <div className="w-full max-w-7xl mx-auto px-4 pb-20">
            <motion.div
                layout
                className="grid gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4"
            >
                <AnimatePresence>
                    {items.map((chapter) => (
                        <ChapterCard
                            key={`${chapter.book}-${chapter.chapter_number}`}
                            chapter={chapter}
                            onClick={() => onChapterSelect(chapter.book, chapter.chapter_number, chapter.chapter_title)}
                        />
                    ))}
                </AnimatePresence>
            </motion.div>
        </div>
    );
}

function ChapterCard({ chapter, onClick }: { chapter: Chapter; onClick: () => void }) {
    const isNoli = chapter.book === "noli";

    return (
        <motion.div
            layout
            onClick={onClick}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            whileHover={{ y: -8 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
            className={cn(
                "group relative p-6 h-56 flex flex-col justify-between overflow-hidden cursor-pointer",
                "bg-brand-paper shadow-sm hover:shadow-xl transition-all duration-300",
                "border border-transparent",
                isNoli ? "hover:border-noli-accent/20" : "hover:border-fili-accent/20"
            )}
        >
            {/* Decorative Background Number */}
            <span className={cn(
                "absolute -right-4 -top-6 text-9xl font-serif font-black opacity-[0.03] select-none transition-opacity group-hover:opacity-[0.07]",
                isNoli ? "text-noli-accent" : "text-fili-accent"
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
                    Chapter {chapter.chapter_number}
                </h3>
                <p className="text-sm font-serif text-brand-text/60 mt-2 italic line-clamp-2">
                    {chapter.chapter_title}
                </p>
            </div>

            <div className="flex items-center text-xs font-bold tracking-widest text-brand-text-light group-hover:text-brand-gold transition-colors duration-300">
                <span className="mr-2">READ CHAPTER</span>
                <svg className="w-4 h-4 transform group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
            </div>
        </motion.div>
    );
}

