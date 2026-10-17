"use client";


import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, BookOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import { ItemModal } from "@/components/ItemModal";

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
    return (
        <motion.div
            variants={{
                hidden: { opacity: 0, y: 20 },
                show: { opacity: 1, y: 0 }
            }}
            whileHover={{ y: -5, boxShadow: "0 20px 40px -15px rgba(0,0,0,0.1)" }}
            onClick={onClick}
            className="@container bg-brand-paper dark:bg-[#18202C] p-4 sm:p-6 lg:p-8 rounded-sm relative overflow-hidden group border border-brand-gold/10 dark:border-brand-gold/20 hover:border-brand-gold/40 transition-all duration-300 cursor-pointer"
        >
            {/* Background Icon */}
            <div className="absolute -right-6 -bottom-6 opacity-[0.03] group-hover:opacity-[0.08] transition-opacity duration-500 transform group-hover:rotate-12 group-hover:scale-110">
                <BookOpen size={140} className="text-brand-navy dark:text-brand-gold" />
            </div>

            <div className="relative z-10">
                <div className="flex items-center space-x-2 mb-3 sm:mb-4">
                    <span className={cn(
                        "text-[10px] font-bold tracking-[0.2em] uppercase px-2 py-1 rounded-sm",
                        "bg-brand-gold/10 text-brand-gold"
                    )}>
                        Paksa
                    </span>
                </div>

                <h3 className="font-serif text-brand-navy dark:text-foreground font-bold leading-tight mb-3 group-hover:text-brand-gold transition-colors duration-300 whitespace-nowrap text-[clamp(0.75rem,4.2cqi,1.5rem)] truncate">
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
