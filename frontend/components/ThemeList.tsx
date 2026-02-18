"use client";


import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, BookOpen } from "lucide-react";
import { cn } from "@/lib/utils";
import { ItemModal } from "@/components/ItemModal";

interface ThemeContext {
    book: string;
    chapter_number: number;
    chapter_title: string;
    sentence_text: string;
    sentence_index: number;
}

interface Theme {
    book: string; // Deprecated in UI but still in response structure if needed, or we ignore
    tagalog_title: string;
    meaning: string;
    best_match: ThemeContext | null;
}


interface ThemeListProps {
    onChapterSelect?: (book: string, chapter: number, title?: string, sentenceIndex?: number) => void;
    selectedNovel: "noli" | "fili" | "both";
}

export function ThemeList({ onChapterSelect, selectedNovel }: ThemeListProps) {
    const [themes, setThemes] = useState<Theme[]>([]);
    const [loading, setLoading] = useState(true);

    // Modal
    const [selectedTheme, setSelectedTheme] = useState<Theme | null>(null);
    const [isModalOpen, setIsModalOpen] = useState(false);

    useEffect(() => {
        async function fetchThemes() {
            try {
                const res = await fetch("http://localhost:8000/api/v1/themes");
                if (!res.ok) throw new Error("Failed to fetch");
                const data = await res.json();
                setThemes(data);
            } catch (error) {
                console.error("Error fetching themes:", error);
            } finally {
                setLoading(false);
            }
        }
        fetchThemes();
    }, []);

    const handleThemeClick = (theme: Theme) => {
        setSelectedTheme(theme);
        setIsModalOpen(true);
    };

    const handleClose = () => {
        setIsModalOpen(false);
        setTimeout(() => setSelectedTheme(null), 300);
    };

    if (loading) {
        return <div className="text-center py-20 text-brand-gold font-serif animate-pulse">Loading Themes...</div>;
    }

    return (
        <div className="max-w-7xl mx-auto px-4 pb-20">
            {selectedNovel === 'both' ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Noli Side */}
                    <div className="space-y-4">
                        <motion.div
                            className="grid grid-cols-1 sm:grid-cols-2 gap-6"
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
                                .filter(t => !t.best_match || t.best_match.book === 'noli')
                                .map((theme, idx) => (
                                    <ThemeCard key={`noli-${idx}`} theme={theme} onClick={() => handleThemeClick(theme)} />
                                ))}
                        </motion.div>
                    </div>

                    {/* Fili Side */}
                    <div className="space-y-4">
                        <motion.div
                            className="grid grid-cols-1 sm:grid-cols-2 gap-6"
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
                                .filter(t => t.best_match?.book === 'elfili' || t.best_match?.book === 'fili')
                                .map((theme, idx) => (
                                    <ThemeCard key={`fili-${idx}`} theme={theme} onClick={() => handleThemeClick(theme)} />
                                ))}
                        </motion.div>
                    </div>
                </div>
            ) : (
                <motion.div
                    key={selectedNovel}
                    className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6"
                    initial="hidden"
                    animate="show"
                    variants={{
                        hidden: { opacity: 0 },
                        show: {
                            opacity: 1,
                            transition: {
                                staggerChildren: 0.1
                            }
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
            )}

            {selectedTheme && (
                <ItemModal
                    isOpen={isModalOpen}
                    onClose={handleClose}
                    title={selectedTheme.tagalog_title}
                    type="theme"
                    meaning={selectedTheme.meaning}
                    themeContext={selectedTheme.best_match || undefined}
                    isLoading={false} // Data is already loaded
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
    // Determine novel based on best_match if available, otherwise default or mixed
    const isNoli = theme.best_match?.book === "noli";

    return (
        <motion.div
            variants={{
                hidden: { opacity: 0, y: 20 },
                show: { opacity: 1, y: 0 }
            }}
            whileHover={{ y: -5, boxShadow: "0 20px 40px -15px rgba(0,0,0,0.1)" }}
            onClick={onClick}
            className="bg-brand-paper p-8 rounded-sm relative overflow-hidden group border border-transparent hover:border-brand-gold/20 transition-all duration-300 cursor-pointer"
        >
            {/* Background Icon */}
            <div className="absolute -right-6 -bottom-6 opacity-[0.03] group-hover:opacity-[0.08] transition-opacity duration-500 transform group-hover:rotate-12 group-hover:scale-110">
                <BookOpen size={140} className="text-brand-navy" />
            </div>

            <div className="relative z-10">
                <div className="flex items-center space-x-2 mb-4">
                    <span className={cn(
                        "text-[10px] font-bold tracking-[0.2em] uppercase px-2 py-1 rounded-sm bg-opacity-5", // Reduced opacity
                        isNoli
                            ? "bg-brand-navy text-brand-navy"
                            : "bg-brand-gold text-brand-gold"
                    )}>
                        Theme
                    </span>
                </div>

                <h3 className="text-2xl font-serif text-brand-navy font-bold leading-tight mb-3 group-hover:text-brand-gold transition-colors duration-300">
                    {theme.tagalog_title}
                </h3>

                <div className="w-12 h-1 bg-brand-gold/30 mb-4 group-hover:w-full transition-all duration-500 ease-out" />

                <p className="text-brand-text-light font-body leading-relaxed text-sm line-clamp-3">
                    {theme.meaning}
                </p>

                <div className="mt-4 text-xs font-bold text-brand-navy/40 uppercase tracking-widest group-hover:text-brand-navy transition-colors">
                    Click to view context
                </div>
            </div>
        </motion.div>
    );
}
