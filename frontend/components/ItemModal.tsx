"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { X, Quote } from "lucide-react";
import { CharacterAvatar } from "@/components/CharacterAvatar";

interface ChapterInfo {
    book: string;
    chapter_number: number;
    chapter_title: string;
    score: number;
    preview_text?: string;
    sentence_index?: number;
}

interface ThemeContext {
    book: string;
    chapter_number: number;
    chapter_title: string;
    sentence_text: string;
}

interface ItemModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    subtitle?: string;
    type: "character" | "theme";
    // For Characters
    // appearances?: Appearance[]; // Deprecated in favor of chapters
    chapterAppearances?: ChapterInfo[];
    // For Themes
    themeContext?: ThemeContext;
    meaning?: string;
    isLoading: boolean;
    onNavigate?: (book: string, chapter: number, sentenceIndex?: number) => void;
    onSort?: (mode: 'number' | 'relevance') => void;
    sortBy?: 'number' | 'relevance';
    selectedNovel: 'noli' | 'fili';
}

export function ItemModal({
    isOpen,
    onClose,
    title,
    subtitle,
    type,
    // appearances,
    chapterAppearances,
    themeContext,
    meaning,
    isLoading,
    onNavigate,
    onSort,
    sortBy,
    selectedNovel
}: ItemModalProps) {
    const modalRef = useRef<HTMLDivElement>(null);
    const [isAvatarZoomed, setIsAvatarZoomed] = useState(false);
    // Removed internal novelFilter state in favor of selectedNovel prop

    // Paksa themes for character modal
    interface PaksaTheme { label: string; description: string; }
    const [paksaThemes, setPaksaThemes] = useState<PaksaTheme[]>([]);
    const [paksaLoading, setPaksaLoading] = useState(false);
    const [charView, setCharView] = useState<'sentences' | 'paksa'>('sentences');

    // Reset view when opening or changing character
    useEffect(() => {
        if (isOpen) setCharView('sentences');
    }, [isOpen, title]);

    useEffect(() => {
        if (!isOpen || type !== 'character') return;
        const fetchPaksa = async () => {
            setPaksaLoading(true);
            setPaksaThemes([]);
            try {
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
                const bookParam = selectedNovel === 'noli' ? 'noli' : 'elfili';
                const res = await fetch(`${apiUrl}/api/v1/characters/${encodeURIComponent(title)}/paksa?book=${bookParam}`);
                if (res.ok) {
                    const data = await res.json();
                    setPaksaThemes(data.themes || []);
                }
            } catch (e) {}
            setPaksaLoading(false);
        };
        fetchPaksa();
    }, [isOpen, type, title, selectedNovel]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        };
        if (isOpen) {
            window.addEventListener("keydown", handleKeyDown);
            document.body.style.overflow = "hidden";
        }
        return () => {
            window.removeEventListener("keydown", handleKeyDown);
            document.body.style.overflow = "unset";
        };
    }, [isOpen, onClose]);

    const handleBackdropClick = (e: React.MouseEvent) => {
        if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
            onClose();
        }
    };

    const filteredChapters = chapterAppearances?.filter(c => {
        if (selectedNovel === 'noli') return c.book === 'noli';
        return c.book === 'fili' || c.book === 'elfili';
    });


    const renderChapter = (chapter: ChapterInfo, idx: number) => (
        <div
            key={`${chapter.book}-${chapter.chapter_number}-${idx}`}
            onClick={() => onNavigate?.(chapter.book, chapter.chapter_number, chapter.sentence_index)}
            className={`
                group p-6 rounded-sm border cursor-pointer transition-all hover:shadow-md bg-white
                ${chapter.book === 'noli' ? 'border-noli-accent/10 hover:border-noli-accent' : 'border-fili-accent/10 hover:border-fili-accent'}
            `}
        >
            <div className="flex items-center gap-2 mb-3">
                <span className={`text-[9px] font-bold px-2 py-0.5 rounded uppercase ${chapter.book === 'noli' ? 'bg-noli-accent/10 text-noli-accent' : 'bg-fili-accent/10 text-fili-accent'}`}>
                    {chapter.book === 'noli' ? 'Noli' : 'Fili'}
                </span>
                <span className="text-xl font-serif font-bold text-brand-navy">
                    Kabanata {chapter.chapter_number}
                </span>
            </div>

            <p className="text-sm text-brand-text-light font-serif line-clamp-2 mb-3 font-bold min-h-[2.5em]">
                {chapter.chapter_title}
            </p>

            {chapter.preview_text && (
                <p className="text-xs text-brand-text/60 italic line-clamp-4 border-t border-brand-gold/5 pt-3 mt-2 leading-relaxed h-[5.5em]">
                    "{chapter.preview_text}"
                </p>
            )}
        </div>
    );

    const getNovelLabel = (novel: 'noli' | 'fili') => {
        switch (novel) {
            case 'noli': return 'Noli Me Tangere';
            case 'fili': return 'El Filibusterismo';
        }
    };

    return (
        <AnimatePresence>
            {isOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6" key="modal-content">
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={handleBackdropClick}
                        className="absolute inset-0 bg-brand-navy/60 backdrop-blur-sm"
                    />

                    <motion.div
                        ref={modalRef}
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        transition={{ type: "spring", damping: 25, stiffness: 300 }}
                        className={`relative w-full ${type === 'character' ? 'max-w-6xl' : 'max-w-2xl'} max-h-[90vh] flex flex-col bg-brand-paper shadow-2xl rounded-lg overflow-hidden border border-brand-gold/20`}
                    >
                        {/* Header */}
                        <div className="flex items-center justify-between p-6 border-b border-brand-gold/10 bg-brand-cream flex-wrap gap-4">
                            <div className="flex items-center gap-6 flex-1 min-w-0">
                                {type === "character" && (
                                    <div className="hidden sm:block shrink-0">
                                        <CharacterAvatar
                                            name={title}
                                            size={100}
                                            className="shadow-md border-2 border-brand-gold/20"
                                            onClick={() => setIsAvatarZoomed(true)}
                                        />
                                    </div>
                                )}

                                <div className="min-w-0">
                                    <span className="text-[10px] font-bold tracking-[0.2em] uppercase text-brand-gold mb-1 block">
                                        {type === "character" ? "Profil ng Tauhan" : "Thematikong Insight"}
                                    </span>
                                    <h2 className="text-2xl md:text-4xl font-serif text-brand-navy font-bold truncate tracking-tight">
                                        {title}
                                    </h2>
                                    {subtitle && (
                                        <p className="text-brand-text/70 font-serif italic mt-1 text-lg">{subtitle}</p>
                                    )}
                                </div>
                            </div>

                            <div className="flex items-center gap-4 shrink-0">
                                {/* Static Novel Label */}
                                <div className={`
                                    px-3 py-1.5 rounded-full text-[9px] font-bold uppercase tracking-[0.15em]
                                    flex items-center gap-2 border shadow-sm whitespace-nowrap
                                    ${selectedNovel === 'noli' ? 'bg-noli-accent/10 text-noli-accent border-noli-accent/20' :
                                        'bg-fili-accent/10 text-fili-accent border-fili-accent/20'}
                                `}>
                                    <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
                                    {getNovelLabel(selectedNovel)}
                                </div>

                                {type === "character" && (
                                    <div className="flex bg-white/40 p-1 rounded-full border border-brand-gold/20 shadow-inner">
                                        <button
                                            onClick={() => setCharView('sentences')}
                                            className={`
                                                px-3 py-1.5 rounded-full text-[9px] font-bold uppercase tracking-widest transition-all
                                                ${charView === 'sentences' ? 'bg-brand-navy text-white shadow-sm' : 'text-brand-navy/60 hover:text-brand-navy'}
                                            `}
                                        >
                                            Mga Pangungusap
                                        </button>
                                        <button
                                            onClick={() => setCharView('paksa')}
                                            className={`
                                                px-3 py-1.5 rounded-full text-[9px] font-bold uppercase tracking-widest transition-all
                                                ${charView === 'paksa' ? 'bg-brand-navy text-white shadow-sm' : 'text-brand-navy/60 hover:text-brand-navy'}
                                            `}
                                        >
                                            Mga Paksa
                                        </button>
                                    </div>
                                )}

                                {type === "character" && onSort && charView === 'sentences' && (
                                    <button
                                        onClick={() => onSort(sortBy === 'relevance' ? 'number' : 'relevance')}
                                        className={`
                                            px-3 py-1.5 rounded-full text-[9px] font-bold uppercase tracking-widest transition-all whitespace-nowrap shadow-sm border
                                            ${sortBy === 'relevance'
                                                ? 'bg-brand-gold text-white border-brand-gold'
                                                : 'bg-white text-brand-text border-brand-gold/20 hover:border-brand-gold'}
                                        `}
                                    >
                                        {sortBy === 'relevance' ? 'NAKA-RANK' : 'I-RANK'}
                                    </button>
                                )}
                                <button
                                    onClick={onClose}
                                    className="p-2 hover:bg-black/5 rounded-full transition-colors text-brand-text/60 hover:text-brand-navy"
                                >
                                    <X size={24} />
                                </button>
                            </div>
                        </div>

                        {/* Content */}
                        <div className="flex-1 overflow-y-auto p-6 bg-white/50 scrollbar-thin scrollbar-thumb-brand-gold/20 scrollbar-track-transparent">
                            {isLoading ? (
                                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 animate-pulse">
                                    {[...Array(8)].map((_, i) => (
                                        <div key={i} className="h-40 bg-gray-200 rounded-lg" />
                                    ))}
                                </div>
                            ) : (
                                <>
                                    {/* Theme View */}
                                    {type === "theme" && (
                                        <div className="space-y-6">
                                            <div className="bg-brand-cream/50 p-4 rounded-lg border border-brand-brown/5">
                                                <h4 className="font-bold text-brand-brown mb-2 uppercase tracking-wide text-xs">Definition</h4>
                                                <p className="font-serif text-brand-text leading-relaxed">
                                                    {meaning}
                                                </p>
                                            </div>

                                            {themeContext ? (
                                                (
                                                    (selectedNovel === 'noli' && themeContext.book === 'noli') ||
                                                    (selectedNovel === 'fili' && (themeContext.book === 'fili' || themeContext.book === 'elfili'))) ? (
                                                    <div>
                                                        <h4 className="font-bold text-brand-brown mb-4 uppercase tracking-wide text-xs flex items-center gap-2">
                                                            <Quote size={14} /> Best Contextual Match
                                                        </h4>
                                                        <div
                                                            onClick={() => onNavigate?.(themeContext.book, themeContext.chapter_number, undefined)}
                                                            className="bg-white p-6 rounded-lg border-l-4 border-brand-gold shadow-sm cursor-pointer hover:bg-brand-gold/5 transition-colors"
                                                        >
                                                            <p className="font-serif text-lg text-brand-navy italic mb-4 leading-relaxed">
                                                                "{themeContext.sentence_text}"
                                                            </p>
                                                            <div className="flex items-center justify-between text-xs text-brand-text-light font-sans border-t border-gray-100 pt-3">
                                                                <span className="font-bold uppercase tracking-wider">
                                                                    {themeContext.book === 'noli' ? 'Noli Me Tangere' : 'El Filibusterismo'}
                                                                </span>
                                                                <span className="group-hover:text-brand-gold transition-colors">
                                                                    Chapter {themeContext.chapter_number}: {themeContext.chapter_title} (Click to Read)
                                                                </span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                ) : (
                                                    <p className="text-brand-text-light italic">
                                                        Best match is in {themeContext.book === 'noli' ? 'Noli Me Tangere' : 'El Filibusterismo'}.
                                                        Switch story selection on main page to view.
                                                    </p>
                                                )
                                            ) : (
                                                <p className="text-brand-text-light italic">No context available.</p>
                                            )}
                                        </div>
                                    )}

                                    {/* Character View */}
                                    {type === "character" && (
                                        <div className="space-y-6">
                                            {charView === 'sentences' ? (
                                                <>
                                                    <div className="flex items-center justify-between">
                                                        <h4 className="font-bold text-brand-brown uppercase tracking-wide text-xs">
                                                            {`Mga Kabanata (${filteredChapters?.length || 0})`}
                                                        </h4>
                                                    </div>

                                                    {chapterAppearances && chapterAppearances.length > 0 ? (
                                                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                                                            {filteredChapters?.map((c, i) => renderChapter(c, i))}
                                                        </div>
                                                    ) : (
                                                        <p className="text-brand-text-light italic text-center py-10 col-span-4">
                                                            Walang nakitang kabanata para sa tauhang ito.
                                                        </p>
                                                    )}
                                                </>
                                            ) : (
                                                <>
                                                    <div className="flex items-center justify-between">
                                                        <h4 className="font-bold text-brand-brown uppercase tracking-wide text-xs">
                                                            Mga Paksa ng Tauhan
                                                        </h4>
                                                    </div>

                                                    {paksaLoading ? (
                                                        <div className="flex flex-col items-center justify-center py-20 gap-4">
                                                            <div className="w-8 h-8 border-4 border-brand-gold border-t-transparent rounded-full animate-spin" />
                                                            <span className="text-brand-text/50 font-serif italic">Naghahanap ng mga paksa...</span>
                                                        </div>
                                                    ) : paksaThemes.length > 0 ? (
                                                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
                                                            {paksaThemes.map((theme, i) => (
                                                                <div
                                                                    key={i}
                                                                    className={`
                                                                        p-6 rounded-sm border bg-white transition-all shadow-sm
                                                                        ${selectedNovel === 'noli' ? 'border-noli-accent/10' : 'border-fili-accent/10'}
                                                                    `}
                                                                >
                                                                    <div className="flex items-center gap-2 mb-4">
                                                                        <span className={`text-[9px] font-bold px-2 py-0.5 rounded uppercase ${selectedNovel === 'noli' ? 'bg-noli-accent/10 text-noli-accent' : 'bg-fili-accent/10 text-fili-accent'}`}>
                                                                            Paksa
                                                                        </span>
                                                                        <h3 className="text-xl font-serif font-bold text-brand-navy">
                                                                            {theme.label}
                                                                        </h3>
                                                                    </div>
                                                                    {theme.description && (
                                                                        <p className="text-sm text-brand-text/70 leading-relaxed font-serif italic border-t border-brand-gold/5 pt-4">
                                                                            “{theme.description}”
                                                                        </p>
                                                                    )}
                                                                </div>
                                                            ))}
                                                        </div>
                                                    ) : (
                                                        <p className="text-brand-text-light italic text-center py-10">
                                                            Walang nakitang paksa para sa tauhang ito.
                                                        </p>
                                                    )}
                                                </>
                                            )}
                                        </div>
                                    )}

                                </>
                            )}
                        </div>
                    </motion.div>
                </div>
            )}

            {/* Avatar Lightbox Overlay */}
            {isAvatarZoomed && (
                <motion.div
                    key="lightbox-overlay"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-[60] bg-black/90 flex items-center justify-center p-4 cursor-pointer"
                    onClick={() => setIsAvatarZoomed(false)}
                >
                    <motion.div
                        initial={{ scale: 0.8, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.8, opacity: 0 }}
                        transition={{ type: "spring", damping: 25, stiffness: 300 }}
                        className="relative"
                        onClick={(e) => e.stopPropagation()} // Prevent closing when clicking the image itself
                    >
                        <CharacterAvatar
                            name={title}
                            size={400}
                            className="shadow-2xl border-4 border-brand-gold/50"
                            priority={true}
                        />
                        <button
                            onClick={() => setIsAvatarZoomed(false)}
                            className="absolute -top-12 right-0 text-white/50 hover:text-white transition-colors"
                        >
                            <X size={32} />
                        </button>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
}
