"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { X, Maximize2, Minimize2, ChevronLeft, ChevronRight, Users, BookOpen } from "lucide-react";
import { CHARACTERS, Character } from "@/lib/characterData";
import { ItemModal } from "@/components/ItemModal";

interface ThemeMatch {
    id: string;
    label: string;
    score: number;
    explanation: string;
}

interface ChapterContent {
    sentence_index: number;
    sentence_text: string;
    themes: ThemeMatch[];
}

interface ChapterModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    chapterNumber: number;
    book: string;
    content: ChapterContent[];
    isLoading: boolean;
    highlightSentenceIndex?: number;
    onNavigate?: (book: string, chapter: number) => void;
}

interface LoadedChapter {
    chapterNumber: number;
    title: string;
    content: ChapterContent[];
    isLoading: boolean;
}

export function ChapterModal({
    isOpen,
    onClose,
    title,
    chapterNumber,
    book,
    content,
    isLoading,
    highlightSentenceIndex,
    onNavigate,
}: ChapterModalProps) {
    const modalRef = useRef<HTMLDivElement>(null);
    const highlightRef = useRef<HTMLSpanElement>(null);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [showCharacters, setShowCharacters] = useState(false);
    const [showThemes, setShowThemes] = useState(false);
    const [characterKeywords, setCharacterKeywords] = useState<string[]>([]);

    // Inline character modal state
    const [selectedChar, setSelectedChar] = useState<Character | null>(null);
    const [charAppearances, setCharAppearances] = useState<{
        book: string; chapter_number: number; chapter_title: string;
        score: number; preview_text?: string; sentence_index?: number;
    }[]>([]);
    const [charLoading, setCharLoading] = useState(false);
    const [isCharModalOpen, setIsCharModalOpen] = useState(false);
    const [charSortBy, setCharSortBy] = useState<'number' | 'relevance'>('number');

    // We don't need themeKeywords anymore as we use backend data
    // const [themeKeywords, setThemeKeywords] = useState<string[]>([]);

    const [showReference, setShowReference] = useState(false);
    const [themeFullscreen, setThemeFullscreen] = useState(false);
    const [selectedSentenceForTheme, setSelectedSentenceForTheme] = useState<number | null>(null);

    // Fullscreen multi-chapter state
    const [loadedChapters, setLoadedChapters] = useState<LoadedChapter[]>([]);
    const [chapterMetadata, setChapterMetadata] = useState<Record<number, string>>({});
    const [isLoadingMore, setIsLoadingMore] = useState(false);
    const bottomSentinelRef = useRef<HTMLDivElement>(null);
    const topSentinelRef = useRef<HTMLDivElement>(null);

    // Fetch metadata once
    useEffect(() => {
        if (isOpen) {
            const fetchMetadata = async () => {
                try {
                    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
                    const res = await fetch(`${apiUrl}/api/v1/chapters`);
                    if (res.ok) {
                        const data = await res.json();
                        const map: Record<number, string> = {};
                        data.filter((c: any) => (book === 'noli' ? c.book === 'noli' : c.book === 'elfili'))
                            .forEach((c: any) => {
                                map[c.chapter_number] = c.chapter_title;
                            });
                        setChapterMetadata(map);
                    }
                } catch (err) {
                    console.error("Error fetching metadata:", err);
                }
            };
            fetchMetadata();
        }
    }, [isOpen, book]);

    // Initialize loadedChapters when entering fullscreen or when initial content changes
    useEffect(() => {
        if (isFullscreen) {
            setLoadedChapters([{
                chapterNumber,
                title,
                content: content || [],
                isLoading: false
            }]);
        }
    }, [isFullscreen, chapterNumber, title, content]);

    // Fetch characters for this chapter
    useEffect(() => {
        if (isOpen && book && chapterNumber) {
            const allKeywords = CHARACTERS.flatMap(c => [c.name, ...(c.aliases || [])]);
            const uniqueKeywords = Array.from(new Set(allKeywords)).sort((a, b) => b.length - a.length);
            setCharacterKeywords(uniqueKeywords);
        }
    }, [isOpen, book, chapterNumber]);

    // Close on escape key
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                if (isFullscreen) {
                    setIsFullscreen(false);
                } else {
                    onClose();
                }
            }
        };
        if (isOpen) {
            window.addEventListener("keydown", handleKeyDown);
            document.body.style.overflow = "hidden";
        }
        return () => {
            window.removeEventListener("keydown", handleKeyDown);
            document.body.style.overflow = "unset";
        };
    }, [isOpen, isFullscreen, onClose]);

    // Scroll to highlighted sentence
    useEffect(() => {
        if (isOpen && !isLoading && highlightSentenceIndex !== undefined && highlightRef.current) {
            setTimeout(() => {
                highlightRef.current?.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center',
                });
            }, 300);
        }
    }, [isOpen, isLoading, highlightSentenceIndex]);

    // Handle click outside (only when not fullscreen)
    const handleBackdropClick = (e: React.MouseEvent) => {
        if (!isFullscreen && modalRef.current && !modalRef.current.contains(e.target as Node)) {
            onClose();
        }
    };

    const isNoli = book === "noli";
    const accentColor = isNoli ? "text-noli-accent" : "text-fili-accent";
    const borderColor = isNoli ? "border-noli-accent" : "border-fili-accent";

    // Standard chapter counts
    const NOLI_CHAPTERS = 64;
    const FILI_CHAPTERS = 39;

    const handlePrevChapter = () => {
        if (!onNavigate) return;

        const maxChapters = isNoli ? NOLI_CHAPTERS : FILI_CHAPTERS;

        if (chapterNumber <= 1) {
            // Loop to last chapter
            onNavigate(book, maxChapters);
        } else {
            onNavigate(book, chapterNumber - 1);
        }
    };

    const handleNextChapter = () => {
        if (!onNavigate) return;

        const maxChapters = isNoli ? NOLI_CHAPTERS : FILI_CHAPTERS;

        if (chapterNumber >= maxChapters) {
            // Loop to first chapter
            onNavigate(book, 1);
        } else {
            onNavigate(book, chapterNumber + 1);
        }
    };

    const fetchSpecificChapter = async (num: number): Promise<ChapterContent[]> => {
        try {
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
            const res = await fetch(`${apiUrl}/api/v1/chapters/${book}/${num}`);
            if (!res.ok) throw new Error("Failed");
            return await res.json();
        } catch (err) {
            console.error(err);
            return [];
        }
    };

    const loadNextChapter = async () => {
        if (isLoadingMore || loadedChapters.length === 0) return;
        const lastChapter = loadedChapters[loadedChapters.length - 1];
        const nextNum = lastChapter.chapterNumber + 1;
        const maxChapters = isNoli ? NOLI_CHAPTERS : FILI_CHAPTERS;

        if (nextNum > maxChapters) return;

        setIsLoadingMore(true);
        const content = await fetchSpecificChapter(nextNum);
        setLoadedChapters(prev => [...prev, {
            chapterNumber: nextNum,
            title: chapterMetadata[nextNum] || `Kabanata ${nextNum}`,
            content,
            isLoading: false
        }]);
        setIsLoadingMore(false);
    };

    const loadPrevChapter = async () => {
        if (isLoadingMore || loadedChapters.length === 0) return;
        const firstChapter = loadedChapters[0];
        const prevNum = firstChapter.chapterNumber - 1;

        if (prevNum < 1) return;

        setIsLoadingMore(true);
        const content = await fetchSpecificChapter(prevNum);
        setLoadedChapters(prev => [{
            chapterNumber: prevNum,
            title: chapterMetadata[prevNum] || `Kabanata ${prevNum}`,
            content,
            isLoading: false
        }, ...prev]);
        setIsLoadingMore(false);
    };

    // Intersection Observer for infinite scroll
    useEffect(() => {
        if (!isFullscreen) return;

        const bottomObserver = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) {
                loadNextChapter();
            }
        }, { threshold: 0.1 });

        const topObserver = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) {
                loadPrevChapter();
            }
        }, { threshold: 0.1 });

        if (bottomSentinelRef.current) {
            bottomObserver.observe(bottomSentinelRef.current);
        }
        if (topSentinelRef.current) {
            topObserver.observe(topSentinelRef.current);
        }

        return () => {
            bottomObserver.disconnect();
            topObserver.disconnect();
        };
    }, [isFullscreen, loadedChapters, isLoadingMore]);

    // ---------- Inline character modal helpers ----------
    const fetchCharAppearances = async (char: Character, sort: 'number' | 'relevance') => {
        setCharLoading(true);
        try {
            const searchTerms = [char.name, ...(char.aliases || [])];
            const searchTerm = searchTerms.join(",");
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
            const res = await fetch(
                `${apiUrl}/api/v1/characters/chapters?name=${encodeURIComponent(searchTerm)}&sort_by=${sort}`
            );
            if (!res.ok) throw new Error("Failed to fetch chapters");
            const data = await res.json();
            setCharAppearances(data);
        } catch (err) {
            console.error("Error fetching character chapters:", err);
        } finally {
            setCharLoading(false);
        }
    };

    const handleCharMarkClick = async (clickedText: string) => {
        // Find the Character whose name or alias matches the clicked text
        const match = CHARACTERS.find(c =>
            c.name.toLowerCase() === clickedText.toLowerCase() ||
            (c.aliases || []).some(a => a.toLowerCase() === clickedText.toLowerCase())
        );
        if (!match) return;
        setSelectedChar(match);
        setIsCharModalOpen(true);
        setCharAppearances([]);
        setCharSortBy('number');
        await fetchCharAppearances(match, 'number');
    };

    const handleCharModalClose = () => {
        setIsCharModalOpen(false);
        setTimeout(() => {
            setSelectedChar(null);
            setCharAppearances([]);
        }, 300);
    };

    // Helper function to highlight keywords in text
    const highlightText = (text: string) => {
        // Only highlight characters
        if (!showCharacters) {
            return text;
        }

        const keywords = characterKeywords;

        if (keywords.length === 0) {
            return text;
        }

        // Build a regex that works for:
        //   - Multi-word names  ("Maria Clara", "Padre Damaso")
        //   - Hyphenated names  ("Ben-Zayb", "Ka-Tales")
        //   - Single words      ("Ibarra", "Simoun")
        //
        // Strategy: use a negative lookbehind for word-char at the start and
        // a negative lookahead for word-char at the end.  This means the
        // matched name must not be immediately preceded or followed by a
        // letter, digit, or underscore – i.e. it must stand alone as a token.
        const pattern = keywords
            .map(k => {
                const escaped = k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                return `(?<![\\wÀ-ÖØ-öø-ÿ])${escaped}(?![\\wÀ-ÖØ-öø-ÿ])`;
            })
            .join('|');

        const regex = new RegExp(`(${pattern})`, 'gi');

        const parts = text.split(regex);

        return parts.map((part, index) => {
            const isKeyword = keywords.some(k => k.toLowerCase() === part.toLowerCase());
            if (isKeyword) {
                return (
                    <mark
                        key={index}
                        onClick={(e) => { e.stopPropagation(); handleCharMarkClick(part); }}
                        className="bg-yellow-200 font-semibold px-1 rounded cursor-pointer hover:bg-yellow-300 hover:shadow-sm transition-all underline decoration-dotted decoration-brand-navy/40"
                        title={`View character: ${part}`}
                    >
                        {part}
                    </mark>
                );
            }
            return part;
        });
    };

    const selectedSentence = content.find(s => s.sentence_index === selectedSentenceForTheme);

    return (
        <>
            <AnimatePresence>
                {isOpen && (
                    <div key="chapter-modal" className={`fixed inset-0 z-50 flex items-center justify-center ${isFullscreen ? 'p-0' : 'p-4 sm:p-6'}`}>
                        {/* Backdrop */}
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={handleBackdropClick}
                            className="absolute inset-0 bg-brand-navy/70 backdrop-blur-sm"
                        />

                        {/* Modal Container */}
                        <motion.div
                            ref={modalRef}
                            initial={{ opacity: 0, scale: 0.95, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 20 }}
                            transition={{ type: "spring", damping: 25, stiffness: 300 }}
                            className={`relative ${isFullscreen ? 'w-full h-full' : 'w-full max-w-4xl max-h-[90vh]'} flex flex-col bg-brand-paper shadow-2xl ${isFullscreen ? 'rounded-none' : 'rounded-lg'} overflow-hidden`}
                        >
                            {/* Header */}
                            <div className={`flex items-start justify-between p-4 md:p-6 border-b border-brand-gold/10 bg-brand-cream`}>
                                <div className="flex-1">
                                    <span className={`text-[10px] font-bold tracking-[0.2em] uppercase ${accentColor}`}>
                                        {isNoli ? "Noli Me Tangere" : "El Filibusterismo"}
                                    </span>
                                    <h2 className="text-xl md:text-2xl font-serif text-brand-navy mt-1 font-bold">
                                        Kabanata {chapterNumber}
                                    </h2>
                                    <p className="text-brand-text/70 font-serif italic mt-1 text-sm">
                                        {title}
                                    </p>
                                </div>
                                <div className="flex items-center gap-2">
                                    <button
                                        onClick={() => setIsFullscreen(!isFullscreen)}
                                        className="p-2 hover:bg-black/5 rounded-full transition-colors text-brand-text/60 hover:text-brand-navy"
                                        title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
                                    >
                                        {isFullscreen ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
                                    </button>
                                    <button
                                        onClick={onClose}
                                        className="p-2 hover:bg-black/5 rounded-full transition-colors text-brand-text/60 hover:text-brand-navy"
                                    >
                                        <X size={24} />
                                    </button>
                                </div>
                            </div>

                            {/* Action Buttons */}
                            <div className="flex gap-2 p-4 bg-brand-cream/50 border-b border-brand-gold/10 overflow-x-auto no-scrollbar">
                                <button
                                    onClick={() => setShowCharacters(!showCharacters)}
                                    className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all whitespace-nowrap ${showCharacters ? 'bg-brand-gold text-white shadow-sm' : 'bg-white/50 text-brand-text hover:bg-brand-gold/20'}`}
                                >
                                    <Users size={16} />
                                    <span className="text-sm font-bold uppercase tracking-widest">Tauhan</span>
                                </button>
                                <button
                                    onClick={() => setShowThemes(!showThemes)}
                                    className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all whitespace-nowrap ${showThemes ? 'bg-brand-gold text-white shadow-sm' : 'bg-white/50 text-brand-text hover:bg-brand-gold/20'}`}
                                >
                                    <BookOpen size={16} />
                                    <span className="text-sm font-bold uppercase tracking-widest">Paksa</span>
                                </button>
                                <button
                                    onClick={() => setShowReference(!showReference)}
                                    className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all whitespace-nowrap ${showReference ? 'bg-brand-gold text-white shadow-sm' : 'bg-white/50 text-brand-text hover:bg-brand-gold/20'}`}
                                >
                                    <BookOpen size={16} />
                                    <span className="text-sm font-bold uppercase tracking-widest">Sanggunian</span>
                                </button>
                            </div>

                            {/* Content */}
                            <div className={`flex-1 overflow-y-auto bg-brand-paper scrollbar-thin scrollbar-thumb-brand-gold/30 scrollbar-track-transparent ${isFullscreen ? 'p-0' : 'p-6 md:p-14'}`}>
                                {isLoading && !isFullscreen ? (
                                    <div className="space-y-4 animate-pulse p-6 md:p-14">
                                        {[...Array(8)].map((_, i) => (
                                            <div key={i} className="h-4 bg-brand-gold/10 rounded w-full last:w-3/4" />
                                        ))}
                                    </div>
                                ) : (
                                    <div className={`${isFullscreen ? 'max-w-4xl mx-auto px-6 py-20' : 'max-w-3xl mx-auto'}`}>
                                        {/* Top Sentinel for Intersection Observer */}
                                        {isFullscreen && (
                                            <div ref={topSentinelRef} className="h-20 flex items-center justify-center">
                                                {isLoadingMore && (
                                                    <div className="animate-pulse text-brand-gold font-serif italic text-lg">
                                                        Kinakarga ang nakaraang kabanata...
                                                    </div>
                                                )}
                                            </div>
                                        )}

                                        {(isFullscreen ? loadedChapters : [{ chapterNumber, title, content, isLoading }]).map((chap, chapIdx) => (
                                            <motion.div
                                                key={`${chap.chapterNumber}-${chapIdx}`}
                                                initial={isFullscreen ? { opacity: 0, y: 30 } : {}}
                                                animate={isFullscreen ? { opacity: 1, y: 0 } : {}}
                                                transition={{ duration: 1, ease: "easeOut" }}
                                                className={isFullscreen ? "mb-32" : ""}
                                            >
                                                {/* Fullscreen Header (Title Page Style) */}
                                                {isFullscreen && (
                                                    <div className="flex flex-col items-center text-center mb-16 pt-10 border-b border-brand-gold/5 pb-16">
                                                        <span className={`text-xs font-bold tracking-[0.4em] uppercase mb-6 ${accentColor}`}>
                                                            {book === "noli" ? "Noli Me Tangere" : "El Filibusterismo"}
                                                        </span>
                                                        <h1 className="text-5xl md:text-7xl font-serif text-brand-navy font-black mb-6">
                                                            Kabanata {chap.chapterNumber}
                                                        </h1>
                                                        <p className="text-2xl md:text-3xl font-serif italic text-brand-text/60 max-w-2xl leading-relaxed">
                                                            {chap.title}
                                                        </p>
                                                        <div className="w-24 h-px bg-brand-gold/30 mt-12" />
                                                    </div>
                                                )}

                                                <div className="font-serif text-brand-text leading-[2.2] text-justify space-y-8">
                                                    {chap.content.reduce((paragraphs: ChapterContent[][], sentence) => {
                                                        // Simple paragraph grouping logic (every 5-7 sentences or based on content)
                                                        // For now, let's keep it as is but style it as a block
                                                        if (paragraphs.length === 0 || paragraphs[paragraphs.length - 1].length >= 6) {
                                                            paragraphs.push([sentence]);
                                                        } else {
                                                            paragraphs[paragraphs.length - 1].push(sentence);
                                                        }
                                                        return paragraphs;
                                                    }, []).map((para, paraIdx) => (
                                                        <p key={paraIdx} className="text-lg md:text-2xl first-letter:text-4xl first-letter:font-serif first-letter:mr-1 first-letter:float-left first-letter:leading-none indent-8">
                                                            {para.map((sentence) => {
                                                                const isHighlighted = !isFullscreen && sentence.sentence_index === highlightSentenceIndex;
                                                                const themes = sentence.themes || [];
                                                                const themeCount = themes.length;

                                                                return (
                                                                    <span
                                                                        key={sentence.sentence_index}
                                                                        ref={isHighlighted ? highlightRef : null}
                                                                        className={`
                                                                        hover:bg-brand-gold/10 transition-all duration-200 rounded px-0.5 relative inline
                                                                        ${isHighlighted ? 'bg-brand-gold/20 font-bold border-b-2 border-brand-gold' : ''}
                                                                    `}
                                                                    >
                                                                        {showThemes && themeCount > 0 && (
                                                                            <span
                                                                                onClick={(e) => {
                                                                                    e.stopPropagation();
                                                                                    setSelectedSentenceForTheme(sentence.sentence_index);
                                                                                    setThemeFullscreen(true);
                                                                                }}
                                                                                className={`
                                                                                inline-flex items-center justify-center 
                                                                                w-[14px] h-[14px] rounded-full shadow-sm
                                                                                cursor-pointer transition-transform mr-1 flex-shrink-0
                                                                                ${themeCount === 1
                                                                                        ? 'bg-brand-gold hover:bg-brand-gold/80'
                                                                                        : 'bg-brand-navy hover:bg-brand-navy/80'}
                                                                                text-white text-[8px] font-sans font-bold select-none leading-none z-10 p-0 indent-0
                                                                                transform -translate-y-[8px] align-middle
                                                                            `}
                                                                                title={`${themeCount} paksa sa pangungusap na ito`}
                                                                            >
                                                                                {themeCount}
                                                                            </span>
                                                                        )}

                                                                        {highlightText(sentence.sentence_text)}

                                                                        {showReference && (
                                                                            <sup className="text-brand-gold cursor-help ml-0.5 font-bold text-xs">
                                                                                [0]
                                                                            </sup>
                                                                        )}
                                                                        {" "}
                                                                    </span>
                                                                );
                                                            })}
                                                        </p>
                                                    ))}
                                                </div>

                                                {/* Soft Divider between chapters in fullscreen */}
                                                {isFullscreen && chapIdx < loadedChapters.length - 1 && (
                                                    <div className="flex items-center justify-center py-24">
                                                        <div className="flex gap-4">
                                                            <div className="w-1.5 h-1.5 rounded-full bg-brand-gold/20" />
                                                            <div className="w-1.5 h-1.5 rounded-full bg-brand-gold/40" />
                                                            <div className="w-1.5 h-1.5 rounded-full bg-brand-gold/20" />
                                                        </div>
                                                    </div>
                                                )}
                                            </motion.div>
                                        ))}

                                        {/* Bottom Sentinel for Intersection Observer */}
                                        {isFullscreen && (
                                            <div ref={bottomSentinelRef} className="h-20 flex items-center justify-center">
                                                {isLoadingMore && (
                                                    <div className="animate-pulse text-brand-gold font-serif italic text-lg">
                                                        Dinatlo ang susunod na kabanata...
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>

                            {/* Navigation Footer */}
                            {!isFullscreen && (
                                <div className="flex items-center justify-between p-4 border-t border-brand-gold/10 bg-brand-cream">
                                    <button
                                        onClick={handlePrevChapter}
                                        className="flex items-center gap-2 px-4 py-2 rounded-md bg-white text-brand-text hover:bg-brand-gold/10 transition-all border border-brand-gold/10"
                                    >
                                        <ChevronLeft size={18} />
                                        <span className="text-xs font-bold uppercase tracking-widest">Nakaraan</span>
                                    </button>
                                    <div className="hidden sm:block text-[10px] text-brand-accent/60 font-serif uppercase tracking-[0.3em] font-bold">
                                        Leksikal at Semantikong Pag-aaral
                                    </div>
                                    <button
                                        onClick={handleNextChapter}
                                        className="flex items-center gap-2 px-4 py-2 rounded-md bg-white text-brand-text hover:bg-brand-gold/10 transition-all border border-brand-gold/10"
                                    >
                                        <span className="text-xs font-bold uppercase tracking-widest">Susunod</span>
                                        <ChevronRight size={18} />
                                    </button>
                                </div>
                            )}
                        </motion.div>

                        {/* Fullscreen Theme Explanation View */}
                        {themeFullscreen && selectedSentence && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="fixed inset-0 z-[60] bg-white flex flex-col"
                            >
                                {/* Header */}
                                <div className="border-b border-brand-gold/30 bg-brand-cream/50 py-6">
                                    <div className="max-w-7xl mx-auto px-6">
                                        <div className="text-center space-y-2">
                                            <h2 className="text-2xl font-serif text-brand-navy font-bold">
                                                {book === "noli" ? "Noli Me Tangere" : "El Filibusterismo"}
                                            </h2>
                                            <p className="text-lg text-brand-text">
                                                Chapter {chapterNumber}
                                            </p>
                                            <p className="text-base text-brand-text-light italic">
                                                {title}
                                            </p>

                                            {/* Action Buttons */}
                                            <div className="flex items-center justify-center gap-3 pt-4">
                                                <button
                                                    onClick={() => setShowCharacters(!showCharacters)}
                                                    className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all text-sm ${showCharacters ? 'bg-brand-gold text-white' : 'bg-white text-brand-text hover:bg-brand-gold/20'}`}
                                                >
                                                    <Users size={14} />
                                                    Characters
                                                </button>
                                                <button
                                                    onClick={() => setShowThemes(!showThemes)}
                                                    className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all text-sm ${showThemes ? 'bg-brand-gold text-white' : 'bg-white text-brand-text hover:bg-brand-gold/20'}`}
                                                >
                                                    <BookOpen size={14} />
                                                    Themes
                                                </button>
                                            </div>
                                        </div>

                                        {/* Close Button */}
                                        <button
                                            onClick={() => {
                                                setThemeFullscreen(false);
                                                setSelectedSentenceForTheme(null);
                                            }}
                                            className="absolute top-6 right-6 p-2 hover:bg-black/5 rounded-full transition-colors"
                                        >
                                            <X size={24} className="text-brand-navy" />
                                        </button>
                                    </div>
                                </div>

                                {/* Content Area */}
                                <div className="flex-1 overflow-hidden flex">
                                    {/* Left Side - Chapter Text */}
                                    <div className="flex-1 overflow-y-auto p-8">
                                        <div className="max-w-3xl mx-auto">
                                            <p className="font-serif text-brand-text leading-loose text-justify text-base">
                                                {selectedSentence.sentence_text}
                                            </p>
                                        </div>
                                    </div>

                                    {/* Right Side - Theme Explanations */}
                                    <div className="w-96 border-l border-brand-gold/30 bg-brand-paper overflow-y-auto p-6">
                                        <h3 className="text-lg font-serif font-bold text-brand-navy mb-4">
                                            Themes in this Sentence
                                        </h3>

                                        {/* Theme Cards (Real Data) */}
                                        <div className="space-y-4">
                                            {selectedSentence.themes && selectedSentence.themes.length > 0 ? (
                                                selectedSentence.themes.map((theme, idx) => (
                                                    <div key={idx} className="bg-white p-4 rounded-lg border border-brand-gold/20 shadow-sm hover:shadow-md transition-shadow">
                                                        <div className="flex items-start gap-3">
                                                            <div className={`
                                                            w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm flex-shrink-0
                                                            ${idx === 0 ? 'bg-gradient-to-br from-purple-500 to-pink-500' :
                                                                    idx === 1 ? 'bg-gradient-to-br from-blue-500 to-cyan-500' :
                                                                        'bg-gradient-to-br from-amber-500 to-orange-500'}
                                                        `}>
                                                                {idx + 1}
                                                            </div>
                                                            <div>
                                                                <h4 className="font-serif font-bold text-brand-navy mb-1">
                                                                    {theme.label}
                                                                </h4>
                                                                <p className="text-sm text-brand-text-light leading-relaxed">
                                                                    {theme.explanation}
                                                                </p>
                                                                <div className="mt-2 text-xs text-brand-text-light/70 font-mono">
                                                                    Relevance Score: {Math.round(theme.score * 100)}%
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                ))
                                            ) : (
                                                <div className="p-4 text-center text-brand-text-light italic">
                                                    No themes found for this sentence.
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                    </div>
                )}
            </AnimatePresence>

            {/* Inline Character Profile Modal – opens when a highlighted name is clicked */}
            {selectedChar && (
                <ItemModal
                    isOpen={isCharModalOpen}
                    onClose={handleCharModalClose}
                    title={selectedChar.name}
                    subtitle={selectedChar.role}
                    type="character"
                    chapterAppearances={charAppearances}
                    isLoading={charLoading}
                    selectedNovel={book === 'noli' ? 'noli' : 'fili'}
                    onNavigate={(navBook, navChapter) => {
                        handleCharModalClose();
                        onNavigate?.(navBook, navChapter);
                    }}
                    onSort={async (mode) => {
                        if (selectedChar) {
                            setCharSortBy(mode);
                            await fetchCharAppearances(selectedChar, mode);
                        }
                    }}
                    sortBy={charSortBy}
                />
            )}
        </>
    );
}
