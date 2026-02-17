"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { X, Maximize2, Minimize2, ChevronLeft, ChevronRight, Users, BookOpen } from "lucide-react";
import { CHARACTERS } from "@/lib/characterData";

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
    onNavigate?: (book: string, chapter: number) => void; // Add navigation callback
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

    // We don't need themeKeywords anymore as we use backend data
    // const [themeKeywords, setThemeKeywords] = useState<string[]>([]);

    const [showReference, setShowReference] = useState(false);
    const [themeFullscreen, setThemeFullscreen] = useState(false);
    const [selectedSentenceForTheme, setSelectedSentenceForTheme] = useState<number | null>(null);

    // Fetch characters for this chapter
    useEffect(() => {
        if (isOpen && book && chapterNumber) {
            // Comprehensive character name variations (Tagalog-friendly)
            // Comprehensive character name variations derived from shared data
            const allKeywords = CHARACTERS.flatMap(c => [c.name, ...(c.aliases || [])]);
            // Filter unique and sort by length descending for greedy matching
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

    const handlePrevChapter = () => {
        if (chapterNumber > 1 && onNavigate) {
            onNavigate(book, chapterNumber - 1);
        }
    };

    const handleNextChapter = () => {
        if (onNavigate) {
            onNavigate(book, chapterNumber + 1);
        }
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

        // Create regex pattern with STRICT WORD BOUNDARIES
        // This prevents "Bali" from matching inside "baliw" or "balita"
        const pattern = keywords.map(k => {
            const escaped = k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            return `\\b${escaped}\\b`; // \b ensures complete word match only
        }).join('|');
        const regex = new RegExp(`(${pattern})`, 'gi');

        const parts = text.split(regex);

        return parts.map((part, index) => {
            const isKeyword = keywords.some(k => k.toLowerCase() === part.toLowerCase());
            if (isKeyword) {
                return (
                    <mark key={index} className="bg-yellow-200 font-semibold px-1 rounded">
                        {part}
                    </mark>
                );
            }
            return part;
        });
    };

    const selectedSentence = content.find(s => s.sentence_index === selectedSentenceForTheme);

    return (
        <AnimatePresence>
            {isOpen && (
                <div className={`fixed inset-0 z-50 flex items-center justify-center ${isFullscreen ? 'p-0' : 'p-4 sm:p-6'}`}>
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
                        <div className={`flex items-start justify-between p-4 md:p-6 border-b ${isNoli ? 'border-noli-gold/30' : 'border-fili-magenta/30'} bg-brand-cream`}>
                            <div className="flex-1">
                                <span className={`text-xs font-bold tracking-[0.2em] uppercase ${accentColor}`}>
                                    {isNoli ? "Noli Me Tangere" : "El Filibusterismo"}
                                </span>
                                <h2 className="text-xl md:text-2xl font-serif text-brand-navy mt-1">
                                    Chapter {chapterNumber}
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
                        <div className="flex gap-2 p-4 bg-brand-cream/50 border-b border-brand-gold/20">
                            <button
                                onClick={() => setShowCharacters(!showCharacters)}
                                className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all ${showCharacters ? 'bg-brand-gold text-white' : 'bg-white/50 text-brand-text hover:bg-brand-gold/20'}`}
                            >
                                <Users size={16} />
                                <span className="text-sm font-medium">Characters</span>
                            </button>
                            <button
                                onClick={() => setShowThemes(!showThemes)}
                                className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all ${showThemes ? 'bg-brand-gold text-white' : 'bg-white/50 text-brand-text hover:bg-brand-gold/20'}`}
                            >
                                <BookOpen size={16} />
                                <span className="text-sm font-medium">Themes</span>
                            </button>
                            <button
                                onClick={() => setShowReference(!showReference)}
                                className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all ${showReference ? 'bg-brand-gold text-white' : 'bg-white/50 text-brand-text hover:bg-brand-gold/20'}`}
                            >
                                <BookOpen size={16} />
                                <span className="text-sm font-medium">Reference</span>
                            </button>
                        </div>

                        {/* Content */}
                        <div className="flex-1 overflow-y-auto p-6 md:p-10 bg-brand-paper scrollbar-thin scrollbar-thumb-brand-gold/30 scrollbar-track-transparent">
                            {isLoading ? (
                                <div className="space-y-4 animate-pulse">
                                    {[...Array(8)].map((_, i) => (
                                        <div key={i} className="h-4 bg-brand-gold/10 rounded w-full last:w-3/4" />
                                    ))}
                                </div>
                            ) : (
                                <div className="max-w-3xl mx-auto">
                                    <p className="font-serif text-brand-text leading-loose text-justify indent-8 text-base md:text-lg">
                                        {content.map((sentence) => {
                                            const isHighlighted = sentence.sentence_index === highlightSentenceIndex;

                                            // Ensure themes is an array (backward compatibility)
                                            const themes = sentence.themes || [];
                                            const themeCount = themes.length;

                                            return (
                                                <span
                                                    key={sentence.sentence_index}
                                                    ref={isHighlighted ? highlightRef : null}
                                                    className={`
                                                        hover:bg-brand-gold/10 transition-all duration-200 rounded px-0.5 relative inline
                                                        ${isHighlighted ? 'bg-yellow-200 font-bold shadow-sm border-b-2 border-brand-gold' : ''}
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
                                                                w-[12px] h-[12px] rounded-full shadow-sm
                                                                cursor-pointer transition-transform mr-0.5 flex-shrink-0
                                                                ${themeCount === 1
                                                                    ? 'bg-emerald-500 hover:bg-emerald-600'
                                                                    : 'bg-rose-500 hover:bg-rose-600'}
                                                                text-white text-[8px] font-sans font-bold select-none leading-none z-10 p-0 indent-0
                                                                transform -translate-y-[6px] align-middle
                                                            `}
                                                            title={`${themeCount} theme${themeCount > 1 ? 's' : ''} in this sentence`}
                                                        >
                                                            {themeCount}
                                                        </span>
                                                    )}

                                                    {highlightText(sentence.sentence_text)}

                                                    {/* Reference number (Wikipedia style) - all showing [0] for now */}
                                                    {showReference && (
                                                        <sup className="text-blue-600 cursor-help ml-0.5 font-bold text-xs">
                                                            [0]
                                                        </sup>
                                                    )}
                                                    {" "}
                                                </span>
                                            );
                                        })}
                                    </p>
                                </div>
                            )}
                        </div>

                        {/* Navigation Footer */}
                        <div className="flex items-center justify-between p-4 border-t border-brand-gold/20 bg-brand-cream">
                            <button
                                onClick={handlePrevChapter}
                                disabled={chapterNumber <= 1}
                                className="flex items-center gap-2 px-4 py-2 rounded-md bg-white/50 text-brand-text hover:bg-brand-gold/20 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                            >
                                <ChevronLeft size={18} />
                                <span className="text-sm font-medium">Previous</span>
                            </button>
                            <div className="text-xs text-brand-text-light font-sans uppercase tracking-widest">
                                Rizal Thematic Exploration
                            </div>
                            <button
                                onClick={handleNextChapter}
                                className="flex items-center gap-2 px-4 py-2 rounded-md bg-white/50 text-brand-text hover:bg-brand-gold/20 transition-all"
                            >
                                <span className="text-sm font-medium">Next</span>
                                <ChevronRight size={18} />
                            </button>
                        </div>
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
    );
}
