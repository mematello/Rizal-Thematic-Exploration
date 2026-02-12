
"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef } from "react";
import { X, BookOpen, Quote } from "lucide-react";

interface Appearance {
    book: string;
    chapter_number: number;
    chapter_title: string;
    sentence_text: string;
    sentence_index: number;
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
    appearances?: Appearance[];
    // For Themes
    themeContext?: ThemeContext;
    meaning?: string;
    isLoading: boolean;
    onNavigate?: (book: string, chapter: number, sentenceIndex?: number) => void;
}

export function ItemModal({
    isOpen,
    onClose,
    title,
    subtitle,
    type,
    appearances,
    themeContext,
    meaning,
    isLoading,
    onNavigate,
}: ItemModalProps) {
    const modalRef = useRef<HTMLDivElement>(null);

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

    return (
        <AnimatePresence>
            {isOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
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
                        className="relative w-full max-w-2xl max-h-[85vh] flex flex-col bg-brand-paper shadow-2xl rounded-lg overflow-hidden border border-brand-gold/20"
                    >
                        {/* Header */}
                        <div className="flex items-start justify-between p-6 border-b border-brand-gold/10 bg-brand-cream">
                            <div>
                                <span className="text-xs font-bold tracking-[0.2em] uppercase text-brand-gold mb-1 block">
                                    {type === "character" ? "Character Profile" : "Thematic Insight"}
                                </span>
                                <h2 className="text-2xl md:text-3xl font-serif text-brand-navy">
                                    {title}
                                </h2>
                                {subtitle && (
                                    <p className="text-brand-text/70 font-serif italic mt-1">{subtitle}</p>
                                )}
                            </div>
                            <button
                                onClick={onClose}
                                className="p-2 hover:bg-black/5 rounded-full transition-colors text-brand-text/60 hover:text-brand-navy"
                            >
                                <X size={24} />
                            </button>
                        </div>

                        {/* Content */}
                        <div className="flex-1 overflow-y-auto p-6 bg-white/50 scrollbar-thin scrollbar-thumb-brand-gold/20 scrollbar-track-transparent">
                            {isLoading ? (
                                <div className="space-y-4 animate-pulse">
                                    <div className="h-4 bg-gray-200 rounded w-3/4" />
                                    <div className="h-10 bg-gray-200 rounded w-full" />
                                    <div className="h-4 bg-gray-200 rounded w-1/2" />
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
                                                <p className="text-brand-text-light italic">No context available.</p>
                                            )}
                                        </div>
                                    )}

                                    {/* Character View */}
                                    {type === "character" && appearances && (
                                        <div className="space-y-6">
                                            <div className="flex items-center justify-between">
                                                <h4 className="font-bold text-brand-brown uppercase tracking-wide text-xs">
                                                    Notable Appearances ({appearances.length})
                                                </h4>
                                            </div>

                                            <div className="space-y-4">
                                                {appearances.length > 0 ? (
                                                    appearances.map((app, idx) => (
                                                        <div
                                                            key={idx}
                                                            onClick={() => {
                                                                console.log('Character appearance clicked:', app.book, app.chapter_number, app.sentence_index);
                                                                onNavigate?.(app.book, app.chapter_number, app.sentence_index);
                                                            }}
                                                            className="group hover:bg-white transition-colors p-4 rounded-lg border border-transparent hover:border-brand-gold/10 cursor-pointer"
                                                        >
                                                            <div className="flex flex-col gap-2">
                                                                <div className="flex items-center gap-2 mb-1">
                                                                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase ${app.book === 'noli' ? 'bg-noli-gold/20 text-noli-gold' : 'bg-fili-magenta/20 text-fili-magenta'}`}>
                                                                        {app.book === 'noli' ? 'Noli' : 'Fili'}
                                                                    </span>
                                                                    <span className="text-xs text-brand-text-light font-bold">
                                                                        Chapter {app.chapter_number}
                                                                    </span>
                                                                </div>
                                                                <p className="font-serif text-brand-text text-sm leading-relaxed group-hover:text-brand-navy transition-colors">
                                                                    "{app.sentence_text}"
                                                                </p>
                                                            </div>
                                                        </div>
                                                    ))
                                                ) : (
                                                    <p className="text-brand-text-light italic text-center py-10">
                                                        No specific sentences found for this character name.
                                                    </p>
                                                )}
                                            </div>
                                        </div>
                                    )}
                                </>
                            )}
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>
    );
}
