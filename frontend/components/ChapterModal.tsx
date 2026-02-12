"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef } from "react";
import { X } from "lucide-react";

interface ChapterContent {
    sentence_index: number;
    sentence_text: string;
}

interface ChapterModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    chapterNumber: number;
    book: string;
    content: ChapterContent[];
    isLoading: boolean;
    highlightSentenceIndex?: number; // Optional: sentence to highlight
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
}: ChapterModalProps) {
    const modalRef = useRef<HTMLDivElement>(null);
    const highlightRef = useRef<HTMLSpanElement>(null);

    // Close on escape key
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        };
        if (isOpen) {
            window.addEventListener("keydown", handleKeyDown);
            document.body.style.overflow = "hidden"; // Prevent background scrolling
        }
        return () => {
            window.removeEventListener("keydown", handleKeyDown);
            document.body.style.overflow = "unset";
        };
    }, [isOpen, onClose]);

    // Scroll to highlighted sentence
    useEffect(() => {
        if (isOpen && !isLoading && highlightSentenceIndex !== undefined && highlightRef.current) {
            // Small delay to ensure DOM is ready
            setTimeout(() => {
                highlightRef.current?.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center',
                });
            }, 300);
        }
    }, [isOpen, isLoading, highlightSentenceIndex]);

    // Debug logging
    useEffect(() => {
        if (content.length > 0) {
            console.log('ChapterModal content loaded. Total sentences:', content.length);
            console.log('Highlight target:', highlightSentenceIndex);
            console.log('First 3 sentence indices:', content.slice(0, 3).map(s => s.sentence_index));
        }
    }, [content, highlightSentenceIndex]);

    // Handle click outside
    const handleBackdropClick = (e: React.MouseEvent) => {
        if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
            onClose();
        }
    };

    const isNoli = book === "noli";
    const accentColor = isNoli ? "text-noli-accent" : "text-fili-accent";
    const borderColor = isNoli ? "border-noli-accent" : "border-fili-accent";

    return (
        <AnimatePresence>
            {isOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={handleBackdropClick}
                        className="absolute inset-0 bg-brand-navy/60 backdrop-blur-sm"
                    />

                    {/* Modal Container */}
                    <motion.div
                        ref={modalRef}
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        transition={{ type: "spring", damping: 25, stiffness: 300 }}
                        className="relative w-full max-w-3xl max-h-[85vh] flex flex-col bg-brand-paper shadow-2xl rounded-lg overflow-hidden"
                    >
                        {/* Header */}
                        <div className={`flex items-start justify-between p-6 border-b ${isNoli ? 'border-noli-gold/30' : 'border-fili-magenta/30'} bg-brand-cream`}>
                            <div>
                                <span className={`text-xs font-bold tracking-[0.2em] uppercase ${accentColor}`}>
                                    {isNoli ? "Noli Me Tangere" : "El Filibusterismo"}
                                </span>
                                <h2 className="text-2xl md:text-3xl font-serif text-brand-navy mt-1">
                                    Chapter {chapterNumber}
                                </h2>
                                <p className="text-brand-text/70 font-serif italic mt-1">
                                    {title}
                                </p>
                            </div>
                            <button
                                onClick={onClose}
                                className="p-2 hover:bg-black/5 rounded-full transition-colors text-brand-text/60 hover:text-brand-navy"
                            >
                                <X size={24} />
                            </button>
                        </div>

                        {/* Content */}
                        <div className="flex-1 overflow-y-auto p-6 md:p-8 bg-white/50 scrollbar-thin scrollbar-thumb-brand-gold/20 scrollbar-track-transparent">
                            {isLoading ? (
                                <div className="space-y-4 animate-pulse">
                                    {[...Array(6)].map((_, i) => (
                                        <div key={i} className="h-4 bg-gray-200 rounded w-full last:w-3/4" />
                                    ))}
                                </div>
                            ) : (
                                <div className="prose prose-lg max-w-none font-serif text-brand-text leading-relaxed">
                                    {content.map((sentence) => {
                                        const isHighlighted = sentence.sentence_index === highlightSentenceIndex;
                                        if (isHighlighted) {
                                            console.log('Highlighting sentence index:', sentence.sentence_index, 'matches:', highlightSentenceIndex);
                                        }
                                        return (
                                            <span
                                                key={sentence.sentence_index}
                                                ref={isHighlighted ? highlightRef : null}
                                                className={`
                                                    hover:bg-brand-gold/10 transition-all duration-300 rounded px-1
                                                    ${isHighlighted ? 'bg-yellow-300 font-extrabold shadow-md border-2 border-brand-gold scale-[1.02] inline-block z-10 text-brand-navy px-2 transform transition-transform' : ''}
                                                `}
                                            >
                                                {sentence.sentence_text}{" "}
                                            </span>
                                        );
                                    })}
                                </div>
                            )}
                        </div>

                        {/* Footer */}
                        <div className="p-4 border-t border-brand-gold/10 bg-brand-cream text-center text-xs text-brand-text-light font-sans uppercase tracking-widest">
                            Rizal Thematic Exploration
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>
    );
}
