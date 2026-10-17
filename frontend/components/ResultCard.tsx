"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, BookOpen, ExternalLink } from "lucide-react";
import { ResultCardProps } from "../types";
import { ScoreVisualizer } from "./ScoreVisualizer";
import { motion, AnimatePresence } from "framer-motion";
import { formatChapterTitle, cleanChapterTitle } from "@/lib/utils";

interface ResultCardExtendedProps extends ResultCardProps {
    showScores?: boolean;
    showNovelBadge?: boolean;
    onChapterOpen?: (book: string, chapter: number, sentenceIndex: number, chapterTitle?: string) => void;
}

export function ResultCard({
    id,
    novel,
    chapter,
    chapterTitle,
    passageHtml,
    contextHtml,
    scores,
    confidenceBadge = false,
    themes = [],
    showScores = false,
    showNovelBadge = false,
    onChapterOpen,
    sentenceIndex: sentenceIndexProp,
}: ResultCardExtendedProps) {
    const [isExpanded, setIsExpanded] = useState(false);

    const novelTheme = {
        noli: {
            border: 'border-l-noli-gold',
            innerBorder: 'border-noli-gold',
            novelBadge: 'bg-noli-gold/15 text-noli-gold dark:bg-noli-gold/25',
            novelName: 'Noli Me Tangere',
            btnBorder: 'border-noli-gold/40 hover:border-noli-gold text-noli-gold hover:bg-noli-gold/5',
            book: 'noli',
        },
        fili: {
            border: 'border-l-fili-magenta',
            innerBorder: 'border-fili-magenta',
            novelBadge: 'bg-fili-magenta/15 text-fili-magenta dark:bg-fili-magenta/25',
            novelName: 'El Filibusterismo',
            btnBorder: 'border-fili-magenta/40 hover:border-fili-magenta text-fili-magenta hover:bg-fili-magenta/5',
            book: 'elfili',
        },
    }[novel];

    const hasThemes = themes.length > 0;
    const sentenceIndex = sentenceIndexProp ?? 0;
    const cleanedTitle = cleanChapterTitle(chapterTitle, chapter);

    return (
        <article
            id={id}
            className={`bg-brand-paper rounded-sm border border-brand-gold/20 border-l-4 ${novelTheme.border} hover:shadow-md transition-all duration-200 scroll-mt-20 overflow-hidden`}
            aria-labelledby={`result-title-${id}`}
        >
            {/* Card Header — clickable to open chapter */}
            <div
                className="p-4 sm:p-5 cursor-pointer group"
                onClick={() => onChapterOpen?.(novelTheme.book, chapter, sentenceIndex, cleanedTitle || "")}
                title="Buksan ang kabanata"
            >
                {/* Line 1: Kabanata N + Optional Novel Badge (if novel=both) + Confidence Badge */}
                <div className="flex items-center justify-between gap-2 mb-1">
                    <div className="flex items-center gap-2 flex-wrap min-w-0">
                        <span className="font-serif font-bold text-xs uppercase tracking-wider text-brand-navy/70 dark:text-brand-gold">
                            Kabanata {chapter}
                        </span>
                        {showNovelBadge && (
                            <span className={`text-[9px] font-sans font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${novelTheme.novelBadge}`}>
                                {novelTheme.novelName}
                            </span>
                        )}
                    </div>
                    {confidenceBadge && (
                        <span className="shrink-0 text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wide bg-brand-gold/10 text-brand-gold border border-brand-gold/20">
                            Mataas
                        </span>
                    )}
                </div>

                {/* Line 2: Chapter Title (hidden if empty or identical to Kabanata N) */}
                {cleanedTitle ? (
                    <h3
                        id={`result-title-${id}`}
                        className="font-serif font-bold text-brand-navy dark:text-foreground text-sm sm:text-base leading-snug group-hover:text-brand-gold transition-colors flex items-center gap-1.5 mb-2.5 break-words [overflow-wrap:anywhere]"
                    >
                        <span>{cleanedTitle}</span>
                        <ExternalLink size={13} className="opacity-0 group-hover:opacity-60 transition-opacity shrink-0 inline" />
                    </h3>
                ) : null}

                {/* Passage */}
                <div
                    className="font-serif text-sm leading-relaxed text-brand-text/80 break-words [overflow-wrap:anywhere]"
                    dangerouslySetInnerHTML={{ __html: passageHtml }}
                    aria-label="Sipi mula sa nobela"
                />
            </div>

            {/* Action Row */}
            <div className="px-4 sm:px-5 py-2 sm:py-2.5 border-t border-brand-gold/10 flex items-center gap-2 bg-brand-cream/30">
                <button
                    onClick={(e) => { 
                        e.stopPropagation(); 
                        setIsExpanded(!isExpanded); 
                    }}
                    className={`min-h-[44px] flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider rounded-full px-3.5 py-2 border transition-all active:scale-95 ${novelTheme.btnBorder}`}
                    aria-expanded={isExpanded}
                >
                    <BookOpen size={13} />
                    <span>{isExpanded ? 'Itago' : 'Ipakita ang Konteksto'}</span>
                    {isExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                </button>
            </div>

            {/* Score Visualizer (controlled by parent toggle) */}
            <AnimatePresence>
                {showScores && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        className="overflow-hidden"
                    >
                        <div className="px-4 sm:px-5 pb-3 pt-2 border-t border-brand-gold/10 bg-brand-cream/20">
                            <ScoreVisualizer 
                                semantic={Math.round(scores.semantic)} 
                                lexical={Math.round(scores.lexical)} 
                            />
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Context + Meaning Expansion */}
            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.25 }}
                        className="overflow-hidden"
                    >
                        {/* Context */}
                        <div
                            className="mx-4 sm:mx-5 mt-3 p-3.5 sm:p-4 rounded-sm bg-brand-cream border border-brand-gold/10 font-serif text-brand-text/70 text-sm italic leading-relaxed break-words [overflow-wrap:anywhere]"
                            dangerouslySetInnerHTML={{ __html: contextHtml }}
                            aria-label="Konteksto ng sipi"
                        />

                        {/* Meaning section (if themes exist) */}
                        {hasThemes && (
                            <div className="mx-4 sm:mx-5 mt-3 space-y-2">
                                <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-brand-text/40 block">Kahulugan ng Tema</span>
                                {themes.map((t) => (
                                    <div key={t.id} className={`p-3 rounded-sm border-l-4 ${novelTheme.innerBorder} bg-white dark:bg-brand-paper`}>
                                        <h5 className="font-serif font-bold text-brand-navy dark:text-foreground text-sm mb-1">{t.label}</h5>
                                        {t.explanation && (
                                            <p className="font-serif text-xs text-brand-text/60 italic leading-relaxed break-words [overflow-wrap:anywhere]">
                                                &ldquo;{t.explanation}&rdquo;
                                            </p>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Open chapter button at bottom of expansion */}
                        {onChapterOpen && (
                            <div className="px-4 sm:px-5 py-4">
                                <button
                                    onClick={() => onChapterOpen(novelTheme.book, chapter, sentenceIndex, cleanedTitle || "")}
                                    className={`w-full min-h-[44px] flex items-center justify-center gap-2 text-[11px] font-bold uppercase tracking-wider rounded-lg px-4 py-2.5 border transition-all active:scale-95 ${novelTheme.btnBorder}`}
                                >
                                    <ExternalLink size={13} />
                                    Buksan ang Kabanata {chapter}
                                </button>
                            </div>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </article>
    );
}
