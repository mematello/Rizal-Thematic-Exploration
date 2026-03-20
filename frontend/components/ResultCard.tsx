import { useState, useEffect } from "react";
import { ChevronDown, ChevronUp, BookOpen, ExternalLink, Quote } from "lucide-react";
import { ResultCardProps } from "../types";
import { ScoreVisualizer } from "./ScoreVisualizer";
import { motion, AnimatePresence } from "framer-motion";
import { useSearchCacheStore } from "../store/searchCacheStore";

interface ResultCardExtendedProps extends ResultCardProps {
    showScores?: boolean;
    onChapterOpen?: (book: string, chapter: number, sentenceIndex: number) => void;
    onReferenceClick?: (sentenceId: number, sentenceText: string) => void;
}

export function ResultCard({
    id,
    novel,
    chapter,
    chapterTitle,
    passageHtml,
    context = [],
    scores,
    confidenceBadge = false,
    themes = [],
    showScores = false,
    onChapterOpen,
    onReferenceClick,
    sentenceIndex: sentenceIndexProp,
}: ResultCardExtendedProps) {
    const [isExpanded, setIsExpanded] = useState(false);
    const { paksaCache, sanggunianCache, setPaksaBatch, setSanggunianBatch } = useSearchCacheStore();

    const novelTheme = {
        noli: {
            border: 'border-l-noli-gold',
            innerBorder: 'border-noli-gold',
            novelBadge: 'bg-noli-gold/10 text-noli-gold',
            novelName: 'Noli Me Tangere',
            btnBorder: 'border-noli-gold/40 hover:border-noli-gold text-noli-gold hover:bg-noli-gold/5',
            book: 'noli',
            text: 'text-noli-gold',
            bg: 'bg-noli-gold',
        },
        fili: {
            border: 'border-l-fili-magenta',
            innerBorder: 'border-fili-magenta',
            novelBadge: 'bg-fili-magenta/10 text-fili-magenta',
            novelName: 'El Filibusterismo',
            btnBorder: 'border-fili-magenta/40 hover:border-fili-magenta text-fili-magenta hover:bg-fili-magenta/5',
            book: 'elfili',
            text: 'text-fili-magenta',
            bg: 'bg-fili-magenta',
        },
    }[novel];

    const hasThemes = themes.length > 0;
    const sentenceIndex = sentenceIndexProp ?? 0;
    const mainSentenceId = parseInt(id);

    // Batch fetch themes and references when expanded
    useEffect(() => {
        if (!isExpanded || context.length === 0) return;

        const allIds = [mainSentenceId, ...context.map(s => s.id)];
        
        // Filter out those already in cache
        const missingPaksa = allIds.filter(id => !paksaCache[id]);
        const missingSanggunian = allIds.filter(id => !sanggunianCache[id]);

        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

        if (missingPaksa.length > 0) {
            fetch(`${apiUrl}/api/v1/sentences/batch/paksa`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(missingPaksa)
            })
            .then(res => res.json())
            .then(data => setPaksaBatch(data))
            .catch(err => console.error("Batch Paksa Error:", err));
        }

        if (missingSanggunian.length > 0) {
            fetch(`${apiUrl}/api/v1/sentences/batch/sanggunian`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(missingSanggunian)
            })
            .then(res => res.json())
            .then(data => setSanggunianBatch(data))
            .catch(err => console.error("Batch Sanggunian Error:", err));
        }
    }, [isExpanded, context, mainSentenceId, paksaCache, sanggunianCache, setPaksaBatch, setSanggunianBatch]);

    return (
        <article
            id={id}
            className={`bg-brand-paper rounded-sm border border-brand-gold/20 border-l-4 ${novelTheme.border} hover:shadow-md transition-all duration-200 scroll-mt-20 overflow-hidden`}
            aria-labelledby={`result-title-${id}`}
        >
            {/* Card Header — clickable to open chapter */}
            <div
                className="p-5 cursor-pointer group"
                onClick={() => onChapterOpen?.(novelTheme.book, chapter, sentenceIndex)}
                title="Buksan ang kabanata"
            >
                <div className="flex items-start justify-between gap-3 mb-2">
                    <div className="flex-1 min-w-0">
                        <span className={`text-[10px] font-bold uppercase tracking-[0.2em] px-2 py-0.5 rounded-full ${novelTheme.novelBadge} inline-block mb-2`}>
                            {novelTheme.novelName}
                        </span>
                        <h3
                            id={`result-title-${id}`}
                            className="font-serif font-bold text-brand-navy text-base leading-snug group-hover:text-brand-gold transition-colors flex items-center gap-2"
                        >
                            Kabanata {chapter}: {chapterTitle}
                            <ExternalLink size={13} className="opacity-0 group-hover:opacity-60 transition-opacity shrink-0" />
                        </h3>
                    </div>
                    {confidenceBadge && (
                        <span className="shrink-0 text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wide bg-brand-gold/10 text-brand-gold border border-brand-gold/20">
                            Mataas
                        </span>
                    )}
                </div>

                {/* Main Passage */}
                <div className="relative">
                    <div
                        className="font-serif text-sm leading-relaxed text-brand-text/80"
                        dangerouslySetInnerHTML={{ __html: passageHtml }}
                        aria-label="Sipi mula sa nobela"
                    />
                    
                    {/* Inline Indicators for Main Passage */}
                    <div className="mt-2 flex flex-wrap gap-2 items-center">
                        {mainSentenceId && paksaCache[mainSentenceId]?.themes?.map((t, idx) => (
                            <span key={idx} className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-tighter ${novelTheme.novelBadge} border border-current/20`}>
                                {t.label} 
                                <span className="opacity-50">{(t.confidence * 100).toFixed(0)}%</span>
                            </span>
                        ))}
                        {mainSentenceId && sanggunianCache[mainSentenceId]?.has_reference && (
                            <button 
                                onClick={(e) => { 
                                    e.stopPropagation(); 
                                    const mainText = context.find(s => s.is_center)?.text || "";
                                    onReferenceClick?.(mainSentenceId, mainText); 
                                }}
                                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-tighter border transition-colors ${novelTheme.btnBorder}`}
                                title="Buksan ang Sanggunian"
                            >
                                <Quote size={8} /> Sanggunian
                            </button>
                        )}
                    </div>
                </div>
            </div>

            {/* Action Row */}
            <div className="px-5 py-2.5 border-t border-brand-gold/10 flex items-center gap-2 bg-brand-cream/30">
                <button
                    onClick={(e) => { 
                        e.stopPropagation(); 
                        setIsExpanded(!isExpanded); 
                    }}
                    className={`flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider rounded-full px-3 py-1.5 border transition-all ${novelTheme.btnBorder}`}
                    aria-expanded={isExpanded}
                >
                    <BookOpen size={11} />
                    {isExpanded ? 'Itago' : 'Ipakita ang Konteksto'}
                    {isExpanded ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
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
                        <div className="px-5 pb-3 pt-2 border-t border-brand-gold/10 bg-brand-cream/20">
                            <ScoreVisualizer semantic={scores.semantic} lexical={scores.lexical} />
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
                        className="overflow-hidden pb-4"
                    >
                        {/* Structured Context */}
                        <div className="mx-5 mt-3 p-4 rounded-sm bg-brand-cream border border-brand-gold/10 font-serif space-y-3">
                            {context.map((s) => (
                                <div key={s.id} className="relative group">
                                    <p className={`text-sm leading-relaxed ${s.is_center ? 'text-brand-text font-bold' : 'text-brand-text/60 italic'}`}>
                                        {s.is_center ? <strong>{s.text}</strong> : s.text}
                                    </p>
                                    
                                    {/* Inline Indicators for Context Sentences */}
                                    <div className="mt-1 flex flex-wrap gap-1.5 items-center">
                                        {paksaCache[s.id]?.themes?.map((t, idx) => (
                                            <span key={idx} className={`inline-flex items-center gap-1 px-1.5 py-0 rounded-full text-[8px] font-bold uppercase tracking-tighter ${novelTheme.novelBadge} border border-current/10`}>
                                                {t.label}
                                                <span className="opacity-40">{(t.confidence * 100).toFixed(0)}%</span>
                                            </span>
                                        ))}
                                        {sanggunianCache[s.id]?.has_reference && (
                                            <button 
                                                onClick={() => onReferenceClick?.(s.id, s.text)}
                                                className={`inline-flex items-center gap-1 px-1.5 py-0 rounded-full text-[8px] font-bold uppercase tracking-tighter border opacity-40 group-hover:opacity-100 transition-opacity ${novelTheme.btnBorder}`}
                                            >
                                                <Quote size={7} /> Sanggunian
                                            </button>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Summary Meaning section (kept as requested to not change logic) */}
                        {hasThemes && (
                            <div className="mx-5 mt-4 space-y-2">
                                <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-brand-text/40 block">Kahulugan ng Tema</span>
                                {themes.map((t) => (
                                    <div key={t.id} className={`p-3 rounded-sm border-l-4 ${novelTheme.innerBorder} bg-white shadow-sm`}>
                                        <h5 className="font-serif font-bold text-brand-navy text-sm mb-1">{t.label}</h5>
                                        {t.explanation && (
                                            <p className="font-serif text-xs text-brand-text/60 italic leading-relaxed">
                                                &ldquo;{t.explanation}&rdquo;
                                            </p>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Open chapter button */}
                        {onChapterOpen && (
                            <div className="px-5 mt-4">
                                <button
                                    onClick={() => onChapterOpen(novelTheme.book, chapter, sentenceIndex)}
                                    className={`w-full flex items-center justify-center gap-2 text-[11px] font-bold uppercase tracking-wider rounded-sm px-4 py-2.5 border transition-all ${novelTheme.btnBorder}`}
                                >
                                    <ExternalLink size={12} />
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
