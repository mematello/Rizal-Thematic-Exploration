import React, { useRef } from 'react';
import { ResultCardProps } from '@/types';
import { motion } from 'framer-motion';
import { MapPinned, ArrowRight } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ReadingMapViewerProps {
    noliResults: ResultCardProps[];
    filiResults: ResultCardProps[];
    showScores: boolean;
    onChapterOpen: (book: string, chapter: number, sentenceIndex: number, chapterTitle?: string) => void;
    novel?: 'noli' | 'fili' | 'both';
}

export function ReadingMapViewer({
    noliResults,
    filiResults,
    onChapterOpen,
    novel = 'both'
}: ReadingMapViewerProps) {
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const showNoli = novel === 'noli' || novel === 'both';
    const showFili = novel === 'fili' || novel === 'both';

    // Map View sort: Noli chapters ascending (1→N), then Fili chapters ascending (1→N)
    // Filtered by selected novel(s)
    const allResults = [
        ...(showNoli ? noliResults.slice().sort((a, b) => a.chapter - b.chapter) : []),
        ...(showFili ? filiResults.slice().sort((a, b) => a.chapter - b.chapter) : []),
    ];

    if (allResults.length === 0) {
        return (
            <div className="text-center py-20 text-brand-text/40 font-serif italic">
                Walang resulta upang bumuo ng mapa.
            </div>
        );
    }

    return (
        <div className="w-full max-w-6xl mx-auto py-8">
            <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col items-center mb-12 text-center px-4"
            >
                <div className="bg-brand-gold/10 p-3 rounded-full mb-4 border border-brand-gold/20 shadow-sm">
                    <MapPinned className="text-brand-gold h-6 w-6" />
                </div>
                <h2 className="text-xl md:text-2xl font-serif font-black text-brand-navy dark:text-foreground tracking-tight mb-3">
                    Gabay na Pagbasa
                </h2>
                <p className="text-sm font-serif text-brand-text/70 dark:text-brand-text-muted max-w-xl leading-relaxed">
                    Batay sa iyong hinanap, ipinapakita ng mapang ito ang pinakamainam na babasahin para mas madaling maunawaan ang iyong inisyal na paghahanap.
                </p>
                <div className="flex gap-4 mt-6 text-xs font-bold uppercase tracking-wider font-serif">
                    {showNoli && (
                        <div className="flex items-center gap-1.5 text-noli-accent">
                            <span className="w-2 h-2 rounded-full bg-noli-accent"></span> Noli Me Tangere
                        </div>
                    )}
                    {showFili && (
                        <div className="flex items-center gap-1.5 text-fili-accent">
                            <span className="w-2 h-2 rounded-full bg-fili-accent"></span> El Filibusterismo
                        </div>
                    )}
                </div>
            </motion.div>

            {/* Horizontal Map Container */}
            <div className="relative w-full overflow-hidden">
                <div 
                    ref={scrollContainerRef}
                    className="flex overflow-x-auto gap-4 pb-12 pt-8 px-4 md:px-12 snap-x snap-mandatory scroll-smooth"
                    style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
                >
                    {allResults.map((result, i) => {
                        const isNoli = result.novel === 'noli';

                        return (
                            <motion.div
                                key={`${result.novel}-${result.id}-${i}`}
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ duration: 0.4, delay: i * 0.05 }}
                                className="flex items-center gap-4 snap-center shrink-0"
                            >
                                {/* Node Box */}
                                <div 
                                    onClick={() => onChapterOpen(result.novel, result.chapter, result.sentenceIndex ?? 0, result.chapterTitle)}
                                    className={cn(
                                        "group relative w-64 h-48 flex flex-col justify-between overflow-hidden cursor-pointer",
                                        "bg-white/80 backdrop-blur-md shadow-sm transition-all duration-500 rounded-sm p-6 hover:-translate-y-2 hover:shadow-xl",
                                        isNoli 
                                            ? "map-card-noli border border-brand-gold/15 hover:border-brand-gold/40 dark:hover:shadow-black/50" 
                                            : "map-card-fili border border-[#8D2D2D]/20 hover:border-[#8D2D2D]/40 dark:hover:shadow-black/50"
                                    )}
                                >
                                    {/* Decorative Background Number */}
                                    <span
                                        aria-hidden="true"
                                        data-num={result.chapter}
                                        className={cn(
                                        "watermark-numeral absolute -right-4 -top-6 text-9xl font-serif font-black opacity-[0.03] dark:opacity-[0.08] select-none transition-opacity group-hover:opacity-[0.07] dark:group-hover:opacity-[0.15] pointer-events-none",
                                        isNoli ? "text-brand-navy dark:text-brand-gold" : "text-[#2C1810] dark:text-[#E06C75]"
                                    )} />

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
                                        <h3 className={cn(
                                            "text-2xl font-serif text-brand-navy dark:text-foreground mt-3 transition-colors duration-300 leading-tight pb-2",
                                            isNoli ? "group-hover:text-brand-gold border-b border-brand-gold/20 dark:border-brand-gold/30" : "group-hover:text-[#E06C75] border-b border-[#8D2D2D]/20 dark:border-[#E06C75]/30"
                                        )}>
                                            Kabanata {result.chapter}
                                        </h3>
                                        <p className="text-xs font-serif text-brand-text/60 dark:text-brand-text-muted mt-2 italic flex justify-between">
                                            <span>Pangungusap {result.sentenceIndex}</span>
                                        </p>
                                    </div>

                                    <div className={cn(
                                        "flex items-center justify-between text-[10px] font-bold tracking-widest text-brand-text-light dark:text-foreground/75 transition-colors duration-300",
                                        isNoli ? "group-hover:text-brand-gold" : "group-hover:text-[#E06C75]"
                                    )}>
                                        <span className="flex items-center uppercase">
                                            Basahin 
                                            <svg className="w-3 h-3 ml-1 transform group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                                            </svg>
                                        </span>
                                        {/* Ranking Ordinal Badge */}
                                        <div className={cn(
                                            "px-2 py-0.5 rounded-full transition-colors font-bold",
                                            isNoli 
                                                ? "bg-brand-navy/5 dark:bg-brand-gold/15 text-brand-navy/60 dark:text-brand-gold group-hover:bg-brand-gold/20 group-hover:text-brand-gold"
                                                : "bg-[#8D2D2D]/5 dark:bg-[#E06C75]/15 text-[#8D2D2D]/70 dark:text-[#E06C75] group-hover:bg-[#E06C75]/20 group-hover:text-[#E06C75]"
                                        )}>
                                            #{i+1}
                                        </div>
                                    </div>
                                </div>
                                
                                {/* Connector Arrow */}
                                {i < allResults.length - 1 && (
                                    <div className={cn(
                                        "shrink-0 mx-2 animate-pulse",
                                        isNoli ? "text-brand-navy/20 dark:text-brand-gold/40" : "text-[#8D2D2D]/25 dark:text-[#E06C75]/40"
                                    )}>
                                        <ArrowRight className="w-5 h-5" />
                                    </div>
                                )}
                            </motion.div>
                        );
                    })}

                    {/* End node */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: allResults.length * 0.05 }}
                        className="flex items-center justify-center snap-center shrink-0 w-24 h-48"
                    >
                        <div className="flex flex-col items-center">
                            <div className={cn(
                                "w-4 h-4 rounded-full border-[3px] shadow-md",
                                novel === 'fili'
                                    ? "border-brand-cream dark:border-[#1A0C0E] bg-[#8D2D2D]/30 dark:bg-[#E06C75]/50"
                                    : "border-brand-cream dark:border-[#18202C] bg-brand-navy/30 dark:bg-brand-gold/50"
                            )}></div>
                            <span className="text-[9px] font-bold uppercase tracking-widest text-brand-navy/40 dark:text-foreground/50 mt-3 font-serif">Katapusan</span>
                        </div>
                    </motion.div>
                </div>
            </div>
            {/* Scroll hint CSS for hiding scrollbars nicely */}
            <style jsx global>{`
                .scroll-smooth::-webkit-scrollbar {
                    display: none;
                }
            `}</style>
        </div>
    );
}
