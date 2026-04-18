import React, { useRef } from 'react';
import { ResultCardProps } from '@/types';
import { motion } from 'framer-motion';
import { MapPinned, ArrowRight } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ReadingMapViewerProps {
    noliResults: ResultCardProps[];
    filiResults: ResultCardProps[];
    showScores: boolean;
    onChapterOpen: (book: string, chapter: number, sentenceIndex: number) => void;
}

export function ReadingMapViewer({
    noliResults,
    filiResults,
    onChapterOpen
}: ReadingMapViewerProps) {
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    // Map View sort: Noli chapters ascending (1→N), then Fili chapters ascending (1→N)
    // No relevance-based sorting — chapter order is the authoritative sort for the map.
    const allResults = [
        ...noliResults.slice().sort((a, b) => a.chapter - b.chapter),
        ...filiResults.slice().sort((a, b) => a.chapter - b.chapter),
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
                <h2 className="text-xl md:text-2xl font-serif font-black text-brand-navy tracking-tight mb-3">
                    Gabay na Pagbasa
                </h2>
                <p className="text-sm font-serif text-brand-text/70 max-w-xl leading-relaxed">
                    Batay sa iyong hinanap, ipinapakita ng mapang ito ang pinakamainam na babasahin para mas madaling maunawaan ang iyong inisyal na paghahanap.
                </p>
                <div className="flex gap-4 mt-6 text-xs font-bold uppercase tracking-wider font-serif">
                    <div className="flex items-center gap-1.5 text-noli-accent">
                        <span className="w-2 h-2 rounded-full bg-noli-accent"></span> Noli Me Tangere
                    </div>
                    <div className="flex items-center gap-1.5 text-fili-accent">
                        <span className="w-2 h-2 rounded-full bg-fili-accent"></span> El Filibusterismo
                    </div>
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
                                    onClick={() => onChapterOpen(result.novel, result.chapter, result.sentenceIndex ?? 0)}
                                    className={cn(
                                        "group relative w-64 h-48 flex flex-col justify-between overflow-hidden cursor-pointer",
                                        "bg-white/70 backdrop-blur-md shadow-sm transition-all duration-500",
                                        "border border-brand-gold/10 hover:border-brand-gold/40 rounded-sm",
                                        "p-6 hover:-translate-y-2 hover:shadow-xl"
                                    )}
                                >
                                    {/* Decorative Background Number */}
                                    <span className={cn(
                                        "absolute -right-4 -top-6 text-9xl font-serif font-black opacity-[0.03] select-none transition-opacity group-hover:opacity-[0.07]",
                                        "text-brand-navy"
                                    )}>
                                        {result.chapter}
                                    </span>

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
                                        <h3 className="text-2xl font-serif text-brand-navy mt-3 group-hover:text-brand-gold transition-colors duration-300 leading-tight border-b border-brand-gold/20 pb-2">
                                            Kabanata {result.chapter}
                                        </h3>
                                        <p className="text-xs font-serif text-brand-text/60 mt-2 italic flex justify-between">
                                            <span>Pangungusap {result.sentenceIndex}</span>
                                        </p>
                                    </div>

                                    <div className="flex items-center justify-between text-[10px] font-bold tracking-widest text-brand-text-light group-hover:text-brand-gold transition-colors duration-300">
                                        <span className="flex items-center uppercase">
                                            Basahin 
                                            <svg className="w-3 h-3 ml-1 transform group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                                            </svg>
                                        </span>
                                        {/* Ranking Ordinal Badge */}
                                        <div className="bg-brand-navy/5 px-2 py-0.5 rounded-full text-brand-navy/60 group-hover:bg-brand-gold/10 group-hover:text-brand-gold transition-colors">
                                            #{i+1}
                                        </div>
                                    </div>
                                </div>
                                
                                {/* Connector Arrow */}
                                {i < allResults.length - 1 && (
                                    <div className="text-brand-navy/20 shrink-0 mx-2 animate-pulse">
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
                            <div className="w-4 h-4 rounded-full border-[3px] border-brand-cream bg-brand-navy/30 shadow-md"></div>
                            <span className="text-[9px] font-bold uppercase tracking-widest text-brand-navy/40 mt-3 font-serif">Katapusan</span>
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
