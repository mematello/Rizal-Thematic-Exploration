"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useState } from "react";
import { SearchBar } from "@/components/SearchBar";
import { ResultCard } from "@/components/ResultCard";
import { SkeletonLoader } from "@/components/SkeletonLoader";
import { ChapterModal } from "@/components/ChapterModal";
import { useRizalSearch } from "@/hooks/useRizalSearch";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft, BarChart2 } from "lucide-react";
import { Suspense } from "react";

interface ChapterContent {
    sentence_index: number;
    sentence_text: string;
    themes: any[];
}

function SearchContent() {
    const searchParams = useSearchParams();
    const router = useRouter();
    const initialQuery = searchParams.get("q") || "";
    const novelParam = searchParams.get("novel") as 'noli' | 'fili' | 'both' | null;
    const novelFilter = novelParam || 'both';

    const [showScores, setShowScores] = useState(false);

    // Chapter modal state
    const [selectedChapter, setSelectedChapter] = useState<{ book: string; chapter: number; title: string } | null>(null);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [chapterContent, setChapterContent] = useState<ChapterContent[]>([]);
    const [loadingContent, setLoadingContent] = useState(false);
    const [highlightSentenceIndex, setHighlightSentenceIndex] = useState<number | undefined>(undefined);

    const { data, isLoading, error } = useRizalSearch(initialQuery);

    const handleSearch = (newQuery: string) => {
        router.push(`/search?q=${encodeURIComponent(newQuery)}&novel=${novelFilter}`);
    };

    const handleChapterOpen = async (book: string, chapter: number, sentenceIndex: number) => {
        setSelectedChapter({ book, chapter, title: `Kabanata ${chapter}` });
        setIsModalOpen(true);
        setLoadingContent(true);
        setChapterContent([]);
        setHighlightSentenceIndex(sentenceIndex);

        try {
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
            const res = await fetch(`${apiUrl}/api/v1/chapters/${book}/${chapter}`);
            if (!res.ok) throw new Error("Failed to fetch chapter content");
            const data = await res.json();
            setChapterContent(data);
        } catch (err) {
            console.error("Error fetching chapter:", err);
        } finally {
            setLoadingContent(false);
        }
    };

    const handleCloseModal = () => {
        setIsModalOpen(false);
        setHighlightSentenceIndex(undefined);
        setTimeout(() => {
            setSelectedChapter(null);
            setChapterContent([]);
        }, 300);
    };

    const results = data?.results;
    const noliResults = results?.noli || [];
    const filiResults = results?.fili || [];

    const showNoli = novelFilter === 'both' || novelFilter === 'noli';
    const showFili = novelFilter === 'both' || novelFilter === 'fili';
    const filteredNoli = showNoli ? noliResults : [];
    const filteredFili = showFili ? filiResults : [];
    const totalResults = filteredNoli.length + filteredFili.length;

    const novelLabel =
        novelFilter === 'noli' ? 'Noli Me Tangere' :
            novelFilter === 'fili' ? 'El Filibusterismo' :
                'Parehong Nobela';

    // For single-novel mode, split results into 2 columns
    const singleNovelResults = novelFilter === 'noli' ? filteredNoli : novelFilter === 'fili' ? filteredFili : [];
    const midpoint = Math.ceil(singleNovelResults.length / 2);
    const col1 = singleNovelResults.slice(0, midpoint);
    const col2 = singleNovelResults.slice(midpoint);

    return (
        <div className="min-h-screen bg-brand-cream flex flex-col">
            {/* Sticky Header */}
            <header className="bg-brand-cream/98 border-b border-brand-gold/20 sticky top-0 z-40 backdrop-blur-sm">
                <div className="max-w-6xl mx-auto px-4 py-3 flex items-center gap-4">
                    <button
                        onClick={() => router.push('/')}
                        className="flex items-center gap-2 text-brand-navy/50 hover:text-brand-navy transition-colors shrink-0"
                    >
                        <ArrowLeft size={18} />
                        <span className="hidden md:block font-serif font-bold text-base">Bumalik</span>
                    </button>
                    <div className="flex-1">
                        <SearchBar
                            variant="persistent"
                            defaultValue={initialQuery}
                            onSearch={handleSearch}
                            isLoading={isLoading}
                        />
                    </div>
                </div>

                {/* Sub-header: novel context + score toggle */}
                <div className="max-w-6xl mx-auto px-4 pb-3 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-brand-text/50 font-serif">
                            Naghahanap sa: <span className="font-bold text-brand-navy">{novelLabel}</span>
                        </span>
                        {!isLoading && initialQuery && (
                            <span className="text-xs text-brand-text/30 font-serif">
                                · {totalResults} {totalResults === 1 ? 'resulta' : 'mga resulta'}
                            </span>
                        )}
                    </div>

                    <button
                        onClick={() => setShowScores(!showScores)}
                        className={`flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider rounded-full px-3 py-1.5 border transition-all ${showScores
                            ? 'bg-brand-navy text-white border-brand-navy'
                            : 'border-brand-navy/20 text-brand-navy/60 hover:border-brand-navy/40 hover:text-brand-navy'
                            }`}
                    >
                        <BarChart2 size={12} />
                        {showScores ? 'Itago ang Marka' : 'Ipakita ang Marka'}
                    </button>
                </div>
            </header>

            {/* Main Content */}
            <main className="flex-1 max-w-6xl w-full mx-auto px-4 py-6">
                {isLoading ? (
                    <SkeletonLoader />
                ) : error ? (
                    <div className="text-center py-20 text-red-500 font-serif italic">
                        Nagkaroon ng problema sa paghahanap. Subukang muli.
                    </div>
                ) : !initialQuery ? (
                    <div className="text-center py-20 text-brand-text/40 font-serif italic">
                        Maghanap ng salita o tema...
                    </div>
                ) : totalResults === 0 ? (
                    <div className="text-center py-20 text-brand-text/40 font-serif italic">
                        Walang nakitang resulta para sa &ldquo;{initialQuery}&rdquo;.
                    </div>
                ) : (
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={initialQuery + novelFilter}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.3 }}
                        >
                            {/* Both novels: Noli left, Fili right */}
                            {novelFilter === 'both' && (
                                <div className="grid md:grid-cols-2 gap-6">
                                    <section>
                                        <div className="flex items-center gap-2 mb-4 pb-2 border-b border-noli-gold/40">
                                            <span className="font-serif font-bold text-brand-navy text-sm uppercase tracking-wider">Noli Me Tangere</span>
                                            <span className="bg-noli-gold/15 text-noli-gold text-xs px-2 py-0.5 rounded-full font-bold">{filteredNoli.length}</span>
                                        </div>
                                        <div className="space-y-4">
                                            {filteredNoli.map((result, i) => (
                                                <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                    <ResultCard {...result} novel="noli" showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                </motion.div>
                                            ))}
                                            {filteredNoli.length === 0 && <p className="text-center py-10 text-brand-text/40 italic font-serif text-sm">Walang resulta sa Noli.</p>}
                                        </div>
                                    </section>
                                    <section>
                                        <div className="flex items-center gap-2 mb-4 pb-2 border-b border-fili-magenta/40">
                                            <span className="font-serif font-bold text-brand-navy text-sm uppercase tracking-wider">El Filibusterismo</span>
                                            <span className="bg-fili-magenta/15 text-fili-magenta text-xs px-2 py-0.5 rounded-full font-bold">{filteredFili.length}</span>
                                        </div>
                                        <div className="space-y-4">
                                            {filteredFili.map((result, i) => (
                                                <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                    <ResultCard {...result} novel="fili" showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                </motion.div>
                                            ))}
                                            {filteredFili.length === 0 && <p className="text-center py-10 text-brand-text/40 italic font-serif text-sm">Walang resulta sa Fili.</p>}
                                        </div>
                                    </section>
                                </div>
                            )}

                            {/* Single novel: split into 2 columns */}
                            {novelFilter !== 'both' && (
                                <>
                                    <div className="flex items-center gap-2 mb-5 pb-2 border-b border-brand-gold/30">
                                        <span className="font-serif font-bold text-brand-navy text-sm uppercase tracking-wider">{novelLabel}</span>
                                        <span className="bg-brand-gold/15 text-brand-gold text-xs px-2 py-0.5 rounded-full font-bold">{singleNovelResults.length}</span>
                                    </div>
                                    <div className="grid md:grid-cols-2 gap-4">
                                        {/* Column 1 */}
                                        <div className="space-y-4">
                                            {col1.map((result, i) => (
                                                <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                    <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                </motion.div>
                                            ))}
                                        </div>
                                        {/* Column 2 */}
                                        <div className="space-y-4">
                                            {col2.map((result, i) => (
                                                <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (i + midpoint) * 0.04 }}>
                                                    <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                </motion.div>
                                            ))}
                                        </div>
                                    </div>
                                </>
                            )}
                        </motion.div>
                    </AnimatePresence>
                )}
            </main>

            {/* Chapter Modal */}
            {selectedChapter && (
                <ChapterModal
                    isOpen={isModalOpen}
                    onClose={handleCloseModal}
                    title={selectedChapter.title}
                    chapterNumber={selectedChapter.chapter}
                    book={selectedChapter.book}
                    content={chapterContent}
                    isLoading={loadingContent}
                    highlightSentenceIndex={highlightSentenceIndex}
                />
            )}
        </div>
    );
}

export default function SearchPage() {
    return (
        <Suspense fallback={<SkeletonLoader />}>
            <SearchContent />
        </Suspense>
    );
}
