"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import { SearchBar } from "@/components/SearchBar";
import { ResultCard } from "@/components/ResultCard";
import { SkeletonLoader } from "@/components/SkeletonLoader";
import { ChapterModal } from "@/components/ChapterModal";
import { SuggestionsCard } from "@/components/SuggestionsCard";
import { useRizalSearch } from "@/hooks/useRizalSearch";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft, BarChart2 } from "lucide-react";
import { Suspense } from "react";
import { useModeStore } from "@/store/modeStore";
import { useNovelBackground } from "@/hooks/useNovelBackground";

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
    const { mode } = useModeStore();

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
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
            const res = await fetch(`${apiUrl}/api/v1/chapters/${book}/${chapter}?mode=${mode}`);
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

    const handleNavigate = (book: string, chapter: number) => {
        handleChapterOpen(book, chapter, 0); // Open new chapter, reset highlight
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

    const singleNovelResults = novelFilter === 'noli' ? filteredNoli : novelFilter === 'fili' ? filteredFili : [];
    const midpoint = Math.ceil(singleNovelResults.length / 2);
    const col1 = singleNovelResults.slice(0, midpoint);
    const col2 = singleNovelResults.slice(midpoint);

    const isMultiConcept = noliResults.some(r => r.conceptMatchType) || filiResults.some(r => r.conceptMatchType);
    
    // Both mode pre-computes
    const strongNoli = filteredNoli.filter(r => r.conceptMatchType === 'strong');
    const partialNoli = filteredNoli.filter(r => r.conceptMatchType === 'partial');
    const strongFili = filteredFili.filter(r => r.conceptMatchType === 'strong');
    const partialFili = filteredFili.filter(r => r.conceptMatchType === 'partial');

    // Single mode pre-computes
    const strongSingle = singleNovelResults.filter(r => r.conceptMatchType === 'strong');
    const partialSingle = singleNovelResults.filter(r => r.conceptMatchType === 'partial');
    let strongCol1: any[] = [], strongCol2: any[] = [], partialCol1: any[] = [], partialCol2: any[] = [];
    if (isMultiConcept && novelFilter !== 'both') {
        const midStrong = Math.ceil(strongSingle.length / 2);
        strongCol1 = strongSingle.slice(0, midStrong);
        strongCol2 = strongSingle.slice(midStrong);
        const midPartial = Math.ceil(partialSingle.length / 2);
        partialCol1 = partialSingle.slice(0, midPartial);
        partialCol2 = partialSingle.slice(midPartial);
    }

    const [showFallbackNoli, setShowFallbackNoli] = useState(false);
    const [showFallbackFili, setShowFallbackFili] = useState(false);
    const [showFallbackSingle, setShowFallbackSingle] = useState(false);
    useEffect(() => {
        setShowFallbackNoli(false);
        setShowFallbackFili(false);
        setShowFallbackSingle(false);
    }, [initialQuery, novelFilter]);

    const backgroundStyle = useNovelBackground(novelFilter);

    return (
        <div
            className="min-h-screen flex flex-col"
            style={backgroundStyle}
        >
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

                    <div className="flex items-center gap-3">
                        <div className="flex bg-white/80 rounded-full p-1 border border-brand-gold/20 shadow-sm items-center h-8">
                            <button
                                onClick={() => useModeStore.getState().setMode('buod')}
                                className={`px-3 py-1 text-[10px] font-bold uppercase tracking-widest rounded-full transition-all ${mode === 'buod' ? 'bg-brand-navy text-white shadow-sm' : 'text-brand-text hover:bg-brand-gold/10'}`}
                            >
                                Buod
                            </button>
                            <button
                                onClick={() => useModeStore.getState().setMode('full')}
                                className={`px-3 py-1 text-[10px] font-bold uppercase tracking-widest rounded-full transition-all ${mode === 'full' ? 'bg-brand-navy text-white shadow-sm' : 'text-brand-text hover:bg-brand-gold/10'}`}
                            >
                                Buong Kwento
                            </button>
                        </div>
                        <button
                            onClick={() => setShowScores(!showScores)}
                            className={`flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider rounded-full px-3 py-1.5 border transition-all h-8 ${showScores
                                ? 'bg-brand-navy text-white border-brand-navy'
                                : 'border-brand-navy/20 text-brand-navy/60 hover:border-brand-navy/40 hover:text-brand-navy'
                                }`}
                        >
                            <BarChart2 size={12} />
                            <span className="hidden sm:inline">{showScores ? 'Itago ang Marka' : 'Ipakita ang Marka'}</span>
                        </button>
                    </div>
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
                            {/* Metadata Context Banner */}
                            {data?.metadata?.result_mode === "semantic_fallback" && totalResults > 0 && (
                                <div className="mb-6 mx-auto w-full bg-brand-gold/10 border border-brand-gold/30 rounded-lg p-3 text-center transition-all animate-in fade-in slide-in-from-top-4">
                                    <p className="text-sm font-serif text-brand-navy">
                                        <span className="font-bold mr-2">i</span>
                                        Walang eksaktong tugma na nahanap para sa <b>&ldquo;{initialQuery}&rdquo;</b>—nagpapakita ng mga kaugnay na talata.
                                    </p>
                                </div>
                            )}

                            {/* Both novels: Noli left, Fili right */}
                            {novelFilter === 'both' && (
                                <div className="grid md:grid-cols-2 gap-6">
                                    <section>
                                        <div className="flex items-center gap-2 mb-4 pb-2 border-b border-noli-gold/40">
                                            <span className="font-serif font-bold text-brand-navy text-sm uppercase tracking-wider">Noli Me Tangere</span>
                                            <span className="bg-noli-gold/15 text-noli-gold text-xs px-2 py-0.5 rounded-full font-bold">{filteredNoli.length}</span>
                                        </div>
                                        {filteredNoli.length === 0 ? (
                                            <p className="text-center py-10 text-brand-text/40 italic font-serif text-sm">Walang resulta sa Noli.</p>
                                        ) : !isMultiConcept ? (
                                            <div className="space-y-4">
                                                {filteredNoli.map((result, i) => (
                                                    <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                        <ResultCard {...result} novel="noli" showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                    </motion.div>
                                                ))}
                                            </div>
                                        ) : (
                                            <div className="space-y-6">
                                                {strongNoli.length > 0 && (
                                                    <div className="space-y-4">
                                                        <h3 className="text-xs font-bold uppercase tracking-wider text-green-700/80 mb-2 border-b border-green-700/20 pb-1">Tugmang Konsepto</h3>
                                                        {strongNoli.map((result, i) => (
                                                            <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                <ResultCard {...result} novel="noli" showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                            </motion.div>
                                                        ))}
                                                    </div>
                                                )}
                                                {partialNoli.length > 0 && (
                                                    <div className="space-y-4">
                                                        {strongNoli.length > 0 ? (
                                                            <div className="space-y-4">
                                                                <button onClick={() => setShowFallbackNoli(!showFallbackNoli)} className="w-full py-2 text-xs font-bold uppercase tracking-wider text-brand-navy/60 bg-brand-navy/5 rounded-md hover:bg-brand-navy/10 transition-colors">
                                                                    {showFallbackNoli ? "Itago ang mga bahagyang tugma" : "Ipakita ang iba pang kaugnay na resulta"}
                                                                </button>
                                                                {showFallbackNoli && (
                                                                    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="space-y-4 overflow-hidden">
                                                                        <h3 className="text-xs font-bold uppercase tracking-wider text-orange-600/80 mb-2 border-b border-orange-600/20 pb-1">Bahagyang Tugma</h3>
                                                                        {partialNoli.map((result, i) => (
                                                                            <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                                <ResultCard {...result} novel="noli" showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                                            </motion.div>
                                                                        ))}
                                                                    </motion.div>
                                                                )}
                                                            </div>
                                                        ) : (
                                                            <div className="space-y-6">
                                                                <div className="space-y-3">
                                                                    <h3 className="text-xs font-bold uppercase tracking-wider text-green-700/80 mb-2 border-b border-green-700/20 pb-1">Tugmang Konsepto</h3>
                                                                    <div className="bg-green-50/50 border border-green-100 rounded-md p-4 text-center">
                                                                        <p className="text-sm text-green-800/80 font-serif">Walang nakitang sipi na tumutugma sa lahat ng konsepto sa iyong paghahanap.</p>
                                                                        <p className="text-xs text-green-700/60 mt-1">I-click ang &ldquo;Ipakita ang iba pang kaugnay na resulta&rdquo; upang makita ang mga bahagyang tugma.</p>
                                                                    </div>
                                                                </div>
                                                                <div className="space-y-4">
                                                                    <button onClick={() => setShowFallbackNoli(!showFallbackNoli)} className="w-full py-2 text-xs font-bold uppercase tracking-wider text-brand-navy/60 bg-brand-navy/5 rounded-md hover:bg-brand-navy/10 transition-colors">
                                                                        {showFallbackNoli ? "Itago ang mga bahagyang tugma" : "Ipakita ang iba pang kaugnay na resulta"}
                                                                    </button>
                                                                    {showFallbackNoli && (
                                                                        <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="space-y-4 overflow-hidden">
                                                                            <h3 className="text-xs font-bold uppercase tracking-wider text-orange-600/80 mb-2 border-b border-orange-600/20 pb-1">Bahagyang Tugma</h3>
                                                                            {partialNoli.map((result, i) => (
                                                                                <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                                    <ResultCard {...result} novel="noli" showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                                                </motion.div>
                                                                            ))}
                                                                        </motion.div>
                                                                    )}
                                                                </div>
                                                            </div>
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                        )}
                                    </section>
                                    <section>
                                        <div className="flex items-center gap-2 mb-4 pb-2 border-b border-fili-magenta/40">
                                            <span className="font-serif font-bold text-brand-navy text-sm uppercase tracking-wider">El Filibusterismo</span>
                                            <span className="bg-fili-magenta/15 text-fili-magenta text-xs px-2 py-0.5 rounded-full font-bold">{filteredFili.length}</span>
                                        </div>
                                        {filteredFili.length === 0 ? (
                                            <p className="text-center py-10 text-brand-text/40 italic font-serif text-sm">Walang resulta sa Fili.</p>
                                        ) : !isMultiConcept ? (
                                            <div className="space-y-4">
                                                {filteredFili.map((result, i) => (
                                                    <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                        <ResultCard {...result} novel="fili" showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                    </motion.div>
                                                ))}
                                            </div>
                                        ) : (
                                            <div className="space-y-6">
                                                {strongFili.length > 0 && (
                                                    <div className="space-y-4">
                                                        <h3 className="text-xs font-bold uppercase tracking-wider text-green-700/80 mb-2 border-b border-green-700/20 pb-1">Tugmang Konsepto</h3>
                                                        {strongFili.map((result, i) => (
                                                            <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                <ResultCard {...result} novel="fili" showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                            </motion.div>
                                                        ))}
                                                    </div>
                                                )}
                                                {partialFili.length > 0 && (
                                                    <div className="space-y-4">
                                                        {strongFili.length > 0 ? (
                                                            <div className="space-y-4">
                                                                <button onClick={() => setShowFallbackFili(!showFallbackFili)} className="w-full py-2 text-xs font-bold uppercase tracking-wider text-brand-navy/60 bg-brand-navy/5 rounded-md hover:bg-brand-navy/10 transition-colors">
                                                                    {showFallbackFili ? "Itago ang mga bahagyang tugma" : "Ipakita ang iba pang kaugnay na resulta"}
                                                                </button>
                                                                {showFallbackFili && (
                                                                    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="space-y-4 overflow-hidden">
                                                                        <h3 className="text-xs font-bold uppercase tracking-wider text-orange-600/80 mb-2 border-b border-orange-600/20 pb-1">Bahagyang Tugma</h3>
                                                                        {partialFili.map((result, i) => (
                                                                            <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                                <ResultCard {...result} novel="fili" showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                                            </motion.div>
                                                                        ))}
                                                                    </motion.div>
                                                                )}
                                                            </div>
                                                        ) : (
                                                            <div className="space-y-6">
                                                                <div className="space-y-3">
                                                                    <h3 className="text-xs font-bold uppercase tracking-wider text-green-700/80 mb-2 border-b border-green-700/20 pb-1">Tugmang Konsepto</h3>
                                                                    <div className="bg-green-50/50 border border-green-100 rounded-md p-4 text-center">
                                                                        <p className="text-sm text-green-800/80 font-serif">Walang nakitang sipi na tumutugma sa lahat ng konsepto sa iyong paghahanap.</p>
                                                                        <p className="text-xs text-green-700/60 mt-1">I-click ang &ldquo;Ipakita ang iba pang kaugnay na resulta&rdquo; upang makita ang mga bahagyang tugma.</p>
                                                                    </div>
                                                                </div>
                                                                <div className="space-y-4">
                                                                    <button onClick={() => setShowFallbackFili(!showFallbackFili)} className="w-full py-2 text-xs font-bold uppercase tracking-wider text-brand-navy/60 bg-brand-navy/5 rounded-md hover:bg-brand-navy/10 transition-colors">
                                                                        {showFallbackFili ? "Itago ang mga bahagyang tugma" : "Ipakita ang iba pang kaugnay na resulta"}
                                                                    </button>
                                                                    {showFallbackFili && (
                                                                        <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="space-y-4 overflow-hidden">
                                                                            <h3 className="text-xs font-bold uppercase tracking-wider text-orange-600/80 mb-2 border-b border-orange-600/20 pb-1">Bahagyang Tugma</h3>
                                                                            {partialFili.map((result, i) => (
                                                                                <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                                    <ResultCard {...result} novel="fili" showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                                                </motion.div>
                                                                            ))}
                                                                        </motion.div>
                                                                    )}
                                                                </div>
                                                            </div>
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                        )}
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
                                    {!isMultiConcept ? (
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
                                                    <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (i + col1.length) * 0.04 }}>
                                                        <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                    </motion.div>
                                                ))}
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="space-y-8">
                                            {/* Strong Section */}
                                            {(strongCol1.length > 0 || strongCol2.length > 0) && (
                                                <div className="space-y-4">
                                                    <h3 className="text-xs font-bold uppercase tracking-wider text-green-700/80 mb-2 border-b border-green-700/20 pb-1">Tugmang Konsepto</h3>
                                                    <div className="grid md:grid-cols-2 gap-4">
                                                        <div className="space-y-4">
                                                            {strongCol1.map((result, i) => (
                                                                <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                    <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                                </motion.div>
                                                            ))}
                                                        </div>
                                                        <div className="space-y-4">
                                                            {strongCol2.map((result, i) => (
                                                                <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                    <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                                </motion.div>
                                                            ))}
                                                        </div>
                                                    </div>
                                                </div>
                                            )}

                                            {/* Partial Section */}
                                            {partialSingle.length > 0 && (
                                                <div className="space-y-4">
                                                    {strongSingle.length > 0 ? (
                                                        <div className="space-y-4">
                                                            <button 
                                                                onClick={() => setShowFallbackSingle(!showFallbackSingle)}
                                                                className="w-full py-3 text-xs font-bold uppercase tracking-wider text-brand-navy/60 bg-brand-navy/5 rounded-md hover:bg-brand-navy/10 transition-colors"
                                                            >
                                                                {showFallbackSingle ? "Itago ang mga bahagyang tugma" : "Ipakita ang iba pang kaugnay na resulta"}
                                                            </button>
                                                            {showFallbackSingle && (
                                                                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="space-y-4 overflow-hidden">
                                                                    <h3 className="text-xs font-bold uppercase tracking-wider text-orange-600/80 mb-2 border-b border-orange-600/20 pb-1">Bahagyang Tugma</h3>
                                                                    <div className="grid md:grid-cols-2 gap-4">
                                                                        <div className="space-y-4">
                                                                            {partialCol1.map((result, i) => (
                                                                                <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                                    <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                                                </motion.div>
                                                                            ))}
                                                                        </div>
                                                                        <div className="space-y-4">
                                                                            {partialCol2.map((result, i) => (
                                                                                <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                                    <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                                                </motion.div>
                                                                            ))}
                                                                        </div>
                                                                    </div>
                                                                </motion.div>
                                                            )}
                                                        </div>
                                                    ) : (
                                                        <div className="space-y-6">
                                                            <div className="space-y-3">
                                                                <h3 className="text-xs font-bold uppercase tracking-wider text-green-700/80 mb-2 border-b border-green-700/20 pb-1">Tugmang Konsepto</h3>
                                                                <div className="bg-green-50/50 border border-green-100 rounded-md p-4 text-center">
                                                                    <p className="text-sm text-green-800/80 font-serif">Walang nakitang sipi na tumutugma sa lahat ng konsepto sa iyong paghahanap.</p>
                                                                    <p className="text-xs text-green-700/60 mt-1">I-click ang &ldquo;Ipakita ang iba pang kaugnay na resulta&rdquo; upang makita ang mga bahagyang tugma.</p>
                                                                </div>
                                                            </div>
                                                            <div className="space-y-4">
                                                                <button 
                                                                    onClick={() => setShowFallbackSingle(!showFallbackSingle)}
                                                                    className="w-full py-3 text-xs font-bold uppercase tracking-wider text-brand-navy/60 bg-brand-navy/5 rounded-md hover:bg-brand-navy/10 transition-colors"
                                                                >
                                                                    {showFallbackSingle ? "Itago ang mga bahagyang tugma" : "Ipakita ang iba pang kaugnay na resulta"}
                                                                </button>
                                                                {showFallbackSingle && (
                                                                    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="space-y-4 overflow-hidden">
                                                                        <h3 className="text-xs font-bold uppercase tracking-wider text-orange-600/80 mb-2 border-b border-orange-600/20 pb-1">Bahagyang Tugma</h3>
                                                                        <div className="grid md:grid-cols-2 gap-4">
                                                                            <div className="space-y-4">
                                                                                {partialCol1.map((result, i) => (
                                                                                    <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                                        <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                                                    </motion.div>
                                                                                ))}
                                                                            </div>
                                                                            <div className="space-y-4">
                                                                                {partialCol2.map((result, i) => (
                                                                                    <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                                        <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} onChapterOpen={handleChapterOpen} />
                                                                                    </motion.div>
                                                                                ))}
                                                                            </div>
                                                                        </div>
                                                                    </motion.div>
                                                                )}
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </>
                            )}

                            {/* Kaugnay na Paghahanap (Suggestions) */}
                            {data?.metadata?.suggestions && data.metadata.suggestions.length > 0 && (
                                <SuggestionsCard 
                                    suggestions={data.metadata.suggestions} 
                                    onSuggestionClick={handleSearch}
                                />
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
                    onNavigate={handleNavigate}
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
