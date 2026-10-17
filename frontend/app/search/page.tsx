"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import { SearchBar } from "@/components/SearchBar";
import { ResultCard } from "@/components/ResultCard";
import { SkeletonLoader } from "@/components/SkeletonLoader";
import { ChapterModal } from "@/components/ChapterModal";
import { SuggestionsCard } from "@/components/SuggestionsCard";
import { ReadingMapViewer } from "@/components/ReadingMapViewer";
import { useRizalSearch } from "@/hooks/useRizalSearch";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft, BarChart2, LayoutGrid, MapPinned, Menu, X } from "lucide-react";
import { Suspense } from "react";
import { useModeStore } from "@/store/modeStore";
import { useNovelBackground } from "@/hooks/useNovelBackground";
import { ThemeToggle } from "@/components/ThemeToggle";
import { NovelToggle } from "@/components/NovelToggle";
import { useQueryClient } from "@tanstack/react-query";
import { fetchChapterContent } from "@/components/ChapterGrid";
import { ChapterContent } from "@/types";
import { useNovelStore } from "@/store/novelStore";


function SearchContent() {
    const queryClient = useQueryClient();
    const searchParams = useSearchParams();
    const router = useRouter();
    const initialQuery = searchParams.get("q") || "";
    const rawNovel = searchParams.get("novel");
    const normalizedNovel = rawNovel === 'elfili' ? 'fili' : rawNovel;
    const novelFilter: 'noli' | 'fili' | 'both' = (normalizedNovel as 'noli' | 'fili' | 'both') || 'both';

    const [showScores, setShowScores] = useState(false);
    const [viewMode, setViewMode] = useState<'grid' | 'map'>('grid');
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const { mode } = useModeStore();

    // Chapter modal state
    const [selectedChapter, setSelectedChapter] = useState<{ book: string; chapter: number; title: string } | null>(null);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [chapterContent, setChapterContent] = useState<ChapterContent[]>([]);
    const [loadingContent, setLoadingContent] = useState(false);
    const [highlightSentenceIndex, setHighlightSentenceIndex] = useState<number | undefined>(undefined);
    // Map View only: stores the anchor + context window from the clicked result
    const [highlightContext, setHighlightContext] = useState<{ anchor: number; contextMin: number; contextMax: number } | null>(null);

    // Determine which novels to fetch based on novelFilter
    const requiredNovels: ('noli' | 'fili')[] = novelFilter === 'both' 
        ? ['noli', 'fili'] 
        : [novelFilter];

    const { setSelectedNovel } = useNovelStore();

    useEffect(() => {
        document.documentElement.setAttribute('data-novel', novelFilter);
        if (novelFilter === 'noli' || novelFilter === 'fili') {
            setSelectedNovel(novelFilter);
        }
    }, [novelFilter, setSelectedNovel]);

    const { data, isLoading, error } = useRizalSearch(initialQuery, requiredNovels);

    const handleSearch = (newQuery: string) => {
        router.push(`/search?q=${encodeURIComponent(newQuery)}&novel=${novelFilter}`);
    };

    // Grid View handler — no context highlighting, Grid View is unmodified
    const handleChapterOpen = async (book: string, chapter: number, sentenceIndex: number, chapterTitle?: string) => {
        setHighlightContext(null); // Grid View never sets context highlighting
        setSelectedChapter({ book, chapter, title: chapterTitle || '' });
        setIsModalOpen(true);
        setLoadingContent(true);
        setChapterContent([]);
        setHighlightSentenceIndex(sentenceIndex);

        try {
            const data = await queryClient.fetchQuery({
                queryKey: ["chapter", book, chapter, mode],
                queryFn: () => fetchChapterContent(book, chapter, mode),
                staleTime: 1000 * 60 * 60 * 24,
            });
            setChapterContent(data);
        } catch (err) {
            console.error("Error fetching chapter:", err);
        } finally {
            setLoadingContent(false);
        }
    };

    // Map View handler — sets context window from the clicked result's sentenceIndex
    // Context definition comes from the Grid View convention (sentenceIndex ± 2),
    // Map View does NOT compute or generate context independently.
    const handleMapChapterOpen = async (book: string, chapter: number, sentenceIndex: number, chapterTitle?: string) => {
        setHighlightContext({
            anchor: sentenceIndex,
            contextMin: sentenceIndex - 2,
            contextMax: sentenceIndex + 2,
        });
        setSelectedChapter({ book, chapter, title: chapterTitle || '' });
        setIsModalOpen(true);
        setLoadingContent(true);
        setChapterContent([]);
        setHighlightSentenceIndex(sentenceIndex);

        try {
            const data = await queryClient.fetchQuery({
                queryKey: ["chapter", book, chapter, mode],
                queryFn: () => fetchChapterContent(book, chapter, mode),
                staleTime: 1000 * 60 * 60 * 24,
            });
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
        setHighlightContext(null); // Always reset context on close
        setTimeout(() => {
            setSelectedChapter(null);
            setChapterContent([]);
        }, 300);
    };

    const handleNavigate = (book: string, chapter: number) => {
        handleChapterOpen(book, chapter, 0); // Navigate within modal — clears context (Grid View behavior)
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

    const headerTheme = {
        noli: {
            bg: "bg-brand-cream/98 dark:bg-[#0C1424]/95",
            border: "border-b border-brand-gold/20 dark:border-brand-gold/25",
            accent: "text-brand-gold",
            pillBorder: "border border-brand-gold/20 dark:border-brand-gold/30",
            activePill: "bg-brand-navy text-white dark:bg-brand-gold dark:text-[#0F141C]",
            ring: "focus-visible:ring-brand-gold",
            hoverText: "hover:text-brand-navy dark:hover:text-brand-gold",
        },
        fili: {
            bg: "bg-[#EAE3D5]/98 dark:bg-[#180D0E]/95",
            border: "border-b border-[#8D2D2D]/25 dark:border-[#8D2D2D]/35",
            accent: "text-[#8D2D2D] dark:text-[#E06C75]",
            pillBorder: "border border-[#8D2D2D]/25 dark:border-[#8D2D2D]/35",
            activePill: "bg-[#8D2D2D] text-white dark:bg-[#8D2D2D] dark:text-white",
            ring: "focus-visible:ring-[#8D2D2D]",
            hoverText: "hover:text-[#8D2D2D] dark:hover:text-[#E06C75]",
        },
        both: {
            bg: "bg-[#F5F1E9]/98 dark:bg-[#0D131F]/95",
            border: "border-b border-brand-gold/20 dark:border-brand-gold/25",
            accent: "text-brand-gold",
            pillBorder: "border border-brand-gold/20 dark:border-brand-gold/30",
            activePill: "bg-brand-navy text-white dark:bg-brand-gold dark:text-[#0F141C]",
            ring: "focus-visible:ring-brand-gold",
            hoverText: "hover:text-brand-navy dark:hover:text-brand-gold",
        },
    }[novelFilter];

    return (
        <div className="min-h-screen flex flex-col relative" data-novel={novelFilter || 'both'}>
            {/* Fixed viewport backdrop: responsive framing prevents 8x zoom */}
            <div
                className="novel-backdrop-layer"
                data-novel={novelFilter || 'both'}
                aria-hidden="true"
            />
            {/* Sticky Header */}
            <header className={`${headerTheme.bg} ${headerTheme.border} sticky top-0 z-40 backdrop-blur-sm safe-top transition-colors`}>
                <div className="max-w-6xl mx-auto px-4 py-2.5 sm:py-3 relative">
                    <div className="flex flex-col md:flex-row md:items-center gap-2 sm:gap-4">
                        {/* Top row on mobile: Logo/Back + Theme + Menu / Desktop: Logo on left */}
                        <div className="flex items-center justify-between md:flex-initial">
                            <button
                                type="button"
                                onClick={() => router.push('/')}
                                className={`flex items-center gap-2 group focus:outline-none focus-visible:ring-2 ${headerTheme.ring} rounded-full px-2 py-1 select-none`}
                                aria-label="Bumalik sa panimulang pahina ng RizalEngine"
                            >
                                <ArrowLeft size={18} className={`${headerTheme.accent} group-hover:-translate-x-0.5 transition-transform`} />
                                <span className={`text-lg font-serif font-black ${headerTheme.accent} tracking-tighter`}>
                                    RIZAL<span className="text-brand-navy dark:text-foreground font-bold ml-0.5">ENGINE</span>
                                </span>
                            </button>

                            {/* Mobile Actions: Theme icon toggle + Menu button only (44px tap targets) */}
                            <div className="flex items-center gap-1.5 md:hidden">
                                <ThemeToggle className="w-11 h-11 min-w-[44px] min-h-[44px]" />
                                <button
                                    type="button"
                                    onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                                    className={`min-w-[44px] min-h-[44px] w-11 h-11 rounded-full flex items-center justify-center text-brand-navy dark:text-brand-gold hover:bg-black/5 dark:hover:bg-white/5 active:scale-95 transition-all focus:outline-none focus-visible:ring-2 ${headerTheme.ring}`}
                                    aria-expanded={isMobileMenuOpen}
                                    aria-label={isMobileMenuOpen ? "Isara ang menu" : "Buksan ang menu ng pagpipilian"}
                                    aria-controls="mobile-search-navigation-drawer"
                                >
                                    {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                                </button>
                            </div>
                        </div>

                        {/* Search Bar - Center on Desktop, Full Width below logo on Mobile */}
                        <div className="flex-1 max-w-xl mx-auto w-full md:mx-4">
                            <SearchBar
                                variant="persistent"
                                defaultValue={initialQuery}
                                onSearch={handleSearch}
                                isLoading={isLoading}
                            />
                        </div>

                        {/* Desktop Navigation: Right group = novel dropdown, Buod | Buong Kwento, theme icon toggle (consistent h-10, spacing, focus rings) */}
                        <div className="hidden md:flex items-center gap-3 shrink-0">
                            <NovelToggle
                                selected={novelFilter}
                                includeBoth={true}
                                onSelect={(newNovel) => {
                                    if (newNovel === 'noli' || newNovel === 'fili') {
                                        setSelectedNovel(newNovel);
                                    }
                                    router.push(`/search?q=${encodeURIComponent(initialQuery)}&novel=${newNovel}`);
                                }}
                            />
                            <div className={`flex bg-white/80 dark:bg-[#182232] rounded-full p-1 ${headerTheme.pillBorder} shadow-xs items-center h-10`}>
                                <button
                                    type="button"
                                    onClick={() => useModeStore.getState().setMode('buod')}
                                    className={`h-8 px-3.5 py-1 text-xs font-bold uppercase tracking-wider rounded-full transition-all focus:outline-none focus-visible:ring-2 ${headerTheme.ring} ${mode === 'buod' ? `${headerTheme.activePill} shadow-xs font-bold` : `text-brand-text/70 dark:text-foreground/75 ${headerTheme.hoverText} hover:bg-black/5`}`}
                                    aria-pressed={mode === 'buod'}
                                >
                                    Buod
                                </button>
                                <button
                                    type="button"
                                    onClick={() => useModeStore.getState().setMode('full')}
                                    className={`h-8 px-3.5 py-1 text-xs font-bold uppercase tracking-wider rounded-full transition-all focus:outline-none focus-visible:ring-2 ${headerTheme.ring} ${mode === 'full' ? `${headerTheme.activePill} shadow-xs font-bold` : `text-brand-text/70 dark:text-foreground/75 ${headerTheme.hoverText} hover:bg-black/5`}`}
                                    aria-pressed={mode === 'full'}
                                >
                                    Buong Kwento
                                </button>
                            </div>
                            <ThemeToggle className="w-10 h-10 min-w-[40px] min-h-[40px]" />
                        </div>
                    </div>
                </div>

                {/* Sub-header: novel context + Map/Score controls */}
                <div className="border-t border-brand-gold/10 bg-brand-cream/40 dark:bg-black/20">
                    <div className="max-w-6xl mx-auto px-4 py-2 flex items-center justify-between gap-2 sm:gap-4 overflow-x-auto no-scrollbar">
                        <div className="flex items-center gap-2 min-w-0">
                            <span className="text-xs text-brand-text/70 dark:text-brand-text-muted font-serif truncate">
                                Naghahanap sa: <span className="font-bold text-brand-navy dark:text-brand-gold">{novelLabel}</span>
                            </span>
                            {!isLoading && initialQuery && (
                                <span className="text-xs text-brand-text/40 dark:text-brand-text-muted/60 font-serif shrink-0">
                                    · {totalResults} {totalResults === 1 ? 'resulta' : 'mga resulta'}
                                </span>
                            )}
                        </div>

                        <div className="flex items-center gap-2 shrink-0">
                            <button
                                id="map-view-toggle"
                                type="button"
                                onClick={() => setViewMode(viewMode === 'map' ? 'grid' : 'map')}
                                className={`h-8 min-h-[32px] flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider rounded-full px-3 py-1 border transition-all shrink-0 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-gold ${
                                    viewMode === 'map'
                                        ? 'bg-brand-navy text-white dark:bg-brand-gold dark:text-[#0F141C] border-brand-navy dark:border-brand-gold shadow-sm font-bold'
                                        : 'border-brand-navy/20 dark:border-brand-gold/30 text-brand-navy/70 dark:text-foreground/80 hover:border-brand-navy/40 dark:hover:border-brand-gold hover:text-brand-navy dark:hover:text-brand-gold bg-white/80 dark:bg-[#182232]/85'
                                }`}
                                title={viewMode === 'map' ? 'Bumalik sa Listahan' : 'Tingnan ang Gabay na Mapa'}
                                aria-label={viewMode === 'map' ? 'Bumalik sa Listahan' : 'Tingnan ang Gabay na Mapa'}
                            >
                                {viewMode === 'map' ? <LayoutGrid size={13} /> : <MapPinned size={13} />}
                                <span>{viewMode === 'map' ? 'Listahan' : 'Gabay na Mapa'}</span>
                            </button>

                            <button
                                type="button"
                                onClick={() => setShowScores(!showScores)}
                                className={`h-8 min-h-[32px] flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider rounded-full px-3 py-1 border transition-all shrink-0 active:scale-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-gold ${showScores
                                    ? 'bg-brand-navy text-white dark:bg-brand-gold dark:text-[#0F141C] border-brand-navy dark:border-brand-gold shadow-xs font-bold'
                                    : 'border-brand-navy/20 dark:border-brand-gold/30 text-brand-navy/70 dark:text-foreground/80 hover:border-brand-navy/40 dark:hover:border-brand-gold hover:text-brand-navy dark:hover:text-brand-gold bg-white/80 dark:bg-[#182232]/85'
                                    }`}
                                aria-label={showScores ? 'Itago ang Marka' : 'Ipakita ang Marka'}
                            >
                                <BarChart2 size={13} />
                                <span>{showScores ? 'Itago ang Marka' : 'Ipakita ang Marka'}</span>
                            </button>
                        </div>
                    </div>
                </div>

                {/* Mobile Slide-down Navigation Panel for Search */}
                <AnimatePresence>
                    {isMobileMenuOpen && (
                        <>
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 0.2 }}
                                onClick={() => setIsMobileMenuOpen(false)}
                                className="fixed inset-0 top-[110px] z-50 bg-black/40 backdrop-blur-xs md:hidden"
                                aria-hidden="true"
                            />
                            <motion.div
                                id="mobile-search-navigation-drawer"
                                role="region"
                                aria-label="Mobile Search Navigation"
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                transition={{ duration: 0.25, ease: "easeInOut" }}
                                className="relative z-50 bg-white/98 dark:bg-brand-paper/98 border-b border-brand-gold/30 shadow-xl overflow-hidden md:hidden px-4 py-5 space-y-5 pb-[max(1.5rem,env(safe-area-inset-bottom))]"
                            >
                                {/* Novel Selector for Mobile */}
                                <div>
                                    <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-brand-text-light dark:text-brand-gold/80 block mb-2">
                                        Pumili ng Nobela
                                    </span>
                                    <div className="grid grid-cols-3 gap-2">
                                        <button
                                            type="button"
                                            onClick={() => {
                                                router.push(`/search?q=${encodeURIComponent(initialQuery)}&novel=both`);
                                                setIsMobileMenuOpen(false);
                                            }}
                                            className={`min-h-[44px] flex items-center justify-center gap-1.5 px-2 py-2.5 rounded-xl border text-xs font-serif font-bold uppercase tracking-wider transition-all ${
                                                novelFilter === 'both'
                                                    ? 'bg-brand-gold/15 border-brand-gold text-brand-navy dark:text-brand-gold shadow-sm'
                                                    : 'bg-brand-cream/50 dark:bg-black/20 border-brand-gold/20 text-brand-text/70 dark:text-foreground/70'
                                            }`}
                                        >
                                            <span>Lahat</span>
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => {
                                                setSelectedNovel('noli');
                                                router.push(`/search?q=${encodeURIComponent(initialQuery)}&novel=noli`);
                                                setIsMobileMenuOpen(false);
                                            }}
                                            className={`min-h-[44px] flex items-center justify-center gap-1.5 px-2 py-2.5 rounded-xl border text-xs font-serif font-bold uppercase tracking-wider transition-all ${
                                                novelFilter === 'noli'
                                                    ? 'bg-brand-gold/15 border-brand-gold text-brand-navy dark:text-brand-gold shadow-sm'
                                                    : 'bg-brand-cream/50 dark:bg-black/20 border-brand-gold/20 text-brand-text/70 dark:text-foreground/70'
                                            }`}
                                        >
                                            <span>Noli</span>
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => {
                                                setSelectedNovel('fili');
                                                router.push(`/search?q=${encodeURIComponent(initialQuery)}&novel=fili`);
                                                setIsMobileMenuOpen(false);
                                            }}
                                            className={`min-h-[44px] flex items-center justify-center gap-1.5 px-2 py-2.5 rounded-xl border text-xs font-serif font-bold uppercase tracking-wider transition-all ${
                                                novelFilter === 'fili'
                                                    ? 'bg-fili-accent/15 border-fili-accent text-fili-accent shadow-sm'
                                                    : 'bg-brand-cream/50 dark:bg-black/20 border-brand-gold/20 text-brand-text/70 dark:text-foreground/70'
                                            }`}
                                        >
                                            <span>El Fili</span>
                                        </button>
                                    </div>
                                </div>

                                {/* Mode Selector for Mobile */}
                                <div>
                                    <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-brand-text-light dark:text-brand-gold/80 block mb-2">
                                        Modo ng Pagbasa
                                    </span>
                                    <div className="grid grid-cols-2 gap-2 bg-brand-cream/60 dark:bg-black/30 p-1 rounded-xl border border-brand-gold/20">
                                        <button
                                            type="button"
                                            onClick={() => {
                                                useModeStore.getState().setMode('buod');
                                                setIsMobileMenuOpen(false);
                                            }}
                                            className={`min-h-[44px] flex items-center justify-center text-xs font-bold uppercase tracking-widest rounded-lg transition-all ${
                                                mode === 'buod'
                                                    ? 'bg-brand-navy text-white dark:bg-brand-gold dark:text-[#0F141C] shadow-sm font-bold'
                                                    : 'text-brand-text/70 dark:text-foreground/75 hover:bg-brand-gold/10 hover:text-brand-navy dark:hover:text-brand-gold'
                                            }`}
                                        >
                                            Buod
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => {
                                                useModeStore.getState().setMode('full');
                                                setIsMobileMenuOpen(false);
                                            }}
                                            className={`min-h-[44px] flex items-center justify-center text-xs font-bold uppercase tracking-widest rounded-lg transition-all ${
                                                mode === 'full'
                                                    ? 'bg-brand-navy text-white dark:bg-brand-gold dark:text-[#0F141C] shadow-sm font-bold'
                                                    : 'text-brand-text/70 dark:text-foreground/75 hover:bg-brand-gold/10 hover:text-brand-navy dark:hover:text-brand-gold'
                                            }`}
                                        >
                                            Buong Kwento
                                        </button>
                                    </div>
                                </div>

                                {/* Extra View Toggles for Mobile */}
                                <div>
                                    <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-brand-text-light dark:text-brand-gold/80 block mb-2">
                                        Mga Tanawin
                                    </span>
                                    <div className="grid grid-cols-2 gap-2">
                                        <button
                                            type="button"
                                            onClick={() => {
                                                setViewMode(viewMode === 'map' ? 'grid' : 'map');
                                                setIsMobileMenuOpen(false);
                                            }}
                                            className={`min-h-[44px] flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl border text-xs font-serif font-bold uppercase tracking-wider transition-all ${
                                                viewMode === 'map'
                                                    ? 'bg-brand-navy text-white dark:bg-brand-gold dark:text-[#0F141C] border-brand-navy dark:border-brand-gold shadow-sm'
                                                    : 'bg-brand-cream/50 dark:bg-black/20 border-brand-gold/20 text-brand-text/70 dark:text-foreground/70'
                                            }`}
                                        >
                                            {viewMode === 'map' ? <LayoutGrid size={15} /> : <MapPinned size={15} />}
                                            <span>{viewMode === 'map' ? 'Listahan' : 'Gabay na Mapa'}</span>
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => {
                                                setShowScores(!showScores);
                                                setIsMobileMenuOpen(false);
                                            }}
                                            className={`min-h-[44px] flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl border text-xs font-serif font-bold uppercase tracking-wider transition-all ${
                                                showScores
                                                    ? 'bg-brand-navy text-white dark:bg-brand-gold dark:text-[#0F141C] border-brand-navy dark:border-brand-gold shadow-xs'
                                                    : 'bg-brand-cream/50 dark:bg-black/20 border-brand-gold/20 text-brand-text/70 dark:text-foreground/70'
                                            }`}
                                        >
                                            <BarChart2 size={15} />
                                            <span>{showScores ? 'Itago Marka' : 'Marka'}</span>
                                        </button>
                                    </div>
                                </div>

                                {/* Theme Selector for Mobile */}
                                <div>
                                    <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-brand-text-light dark:text-brand-gold/80 block mb-2">
                                        Tema ng Paningin
                                    </span>
                                    <ThemeToggle variant="segmented" className="w-full justify-between" />
                                </div>
                            </motion.div>
                        </>
                    )}
                </AnimatePresence>
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
                            {data?.metadata?.result_mode === "semantic_fallback" && data?.metadata?.has_lexical_hits === false && totalResults > 0 && (
                                <div className="mb-6 mx-auto w-full bg-brand-gold/10 border border-brand-gold/30 rounded-lg p-3 text-center transition-all animate-in fade-in slide-in-from-top-4">
                                    <p className="text-sm font-serif text-brand-navy">
                                        <span className="font-bold mr-2">i</span>
                                        Walang eksaktong tugma na nahanap para sa <b>&ldquo;{initialQuery}&rdquo;</b>—nagpapakita ng mga kaugnay na talata.
                                    </p>
                                </div>
                            )}

                            {viewMode === 'map' ? (
                                <ReadingMapViewer
                                    noliResults={noliResults}
                                    filiResults={filiResults}
                                    showScores={showScores}
                                    onChapterOpen={handleMapChapterOpen}
                                    novel={novelFilter}
                                />
                            ) : (
                                <>
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
                                                        <ResultCard {...result} novel="noli" showScores={showScores} showNovelBadge={true} onChapterOpen={handleChapterOpen} />
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
                                                                <ResultCard {...result} novel="noli" showScores={showScores} showNovelBadge={true} onChapterOpen={handleChapterOpen} />
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
                                                                                <ResultCard {...result} novel="noli" showScores={showScores} showNovelBadge={true} onChapterOpen={handleChapterOpen} />
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
                                                                                    <ResultCard {...result} novel="noli" showScores={showScores} showNovelBadge={true} onChapterOpen={handleChapterOpen} />
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
                                                        <ResultCard {...result} novel="fili" showScores={showScores} showNovelBadge={true} onChapterOpen={handleChapterOpen} />
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
                                                                <ResultCard {...result} novel="fili" showScores={showScores} showNovelBadge={true} onChapterOpen={handleChapterOpen} />
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
                                                                                <ResultCard {...result} novel="fili" showScores={showScores} showNovelBadge={true} onChapterOpen={handleChapterOpen} />
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
                                                                                    <ResultCard {...result} novel="fili" showScores={showScores} showNovelBadge={true} onChapterOpen={handleChapterOpen} />
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
                                                        <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} showNovelBadge={false} onChapterOpen={handleChapterOpen} />
                                                    </motion.div>
                                                ))}
                                            </div>
                                            {/* Column 2 */}
                                            <div className="space-y-4">
                                                {col2.map((result, i) => (
                                                    <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: (i + col1.length) * 0.04 }}>
                                                        <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} showNovelBadge={false} onChapterOpen={handleChapterOpen} />
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
                                                                    <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} showNovelBadge={false} onChapterOpen={handleChapterOpen} />
                                                                </motion.div>
                                                            ))}
                                                        </div>
                                                        <div className="space-y-4">
                                                            {strongCol2.map((result, i) => (
                                                                <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                    <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} showNovelBadge={false} onChapterOpen={handleChapterOpen} />
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
                                                                                    <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} showNovelBadge={false} onChapterOpen={handleChapterOpen} />
                                                                                </motion.div>
                                                                            ))}
                                                                        </div>
                                                                        <div className="space-y-4">
                                                                            {partialCol2.map((result, i) => (
                                                                                <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                                    <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} showNovelBadge={false} onChapterOpen={handleChapterOpen} />
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
                                                                                        <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} showNovelBadge={false} onChapterOpen={handleChapterOpen} />
                                                                                    </motion.div>
                                                                                ))}
                                                                            </div>
                                                                            <div className="space-y-4">
                                                                                {partialCol2.map((result, i) => (
                                                                                    <motion.div key={result.id} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}>
                                                                                        <ResultCard {...result} novel={novelFilter as 'noli' | 'fili'} showScores={showScores} showNovelBadge={false} onChapterOpen={handleChapterOpen} />
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
                    highlightContext={highlightContext}
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
