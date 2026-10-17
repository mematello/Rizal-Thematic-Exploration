
"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { NovelToggle } from "@/components/NovelToggle";
import { ContentTabs } from "@/components/ContentTabs";
import { SearchBar } from "@/components/SearchBar";
import { ChapterGrid, fetchChapters, fetchChapterContent } from "@/components/ChapterGrid";
import { CharacterList } from "@/components/CharacterList";
import { ThemeList, fetchThemes } from "@/components/ThemeList";
import { ChapterModal } from "@/components/ChapterModal";
import { motion, AnimatePresence } from "framer-motion";
import { HeroSection } from "@/components/HeroSection";
import { useModeStore } from "@/store/modeStore";
import { ChapterSidebar } from "@/components/ChapterSidebar";
import { ArrowUp, Menu, X, BookOpen } from "lucide-react";
import { useNovelBackground } from "@/hooks/useNovelBackground";
import { ThemeToggle } from "@/components/ThemeToggle";
import { ChapterContent } from "@/types";
import { Footer } from "@/components/Footer";
import { formatChapterTitle } from "@/lib/utils";
import { useNovelStore } from "@/store/novelStore";

type Novel = "noli" | "fili";
type Tab = "chapters" | "characters" | "themes";


interface SelectedChapter {
  book: string;
  chapter_number: number;
  title: string;
}

export default function Home() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { selectedNovel, setSelectedNovel } = useNovelStore();
  const [novel, setNovelState] = useState<Novel>(selectedNovel);

  useEffect(() => {
    setNovelState(selectedNovel);
    if (typeof document !== 'undefined') {
      document.documentElement.setAttribute('data-novel', selectedNovel);
    }
  }, [selectedNovel]);

  const setNovel = (newNovel: Novel) => {
    setNovelState(newNovel);
    setSelectedNovel(newNovel);
  };
  const [activeTab, setActiveTab] = useState<Tab>("chapters");

  useEffect(() => {
    // Prefetch themes and chapters data immediately on homepage mount for instantaneous tab switching
    queryClient.prefetchQuery({
      queryKey: ["themes"],
      queryFn: fetchThemes,
      staleTime: 1000 * 60 * 60 * 24,
    });
    queryClient.prefetchQuery({
      queryKey: ["chapters", "buod"],
      queryFn: () => fetchChapters("buod"),
      staleTime: 1000 * 60 * 60 * 24,
    });
    queryClient.prefetchQuery({
      queryKey: ["chapters", "full"],
      queryFn: () => fetchChapters("full"),
      staleTime: 1000 * 60 * 60 * 24,
    });
  }, [queryClient]);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [searchQuery, setSearchQuery] = useState("");
  const { mode } = useModeStore();

  // Modal State lifted from ChapterGrid
  const [selectedChapter, setSelectedChapter] = useState<SelectedChapter | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [highlightSentenceIndex, setHighlightSentenceIndex] = useState<number | undefined>(undefined);
  const [showBackToTop, setShowBackToTop] = useState(false);

  // React Query for chapter content with 24-hour cache
  const { data: chapterContent = [], isFetching: loadingContent } = useQuery<ChapterContent[]>({
    queryKey: ["chapter", selectedChapter?.book, selectedChapter?.chapter_number, mode],
    queryFn: () => selectedChapter ? fetchChapterContent(selectedChapter.book, selectedChapter.chapter_number, mode) : Promise.resolve([]),
    enabled: !!selectedChapter && isModalOpen,
    staleTime: 1000 * 60 * 60 * 24,
  });

  // Prefetch adjacent chapters (X+1 and X-1) in buod mode
  useEffect(() => {
    if (isModalOpen && selectedChapter && mode === "buod") {
      const { book, chapter_number } = selectedChapter;
      const maxChapters = book.toLowerCase() === "noli" ? 64 : 39;
      if (chapter_number + 1 <= maxChapters) {
        queryClient.prefetchQuery({
          queryKey: ["chapter", book, chapter_number + 1, "buod"],
          queryFn: () => fetchChapterContent(book, chapter_number + 1, "buod"),
          staleTime: 1000 * 60 * 60 * 24,
        });
      }
      if (chapter_number - 1 >= 1) {
        queryClient.prefetchQuery({
          queryKey: ["chapter", book, chapter_number - 1, "buod"],
          queryFn: () => fetchChapterContent(book, chapter_number - 1, "buod"),
          staleTime: 1000 * 60 * 60 * 24,
        });
      }
    }
  }, [isModalOpen, selectedChapter, mode, queryClient]);

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    if (query.trim()) {
      router.push(`/search?q=${encodeURIComponent(query)}&novel=${novel}`);
    }
  };

  const handleChapterSelect = (book: string, chapter: number, title?: string, sentenceIndex?: number) => {
    const displayTitle = title ? formatChapterTitle(title) : `Chapter ${chapter}`;
    setSelectedChapter({ book, chapter_number: chapter, title: displayTitle });
    setIsModalOpen(true);
    setHighlightSentenceIndex(sentenceIndex);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setHighlightSentenceIndex(undefined);
    setTimeout(() => {
      setSelectedChapter(null);
    }, 300);
  };

  const handleNavigate = (book: string, chapter: number) => {
    // Navigate to a different chapter
    handleChapterSelect(book, chapter);
  };

  useEffect(() => {
    const handleScroll = () => {
      if (typeof window === "undefined") return;
      setShowBackToTop(window.scrollY > 480);
    };

    handleScroll();
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToTop = () => {
    if (typeof window === "undefined") return;
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const backgroundStyle = useNovelBackground(novel);

  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Close mobile menu on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsMobileMenuOpen(false);
    };
    if (isMobileMenuOpen) {
      window.addEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "hidden";
    }
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [isMobileMenuOpen]);

  return (
    <main
      className={`w-full max-w-full min-w-0 min-h-[100dvh] flex flex-col justify-between transition-colors duration-700 ease-in-out relative`}
      data-novel={novel}
    >
      {/* Fixed viewport backdrop: responsive framing prevents 8x zoom */}
      <div
        className="novel-backdrop-layer"
        data-novel={novel}
        aria-hidden="true"
      />
      {/* Header Section */}
      <header
        className={`sticky top-0 z-50 backdrop-blur-xl border-b transition-all duration-500 safe-top ${novel === 'fili'
          ? 'bg-[#EAE3D5]/92 dark:bg-[#180D0E]/92 border-[#8D2D2D]/20 dark:border-[#E06C75]/25'
          : 'bg-[#F7F3EB]/92 dark:bg-[#0C1424]/92 border-brand-gold/20 dark:border-brand-gold/25'
          }`}
      >
        <div className="max-w-7xl mx-auto px-4 py-2.5 sm:py-3 relative">
          <div className="flex flex-col md:flex-row md:items-center gap-2 sm:gap-4">
            {/* Top row on mobile: Logo + Theme + Menu / Desktop: Logo on left */}
            <div className="flex items-center justify-between md:flex-initial">
              <motion.button
                onClick={() => router.push('/')}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className={`text-left text-lg font-serif font-black tracking-tighter cursor-pointer select-none py-1 focus:outline-none focus-visible:ring-2 rounded-sm transition-colors ${
                  novel === 'fili'
                    ? 'text-[#8D2D2D] dark:text-[#E06C75] focus-visible:ring-[#8D2D2D]'
                    : 'text-brand-gold focus-visible:ring-brand-gold'
                }`}
                aria-label="Pumunta sa panimulang pahina ng RizalEngine"
              >
                RIZAL<span className="text-brand-navy dark:text-foreground font-bold ml-1">ENGINE</span>
              </motion.button>

              {/* Mobile Actions: Theme icon toggle + Menu button only (44px tap targets) */}
              <div className="flex items-center gap-1.5 md:hidden">
                <ThemeToggle className="w-11 h-11 min-w-[44px] min-h-[44px]" />
                <button
                  type="button"
                  onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                  className={`min-w-[44px] min-h-[44px] w-11 h-11 rounded-full flex items-center justify-center hover:bg-black/5 dark:hover:bg-white/5 active:scale-95 transition-all focus:outline-none focus-visible:ring-2 ${
                    novel === 'fili'
                      ? 'text-[#8D2D2D] dark:text-[#E06C75] focus-visible:ring-[#8D2D2D]'
                      : 'text-brand-navy dark:text-brand-gold focus-visible:ring-brand-gold'
                  }`}
                  aria-expanded={isMobileMenuOpen}
                  aria-label={isMobileMenuOpen ? "Isara ang menu" : "Buksan ang menu ng pagpipilian"}
                  aria-controls="mobile-navigation-drawer"
                >
                  {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
                </button>
              </div>
            </div>

            {/* Search Bar - Center on Desktop, Full Width below logo on Mobile */}
            <div className="flex-1 max-w-xl mx-auto w-full md:mx-4">
              <SearchBar
                onSearch={handleSearch}
                variant="hero"
                placeholder={`Magsaliksik sa ${novel === 'noli' ? 'Noli Me Tangere' : 'El Filibusterismo'}...`}
              />
            </div>

            {/* Desktop Navigation: Right group = novel dropdown, Buod | Buong Kwento, theme icon toggle (consistent h-10, spacing, focus rings) */}
            <div className="hidden md:flex items-center gap-3 shrink-0">
              <NovelToggle selected={novel} onSelect={setNovel} />
              <div className={`flex bg-white/80 dark:bg-[#182232] rounded-full p-1 border shadow-xs items-center h-10 transition-colors ${
                novel === 'fili'
                  ? 'border-[#8D2D2D]/20 dark:border-[#E06C75]/30'
                  : 'border-brand-gold/20 dark:border-brand-gold/30'
              }`}>
                <button
                  type="button"
                  onClick={() => useModeStore.getState().setMode('buod')}
                  className={`h-8 px-3.5 py-1 text-xs font-bold uppercase tracking-wider rounded-full transition-all focus:outline-none focus-visible:ring-2 ${
                    mode === 'buod'
                      ? (novel === 'fili' ? 'bg-[#8D2D2D] text-white dark:bg-[#E06C75] dark:text-[#180D0E] shadow-xs font-bold' : 'bg-brand-navy text-white dark:bg-brand-gold dark:text-[#0F141C] shadow-xs font-bold')
                      : (novel === 'fili' ? 'text-brand-text/70 dark:text-foreground/75 hover:text-[#8D2D2D] dark:hover:text-[#E06C75] hover:bg-[#8D2D2D]/10' : 'text-brand-text/70 dark:text-foreground/75 hover:text-brand-navy dark:hover:text-brand-gold hover:bg-brand-gold/10')
                  }`}
                  aria-pressed={mode === 'buod'}
                >
                  Buod
                </button>
                <button
                  type="button"
                  onClick={() => useModeStore.getState().setMode('full')}
                  className={`h-8 px-3.5 py-1 text-xs font-bold uppercase tracking-wider rounded-full transition-all focus:outline-none focus-visible:ring-2 ${
                    mode === 'full'
                      ? (novel === 'fili' ? 'bg-[#8D2D2D] text-white dark:bg-[#E06C75] dark:text-[#180D0E] shadow-xs font-bold' : 'bg-brand-navy text-white dark:bg-brand-gold dark:text-[#0F141C] shadow-xs font-bold')
                      : (novel === 'fili' ? 'text-brand-text/70 dark:text-foreground/75 hover:text-[#8D2D2D] dark:hover:text-[#E06C75] hover:bg-[#8D2D2D]/10' : 'text-brand-text/70 dark:text-foreground/75 hover:text-brand-navy dark:hover:text-brand-gold hover:bg-brand-gold/10')
                  }`}
                  aria-pressed={mode === 'full'}
                >
                  Buong Kwento
                </button>
              </div>
              <ThemeToggle className="w-10 h-10 min-w-[40px] min-h-[40px]" />
            </div>

          </div>
        </div>

        {/* Mobile Slide-down Navigation Panel */}
        <AnimatePresence>
          {isMobileMenuOpen && (
            <>
              {/* Backdrop */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                onClick={() => setIsMobileMenuOpen(false)}
                className="fixed inset-0 top-[110px] z-50 bg-black/40 backdrop-blur-xs md:hidden"
                aria-hidden="true"
              />

              {/* Drawer Content */}
              <motion.div
                id="mobile-navigation-drawer"
                role="region"
                aria-label="Mobile Navigation"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.25, ease: "easeInOut" }}
                className="relative z-50 bg-white/98 dark:bg-brand-paper/98 border-b border-brand-gold/30 shadow-xl overflow-hidden md:hidden px-4 py-5 space-y-5"
              >
                {/* Novel Selector for Mobile */}
                <div>
                  <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-brand-text-light block mb-2">
                    Pumili ng Nobela
                  </span>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setNovel("noli");
                        setIsMobileMenuOpen(false);
                      }}
                      className={`min-h-[44px] flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl border text-xs font-serif font-bold uppercase tracking-wider transition-all ${
                        novel === "noli"
                          ? "bg-brand-gold/15 border-brand-gold text-brand-navy dark:text-brand-gold shadow-sm"
                          : "bg-brand-cream/50 dark:bg-black/40 border-brand-gold/20 text-brand-text/70 dark:text-foreground/80"
                      }`}
                    >
                      <BookOpen size={14} className={novel === "noli" ? "text-brand-gold" : "text-brand-text/40 dark:text-foreground/40"} />
                      <span>Noli Me Tangere</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setNovel("fili");
                        setIsMobileMenuOpen(false);
                      }}
                      className={`min-h-[44px] flex items-center justify-center gap-2 px-3 py-2.5 rounded-xl border text-xs font-serif font-bold uppercase tracking-wider transition-all ${
                        novel === "fili"
                          ? "bg-fili-accent/15 border-fili-accent text-fili-accent shadow-sm"
                          : "bg-brand-cream/50 dark:bg-black/40 border-brand-gold/20 text-brand-text/70 dark:text-foreground/80"
                      }`}
                    >
                      <BookOpen size={14} className={novel === "fili" ? "text-fili-accent" : "text-brand-text/40 dark:text-foreground/40"} />
                      <span>El Filibusterismo</span>
                    </button>
                  </div>
                </div>

                {/* Mode Selector for Mobile */}
                <div>
                  <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-brand-text-light block mb-2">
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

                {/* Theme Selector for Mobile */}
                <div>
                  <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-brand-text-light block mb-2">
                    Tema ng Paningin
                  </span>
                  <ThemeToggle variant="segmented" className="w-full justify-between" />
                </div>
              </motion.div>
            </>
          )}
        </AnimatePresence>
      </header>

      <HeroSection novel={novel} />

      {/* Main Content Area */}
      <div className="w-full max-w-7xl mx-auto px-4 mt-6 sm:mt-8 min-w-0">
        {/* Mobile Novel Switcher: Top to bottom on mobile: Header -> Novel Toggle -> ContentTabs -> Sticky Kabanata pill row */}
        <div className="flex md:hidden items-center justify-between gap-3 mb-4 max-w-xl mx-auto px-2 relative z-40">
          <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-brand-text/60 font-serif">
            Piling Nobela
          </span>
          <NovelToggle selected={novel} onSelect={setNovel} />
        </div>

        <ContentTabs activeTab={activeTab} onTabChange={setActiveTab} />

        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="min-h-[50vh] w-full max-w-full min-w-0"
          >
            {activeTab === "chapters" && (
              <ChapterSidebar novel={novel}>
                <ChapterGrid
                  selectedNovel={novel}
                  onChapterSelect={handleChapterSelect}
                />
              </ChapterSidebar>
            )}

            {activeTab === "characters" && (
              <CharacterList
                onChapterSelect={handleChapterSelect}
                selectedNovel={novel}
              />
            )}

            {activeTab === "themes" && (
              <ThemeList
                onChapterSelect={handleChapterSelect}
                selectedNovel={novel}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Back to top button */}
      {showBackToTop && (
        <button
          type="button"
          onClick={scrollToTop}
          className="fixed bottom-6 right-4 md:bottom-8 md:right-10 z-40 inline-flex items-center gap-2 rounded-full bg-brand-navy text-white dark:bg-[#182232]/95 dark:text-brand-gold dark:border dark:border-brand-gold/30 dark:hover:bg-[#202C3F] dark:hover:border-brand-gold/60 backdrop-blur-sm px-4 py-2 shadow-lg shadow-black/20 dark:shadow-black/60 hover:shadow-xl hover:bg-brand-navy/90 active:scale-95 transition-all text-xs md:text-sm tracking-[0.18em] uppercase font-semibold"
          aria-label="Bumalik sa itaas"
        >
          <ArrowUp className="h-4 w-4" />
          <span className="hidden sm:inline">Sa Itaas</span>
        </button>
      )}

      {/* Global Chapter Modal */}
      {selectedChapter && (
        <ChapterModal
          isOpen={isModalOpen}
          onClose={handleCloseModal}
          title={selectedChapter.title}
          chapterNumber={selectedChapter.chapter_number}
          book={selectedChapter.book}
          content={chapterContent}
          isLoading={loadingContent}
          highlightSentenceIndex={highlightSentenceIndex}
          onNavigate={handleNavigate}
        />
      )}

      {/* Homepage Footer */}
      <Footer />
    </main>
  );
}
