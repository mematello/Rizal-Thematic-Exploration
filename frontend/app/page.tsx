
"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { NovelToggle } from "@/components/NovelToggle";
import { ContentTabs } from "@/components/ContentTabs";
import { SearchBar } from "@/components/SearchBar";
import { ChapterGrid } from "@/components/ChapterGrid";
import { CharacterList } from "@/components/CharacterList";
import { ThemeList } from "@/components/ThemeList";
import { ChapterModal } from "@/components/ChapterModal";
import { motion, AnimatePresence } from "framer-motion";
import { HeroSection } from "@/components/HeroSection";
import { useModeStore } from "@/store/modeStore";
import { ChapterSidebar } from "@/components/ChapterSidebar";
import { ArrowUp, Menu, X, BookOpen } from "lucide-react";
import { useNovelBackground } from "@/hooks/useNovelBackground";
import { ChapterContent } from "@/types";
import { Footer } from "@/components/Footer";

type Novel = "noli" | "fili";
type Tab = "chapters" | "characters" | "themes";


interface SelectedChapter {
  book: string;
  chapter_number: number;
  title: string;
}

export default function Home() {
  const router = useRouter();
  const [novel, setNovel] = useState<Novel>("noli");
  const [activeTab, setActiveTab] = useState<Tab>("chapters");
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [searchQuery, setSearchQuery] = useState("");
  const { mode } = useModeStore();

  // Modal State lifted from ChapterGrid
  const [selectedChapter, setSelectedChapter] = useState<SelectedChapter | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [chapterContent, setChapterContent] = useState<ChapterContent[]>([]);
  const [loadingContent, setLoadingContent] = useState(false);
  const [highlightSentenceIndex, setHighlightSentenceIndex] = useState<number | undefined>(undefined);
  const [showBackToTop, setShowBackToTop] = useState(false);

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    if (query.trim()) {
      router.push(`/search?q=${encodeURIComponent(query)}&novel=${novel}`);
    }
  };

  const handleChapterSelect = async (book: string, chapter: number, title?: string, sentenceIndex?: number) => {
    // If title is missing (from cross-nav), we might want to fetch it or just use "Chapter X"
    const displayTitle = title || `Chapter ${chapter}`;

    setSelectedChapter({ book, chapter_number: chapter, title: displayTitle });
    setIsModalOpen(true);
    setLoadingContent(true);
    setChapterContent([]);
    setHighlightSentenceIndex(sentenceIndex);
    console.log('handleChapterSelect - sentenceIndex:', sentenceIndex);

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
      const apiUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
      const res = await fetch(`${apiUrl}/api/v1/chapters/${book}/${chapter}?mode=${mode}`);
      if (!res.ok) throw new Error("Failed to fetch chapter content");
      const data = await res.json();
      setChapterContent(data);


      // If we didn't have a title, maybe we can find it in the response? 
      // The sentence objects don't always carry the chapter title in a unified way (they do in DB, but let's see).
      // Actually checking content.py, currently ChapterContentResponse only has index and text.
      // We can rely on the user knowing what they clicked or update the API later.
    } catch (error) {
      console.error("Error fetching chapter content:", error);
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
      className={`min-h-[100dvh] flex flex-col justify-between transition-colors duration-700 ease-in-out`}
      data-novel={novel}
      style={backgroundStyle}
    >
      {/* Header Section */}
      <header
        className={`sticky top-0 z-40 backdrop-blur-xl border-b transition-all duration-500 safe-top ${novel === 'fili'
          ? 'bg-[#EAE3D5]/90 border-brand-navy/10'
          : 'bg-[#F5F1E9]/90 border-brand-gold/20'
          }`}
      >
        <div className="max-w-7xl mx-auto px-4 py-2.5 sm:py-3 relative">
          <div className="flex flex-col md:flex-row md:items-center gap-2 sm:gap-4">
            {/* Top row on mobile: Logo + Hamburger / Desktop: Logo on left */}
            <div className="flex items-center justify-between md:flex-initial">
              <motion.button
                onClick={() => router.push('/')}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-left text-lg font-serif font-black text-brand-gold tracking-tighter cursor-pointer select-none py-1"
                aria-label="Pumunta sa panimulang pahina ng RizalEngine"
              >
                RIZAL<span className="text-brand-navy font-bold ml-1">ENGINE</span>
              </motion.button>

              {/* Hamburger Button (Mobile only) */}
              <button
                type="button"
                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                className="md:hidden min-w-[44px] min-h-[44px] -mr-2 rounded-full flex items-center justify-center text-brand-navy hover:bg-black/5 active:scale-95 transition-all"
                aria-expanded={isMobileMenuOpen}
                aria-label={isMobileMenuOpen ? "Isara ang menu" : "Buksan ang menu ng pagpipilian"}
                aria-controls="mobile-navigation-drawer"
              >
                {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
            </div>

            {/* Search Bar - Center on Desktop, Full Width below logo on Mobile */}
            <div className="flex-1 max-w-2xl mx-auto w-full">
              <SearchBar
                onSearch={handleSearch}
                variant="hero"
                placeholder={`Magsaliksik sa ${novel === 'noli' ? 'Noli Me Tangere' : 'El Filibusterismo'}...`}
              />
            </div>

            {/* Desktop Navigation - Hidden on mobile to eliminate markup duplication */}
            <div className="hidden md:flex items-center gap-3 shrink-0">
              <div className="flex bg-white/80 rounded-full p-1 border border-brand-gold/20 shadow-sm items-center h-10">
                <button
                  type="button"
                  onClick={() => useModeStore.getState().setMode('buod')}
                  className={`min-h-[36px] px-4 py-1.5 text-xs font-bold uppercase tracking-widest rounded-full transition-all ${mode === 'buod' ? 'bg-brand-navy text-white shadow-sm' : 'text-brand-text hover:bg-brand-gold/10'}`}
                  aria-pressed={mode === 'buod'}
                >
                  Buod
                </button>
                <button
                  type="button"
                  onClick={() => useModeStore.getState().setMode('full')}
                  className={`min-h-[36px] px-4 py-1.5 text-xs font-bold uppercase tracking-widest rounded-full transition-all ${mode === 'full' ? 'bg-brand-navy text-white shadow-sm' : 'text-brand-text hover:bg-brand-gold/10'}`}
                  aria-pressed={mode === 'full'}
                >
                  Buong Kwento
                </button>
              </div>
              <NovelToggle selected={novel} onSelect={setNovel} />
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
                className="fixed inset-0 top-[110px] z-30 bg-black/40 backdrop-blur-xs md:hidden"
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
                className="relative z-40 bg-white/98 border-b border-brand-gold/30 shadow-xl overflow-hidden md:hidden px-4 py-5 space-y-5"
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
                          ? "bg-brand-gold/15 border-brand-gold text-brand-navy shadow-sm"
                          : "bg-brand-cream/50 border-brand-gold/20 text-brand-text/70"
                      }`}
                    >
                      <BookOpen size={14} className={novel === "noli" ? "text-brand-gold" : "text-brand-text/40"} />
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
                          : "bg-brand-cream/50 border-brand-gold/20 text-brand-text/70"
                      }`}
                    >
                      <BookOpen size={14} className={novel === "fili" ? "text-fili-accent" : "text-brand-text/40"} />
                      <span>El Filibusterismo</span>
                    </button>
                  </div>
                </div>

                {/* Mode Selector for Mobile */}
                <div>
                  <span className="text-[11px] font-bold uppercase tracking-[0.2em] text-brand-text-light block mb-2">
                    Modo ng Pagbasa
                  </span>
                  <div className="grid grid-cols-2 gap-2 bg-brand-cream/60 p-1 rounded-xl border border-brand-gold/20">
                    <button
                      type="button"
                      onClick={() => {
                        useModeStore.getState().setMode('buod');
                        setIsMobileMenuOpen(false);
                      }}
                      className={`min-h-[44px] flex items-center justify-center text-xs font-bold uppercase tracking-widest rounded-lg transition-all ${
                        mode === 'buod'
                          ? 'bg-brand-navy text-white shadow-sm font-semibold'
                          : 'text-brand-text hover:bg-brand-gold/10'
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
                          ? 'bg-brand-navy text-white shadow-sm font-semibold'
                          : 'text-brand-text hover:bg-brand-gold/10'
                      }`}
                    >
                      Buong Kwento
                    </button>
                  </div>
                </div>
              </motion.div>
            </>
          )}
        </AnimatePresence>
      </header>

      <HeroSection novel={novel} />

      {/* Main Content Area */}
      <div className="max-w-7xl mx-auto px-4 mt-8">
        <ContentTabs activeTab={activeTab} onTabChange={setActiveTab} />

        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="min-h-[50vh]"
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
          className="fixed bottom-6 right-4 md:bottom-8 md:right-10 z-40 inline-flex items-center gap-2 rounded-full bg-brand-navy text-white px-4 py-2 shadow-lg shadow-black/20 hover:shadow-xl hover:bg-brand-navy/90 transition-all text-xs md:text-sm tracking-[0.18em] uppercase font-semibold"
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
