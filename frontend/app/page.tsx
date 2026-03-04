
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
import { ArrowUp } from "lucide-react";
import { useNovelBackground } from "@/hooks/useNovelBackground";

type Novel = "noli" | "fili";
type Tab = "chapters" | "characters" | "themes";

interface ChapterContent {
  sentence_index: number;
  sentence_text: string;
  themes: any[]; // Using any[] for simplicity in page.tsx as it just passes data through
}

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
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
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

  return (
    <main
      className={`min-h-screen pb-20 transition-colors duration-700 ease-in-out`}
      data-novel={novel}
      style={backgroundStyle}
    >
      {/* Header Section */}
      <header
        className={`sticky top-0 z-40 backdrop-blur-xl border-b transition-all duration-500 ${novel === 'fili'
          ? 'bg-[#EAE3D5]/85 border-brand-navy/10'
          : 'bg-[#F5F1E9]/85 border-brand-gold/20'
          }`}
      >
        <div className="max-w-7xl mx-auto px-4 py-3 relative">
          <div className="flex flex-col md:flex-row md:items-center gap-4">
            {/* Minimal Title/Logo for Sticky Header */}
            <div className="flex items-center justify-between flex-1">
              <motion.h1
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-lg font-serif font-black text-brand-gold tracking-tighter"
              >
                RIZAL<span className="text-brand-navy font-bold ml-1">EXPLORER</span>
              </motion.h1>

              <div className="flex md:hidden items-center gap-2">
                <div className="flex bg-white/80 rounded-full p-1 border border-brand-gold/20 shadow-sm items-center h-9">
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
                <NovelToggle selected={novel} onSelect={setNovel} />
              </div>
            </div>

            <div className="flex-1 max-w-2xl mx-auto w-full">
              <SearchBar
                onSearch={handleSearch}
                variant="hero"
                placeholder={`Magsaliksik sa ${novel === 'noli' ? 'Noli Me Tangere' : 'El Filibusterismo'}...`}
              />
            </div>

            <div className="hidden md:flex items-center gap-3">
              <div className="flex bg-white/80 rounded-full p-1 border border-brand-gold/20 shadow-sm items-center h-10">
                <button
                  onClick={() => useModeStore.getState().setMode('buod')}
                  className={`px-4 py-1.5 text-xs font-bold uppercase tracking-widest rounded-full transition-all ${mode === 'buod' ? 'bg-brand-navy text-white shadow-sm' : 'text-brand-text hover:bg-brand-gold/10'}`}
                >
                  Buod
                </button>
                <button
                  onClick={() => useModeStore.getState().setMode('full')}
                  className={`px-4 py-1.5 text-xs font-bold uppercase tracking-widest rounded-full transition-all ${mode === 'full' ? 'bg-brand-navy text-white shadow-sm' : 'text-brand-text hover:bg-brand-gold/10'}`}
                >
                  Buong Kwento
                </button>
              </div>
              <NovelToggle selected={novel} onSelect={setNovel} />
            </div>
          </div>
        </div>
      </header>

      <HeroSection novel={novel} />

      {/* Main Content Area */}
      <div className="max-w-7xl mx-auto px-4 mt-8">
        <ContentTabs activeTab={activeTab} onTabChange={setActiveTab} />

        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3 }}
            className="min-h-[50vh]"
          >
            {activeTab === "chapters" && (
              <div className="mt-6 flex gap-6">
                {/* Sticky sidebar navigation for Kabanata 1–64 (Noli only) */}
                <ChapterSidebar novel={novel} />

                <div className="flex-1">
                  <ChapterGrid
                    selectedNovel={novel}
                    onChapterSelect={handleChapterSelect}
                  />
                </div>
              </div>
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

      {/* Footer Decoration */}
      <footer className="mt-20 py-10 text-center border-t border-brand-gold/30">
        <p className="text-brand-text-light font-serif italic text-sm">
          &quot;To foretell the destiny of a nation, it is necessary to open the book that tells of her past.&quot;
        </p>
      </footer>

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
    </main>
  );
}
