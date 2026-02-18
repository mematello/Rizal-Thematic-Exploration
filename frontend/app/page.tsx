
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { NovelToggle } from "@/components/NovelToggle";
import { ContentTabs } from "@/components/ContentTabs";
import { SearchBar } from "@/components/SearchBar";
import { ChapterGrid } from "@/components/ChapterGrid";
import { CharacterList } from "@/components/CharacterList";
import { ThemeList } from "@/components/ThemeList";
import { ChapterModal } from "@/components/ChapterModal";
import { motion, AnimatePresence } from "framer-motion";

type Novel = "noli" | "fili" | "both";
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

  // Modal State lifted from ChapterGrid
  const [selectedChapter, setSelectedChapter] = useState<SelectedChapter | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [chapterContent, setChapterContent] = useState<ChapterContent[]>([]);
  const [loadingContent, setLoadingContent] = useState(false);
  const [highlightSentenceIndex, setHighlightSentenceIndex] = useState<number | undefined>(undefined);

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
      const res = await fetch(`http://localhost:8000/api/v1/chapters/${book}/${chapter}`);
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

  return (
    <main className="min-h-screen bg-brand-cream pb-20">
      {/* Header Section */}
      <header className="sticky top-0 z-40 bg-brand-cream/98 backdrop-blur-sm border-b border-brand-gold/20 transition-all duration-300">
        <div className="max-w-7xl mx-auto px-4 py-4 relative">
          <div className="flex justify-center items-center relative mb-4">
            <motion.h1
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-xl md:text-2xl font-serif text-center text-brand-navy/70 tracking-wide font-light"
            >
              Rizal Thematic Exploration
            </motion.h1>

            {/* Absolute positioning for desktop, might overlap on very small screens so we can adjust if needed */}
            <div className="absolute right-0 top-0 hidden md:block">
              <NovelToggle selected={novel} onSelect={setNovel} />
            </div>
          </div>

          {/* Mobile view for Novel Toggle - centered below title if screen is small */}
          <div className="block md:hidden mb-4 flex justify-center">
            <NovelToggle selected={novel} onSelect={setNovel} />
          </div>

          <div className="max-w-2xl mx-auto">
            <SearchBar
              onSearch={handleSearch}
              variant="hero"
              placeholder={`Search within ${novel === 'both' ? 'both novels' : novel === 'noli' ? 'Noli Me Tangere' : 'El Filibusterismo'}...`}
            />
          </div>
        </div>
      </header>

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
              <ChapterGrid
                selectedNovel={novel}
                onChapterSelect={handleChapterSelect}
              />
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
