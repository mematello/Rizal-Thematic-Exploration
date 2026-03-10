"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BookOpen, X } from "lucide-react";
import { ItemModal } from "@/components/ItemModal";
import { CharacterAvatar } from "@/components/CharacterAvatar";
import { CHARACTERS, Character } from "@/lib/characterData";
import { useModeStore } from "@/store/modeStore";

interface Appearance {
    book: string;
    chapter_number: number;
    chapter_title: string;
    sentence_text: string;
    sentence_index: number;
}


interface CharacterListProps {
    onChapterSelect?: (book: string, chapter: number, title?: string, sentenceIndex?: number) => void;
    selectedNovel: "noli" | "fili";
}

interface ChapterInfo {
    book: string;
    chapter_number: number;
    chapter_title: string;
    score: number;
    preview_text?: string;
}

export function CharacterList({ onChapterSelect, selectedNovel }: CharacterListProps) {
    const [selectedChar, setSelectedChar] = useState<Character | null>(null);
    const [chapterAppearances, setChapterAppearances] = useState<ChapterInfo[]>([]);
    const [loading, setLoading] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [sortBy, setSortBy] = useState<'number' | 'relevance'>('number');
    const [zoomedCharName, setZoomedCharName] = useState<string | null>(null);
    const [zoomedDescriptionChar, setZoomedDescriptionChar] = useState<Character | null>(null);
    // Removed local novelFilter state

    const { mode } = useModeStore();

    const fetchChapters = async (char: Character, sort: 'number' | 'relevance') => {
        setLoading(true);
        try {
            // Use aliases if available, otherwise fallback to name logic
            // Combine name and aliases for search
            const searchTerms = [char.name];
            if (char.aliases && char.aliases.length > 0) {
                searchTerms.push(...char.aliases);
            }
            const searchTerm = searchTerms.join(",");

            const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
            const apiUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
            const res = await fetch(`${apiUrl}/api/v1/characters/chapters?name=${encodeURIComponent(searchTerm)}&sort_by=${sort}&mode=${mode}`);
            if (!res.ok) throw new Error("Failed to fetch chapters");
            const data = await res.json();
            setChapterAppearances(data);
        } catch (error) {
            console.error("Error fetching chapters:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleCharClick = async (char: Character) => {
        setSelectedChar(char);
        setIsModalOpen(true);
        setChapterAppearances([]);
        setSortBy('number'); // Default sort
        await fetchChapters(char, 'number');
    };

    const handleSortChange = async (mode: 'number' | 'relevance') => {
        if (selectedChar) {
            setSortBy(mode);
            await fetchChapters(selectedChar, mode);
        }
    };

    const handleClose = () => {
        setIsModalOpen(false);
        setTimeout(() => {
            setSelectedChar(null);
            setChapterAppearances([]);
        }, 300);
    };

    const filteredCharacters = CHARACTERS.filter(char => {
        return char.novel === selectedNovel || char.novel === 'both';
    });

    return (
        <div className="max-w-7xl mx-auto px-4 pb-20">
            <motion.div
                key={selectedNovel}
                className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6"
                initial="hidden"
                animate="show"
                variants={{
                    hidden: { opacity: 0 },
                    show: { opacity: 1 }
                }}
            >
                {filteredCharacters.map((char, idx) => (
                    <CharacterCard
                        key={char.id}
                        char={char}
                        index={idx}
                        columns={4}
                        onClick={() => handleCharClick(char)}
                        onAvatarClick={(e) => {
                            e.stopPropagation();
                            setZoomedCharName(char.name);
                        }}
                        onDescriptionClick={(e) => {
                            e.stopPropagation();
                            setZoomedDescriptionChar(char);
                        }}
                    />
                ))}
            </motion.div>

            {selectedChar && (
                <ItemModal
                    isOpen={isModalOpen}
                    onClose={handleClose}
                    title={selectedChar.name}
                    subtitle={selectedChar.role}
                    type="character"
                    chapterAppearances={chapterAppearances}
                    isLoading={loading}
                    selectedNovel={selectedNovel}
                    onNavigate={(book, chapter, sentenceIndex) => {
                        handleClose();
                        onChapterSelect?.(book, chapter, undefined, sentenceIndex);
                    }}
                    onSort={handleSortChange}
                    sortBy={sortBy}
                />
            )}

            {/* Lightbox Overlay for Character List */}
            <AnimatePresence>
                {zoomedCharName && (
                    <motion.div
                        key="list-lightbox-overlay"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 z-[60] bg-black/90 flex items-center justify-center p-4 cursor-pointer"
                        onClick={() => setZoomedCharName(null)}
                    >
                        <motion.div
                            initial={{ scale: 0.8, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.8, opacity: 0 }}
                            transition={{ type: "spring", damping: 25, stiffness: 300 }}
                            className="relative"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <CharacterAvatar
                                name={zoomedCharName}
                                size={400}
                                className="shadow-2xl border-4 border-brand-gold/50"
                                priority={true}
                            />
                            <button
                                onClick={() => setZoomedCharName(null)}
                                className="absolute -top-12 right-0 text-white/50 hover:text-white transition-colors"
                            >
                                <X size={32} />
                            </button>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Description Modal Overlay */}
            <AnimatePresence>
                {zoomedDescriptionChar && (
                    <motion.div
                        key="description-lightbox-overlay"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 z-[60] bg-black/90 flex items-center justify-center p-4 cursor-pointer"
                        onClick={() => setZoomedDescriptionChar(null)}
                    >
                        <motion.div
                            initial={{ scale: 0.8, opacity: 0, y: 20 }}
                            animate={{ scale: 1, opacity: 1, y: 0 }}
                            exit={{ scale: 0.8, opacity: 0, y: 20 }}
                            transition={{ type: "spring", damping: 25, stiffness: 300 }}
                            className="relative bg-brand-paper p-8 md:p-12 rounded-lg max-w-2xl w-full border border-brand-gold/20 shadow-2xl"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <button
                                onClick={() => setZoomedDescriptionChar(null)}
                                className="absolute top-4 right-4 text-brand-navy/50 hover:text-brand-navy transition-colors"
                            >
                                <X size={24} />
                            </button>
                            
                            <div className="flex flex-col items-center text-center">
                                <CharacterAvatar
                                    name={zoomedDescriptionChar.name}
                                    size={120}
                                    className="mb-6 shadow-lg border-2 border-brand-gold/30"
                                />
                                <h2 className="text-3xl md:text-4xl font-serif text-brand-navy font-bold mb-2">
                                    {zoomedDescriptionChar.name}
                                </h2>
                                <span className="text-sm uppercase tracking-[0.2em] text-brand-gold font-bold mb-8">
                                    {zoomedDescriptionChar.role}
                                </span>
                                
                                <div className="w-full h-px bg-gradient-to-r from-transparent via-brand-gold/30 to-transparent mb-8" />
                                
                                {/* Empty section for future detailed character biography */}
                                <div className="min-h-[200px] w-full flex items-center justify-center text-brand-navy/30">
                                </div>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

function CharacterCard({ char, onClick, onAvatarClick, onDescriptionClick, index, columns = 4 }: { char: Character; onClick: () => void; onAvatarClick: (e: React.MouseEvent) => void; onDescriptionClick: (e: React.MouseEvent) => void; index: number; columns?: number }) {
    return (
        <motion.div
            onClick={onClick}
            custom={index}
            variants={{
                hidden: { opacity: 0, y: 20 },
                show: (i: number) => ({
                    opacity: 1,
                    y: 0,
                    transition: {
                        delay: (i % columns) * 0.1,
                        duration: 0.3
                    }
                })
            }}
            whileHover={{ y: -5, boxShadow: "0 20px 40px -15px rgba(0,0,0,0.1)" }}
            transition={{ duration: 0.2 }}
            className="bg-brand-paper p-6 rounded-sm border border-brand-gold/10 hover:border-brand-gold/40 cursor-pointer group flex flex-col items-center text-center transition-all duration-300"
        >
            {/* Avatar wrapper to handle click separately */}
            <div onClick={onAvatarClick} className="relative z-10 transition-transform duration-300 hover:scale-110 hover:shadow-lg rounded-full mb-4">
                <CharacterAvatar
                    name={char.name}
                    className="group-hover:border-brand-gold transition-colors"
                    size={80}
                />
            </div>

            <h3 className="text-xl font-serif text-brand-navy font-bold">{char.name}</h3>
            <span className="text-[10px] uppercase tracking-[0.2em] text-brand-gold font-bold mt-1 mb-3">{char.role}</span>
            <motion.div
                role="button"
                tabIndex={0}
                onClick={onDescriptionClick}
                onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        onDescriptionClick(e as unknown as React.MouseEvent);
                    }
                }}
                whileHover={{ scale: 1.08, boxShadow: "0 16px 30px -18px rgba(0,0,0,0.25)" }}
                whileTap={{ scale: 0.98 }}
                transition={{ duration: 0.18 }}
                className="mt-2 mb-4 p-3 rounded-lg hover:bg-brand-gold/5 transition-all duration-300 cursor-pointer relative z-10 box-border group/desc focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-gold/60 focus-visible:ring-offset-2 focus-visible:ring-offset-brand-paper"
                aria-label={`View details for ${char.name}`}
            >
                <p className="text-sm text-brand-text-light font-body line-clamp-3 leading-relaxed relative z-10 group-hover/desc:text-brand-navy transition-colors">
                    {char.description}
                </p>
                <div className="absolute inset-0 bg-white/50 opacity-0 group-hover/desc:opacity-100 rounded-lg -z-10 transition-opacity blur-md" />
            </motion.div>

            <div className="mt-auto flex items-center gap-2 text-xs font-bold text-brand-navy/40 group-hover:text-brand-gold transition-colors uppercase tracking-widest pt-2">
                <BookOpen size={12} />
                <span>Mga Kabanata</span>
            </div>
        </motion.div >
    );
}
