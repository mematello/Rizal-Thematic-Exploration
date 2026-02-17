"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { BookOpen } from "lucide-react";
import { ItemModal } from "@/components/ItemModal";
import { CharacterAvatar } from "@/components/CharacterAvatar";
import { CHARACTERS, Character } from "@/lib/characterData";

interface Appearance {
    book: string;
    chapter_number: number;
    chapter_title: string;
    sentence_text: string;
    sentence_index: number;
}


interface CharacterListProps {
    onChapterSelect?: (book: string, chapter: number, title?: string, sentenceIndex?: number) => void;
    selectedNovel: "noli" | "fili" | "both";
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
    // Removed local novelFilter state

    const fetchChapters = async (char: Character, sort: 'number' | 'relevance') => {
        setLoading(true);
        try {
            // Use aliases if available, otherwise fallback to name logic
            let searchTerm = char.name;
            if (char.aliases && char.aliases.length > 0) {
                searchTerm = char.aliases.join(",");
            } else {
                // Fallbacks if no aliases defined (though we defined them above)
                if (char.name === "Crisostomo Ibarra") searchTerm = "Ibarra";
                else if (char.name === "Padre Damaso") searchTerm = "Damaso";
            }

            const res = await fetch(`http://localhost:8000/api/v1/characters/chapters?name=${encodeURIComponent(searchTerm)}&sort_by=${sort}`);
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
        if (selectedNovel === 'both') return true;
        return char.novel === selectedNovel || char.novel === 'both';
    });

    return (
        <div className="max-w-7xl mx-auto px-4 pb-20">
            {/* Filter Buttons Removed - using global filter */}

            {selectedNovel === 'both' ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Noli Side */}
                    <div className="space-y-4">
                        <motion.div
                            className="grid grid-cols-1 sm:grid-cols-2 gap-6"
                            initial="hidden"
                            animate="show"
                            variants={{
                                hidden: { opacity: 0 },
                                show: {
                                    opacity: 1,
                                    transition: { staggerChildren: 0.1 }
                                }
                            }}
                        >
                            {filteredCharacters
                                .filter(c => c.novel === 'noli' || c.novel === 'both')
                                .map((char) => (
                                    <CharacterCard
                                        key={`noli-${char.id}`}
                                        char={char}
                                        onClick={() => handleCharClick(char)}
                                    />
                                ))}
                        </motion.div>
                    </div>

                    {/* Fili Side */}
                    <div className="space-y-4">
                        <motion.div
                            className="grid grid-cols-1 sm:grid-cols-2 gap-6"
                            initial="hidden"
                            animate="show"
                            variants={{
                                hidden: { opacity: 0 },
                                show: {
                                    opacity: 1,
                                    transition: { staggerChildren: 0.1 }
                                }
                            }}
                        >
                            {filteredCharacters
                                .filter(c => c.novel === 'fili' || c.novel === 'both')
                                .map((char) => (
                                    <CharacterCard
                                        key={`fili-${char.id}`}
                                        char={char}
                                        onClick={() => handleCharClick(char)}
                                    />
                                ))}
                        </motion.div>
                    </div>
                </div>
            ) : (
                <motion.div
                    key={selectedNovel}
                    className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6"
                    initial="hidden"
                    animate="show"
                    variants={{
                        hidden: { opacity: 0 },
                        show: {
                            opacity: 1,
                            transition: { staggerChildren: 0.1 }
                        }
                    }}
                >
                    {filteredCharacters.map((char) => (
                        <CharacterCard
                            key={char.id}
                            char={char}
                            onClick={() => handleCharClick(char)}
                        />
                    ))}
                </motion.div>
            )}

            {selectedChar && (
                <ItemModal
                    isOpen={isModalOpen}
                    onClose={handleClose}
                    title={selectedChar.name}
                    subtitle={selectedChar.role}
                    type="character"
                    chapterAppearances={chapterAppearances}
                    isLoading={loading}
                    onNavigate={(book, chapter, sentenceIndex) => {
                        handleClose();
                        onChapterSelect?.(book, chapter, undefined, sentenceIndex);
                    }}
                    onSort={handleSortChange}
                    sortBy={sortBy}
                />
            )}
        </div>
    );
}

function CharacterCard({ char, onClick }: { char: Character; onClick: () => void }) {
    return (
        <motion.div
            onClick={onClick}
            variants={{
                hidden: { opacity: 0, y: 20 },
                show: { opacity: 1, y: 0 }
            }}
            whileHover={{ scale: 1.02 }}
            transition={{ duration: 0.2 }}
            className="bg-brand-paper p-6 rounded-sm border border-brand-gold/20 hover:border-brand-navy/30 cursor-pointer group flex flex-col items-center text-center transition-colors"
        >
            <CharacterAvatar
                name={char.name}
                className="mb-4 group-hover:border-brand-gold transition-colors"
                size={80}
            />

            <h3 className="text-xl font-serif text-brand-navy font-bold">{char.name}</h3>
            <span className="text-xs uppercase tracking-widest text-brand-gold mt-1 mb-3">{char.role}</span>
            <p className="text-sm text-brand-text-light font-body line-clamp-3 leading-relaxed">
                {char.description}
            </p>

            <div className="mt-4 flex items-center gap-2 text-xs font-bold text-brand-navy/60 group-hover:text-brand-navy transition-colors">
                <BookOpen size={12} />
                <span>View Chapters</span>
            </div>
        </motion.div>
    );
}
