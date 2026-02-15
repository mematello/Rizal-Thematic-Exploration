
"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { User, BookOpen } from "lucide-react";
import { ItemModal } from "@/components/ItemModal";

interface Character {
    id: string;
    name: string;
    role: string;
    novel: "noli" | "fili" | "both";
    description: string;
    aliases?: string[];
}

interface Appearance {
    book: string;
    chapter_number: number;
    chapter_title: string;
    sentence_text: string;
    sentence_index: number;
}


const CHARACTERS: Character[] = [
    { id: "1", name: "Crisostomo Ibarra", role: "Pangunahing Tauhan", novel: "noli", description: "Isang mayamang binata na nagbalik mula sa Europa upang magpatayo ng paaralan para sa kanyang mga kababayan.", aliases: ["Crisostomo Ibarra", "Ibarra", "Crisostomo", "Simoun"] },
    { id: "2", name: "Maria Clara", role: "Kasintahan", novel: "noli", description: "Ang kasintahan ni Ibarra at anak-anakan ni Kapitan Tiago; simbolo ng dalagang Pilipina.", aliases: ["Maria Clara", "Maria", "Clara", "Sor Maria Clara"] },
    { id: "3", name: "Simoun", role: "Pangunahing Tauhan", novel: "fili", description: "Ang mayamang mag-aalahas na nagbabalat-kayo; siya si Ibarra na nagbalik upang maghimagsik.", aliases: ["Simoun", "Crisostomo Ibarra", "Ibarra", "Crisostomo"] },
    { id: "4", name: "Basilio", role: "Pangalawang Tauhan", novel: "both", description: "Ang anak ni Sisa na nagsikap mag-aral at naging isang manggagamot.", aliases: ["Basilio"] },
    { id: "5", name: "Padre Damaso", role: "Kontrabida", novel: "noli", description: "Ang dating kura ng San Diego at tunay na ama ni Maria Clara; mapagmataas at malupit.", aliases: ["Padre Damaso", "Damaso"] },
    { id: "6", name: "Elias", role: "Bayani", novel: "noli", description: "Isang takas na naging kaibigan at tagapagligtas ni Ibarra; nagsakripisyo para sa bayan.", aliases: ["Elias"] },
    { id: "7", name: "Kabesang Tales", role: "Biktima", novel: "fili", description: "Isang masipag na magsasaka na naging tulisan dahil sa pang-aagaw ng lupa ng mga prayle.", aliases: ["Kabesang Tales", "Tales", "Kabesa"] },
    { id: "8", name: "Isagani", role: "Idealista", novel: "fili", description: "Isang makatang mag-aaral at kasintahan ni Paulita Gomez; puno ng pangarap para sa bayan.", aliases: ["Isagani"] },
];


interface CharacterListProps {
    onChapterSelect?: (book: string, chapter: number, title?: string, sentenceIndex?: number) => void;
}

interface ChapterInfo {
    book: string;
    chapter_number: number;
    chapter_title: string;
    score: number;
    preview_text?: string;
}

export function CharacterList({ onChapterSelect }: CharacterListProps) {
    const [selectedChar, setSelectedChar] = useState<Character | null>(null);
    const [chapterAppearances, setChapterAppearances] = useState<ChapterInfo[]>([]);
    const [loading, setLoading] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [sortBy, setSortBy] = useState<'number' | 'relevance'>('number');

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

    return (
        <div className="max-w-7xl mx-auto px-4 pb-20">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                {CHARACTERS.map((char) => (
                    <motion.div
                        key={char.id}
                        onClick={() => handleCharClick(char)}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        whileHover={{ scale: 1.02 }}
                        transition={{ duration: 0.2 }}
                        className="bg-brand-paper p-6 rounded-sm border border-brand-gold/20 hover:border-brand-navy/30 cursor-pointer group flex flex-col items-center text-center transition-colors"
                    >
                        <div className="w-20 h-20 bg-brand-cream rounded-full flex items-center justify-center mb-4 border-2 border-brand-gold/10 group-hover:border-brand-gold transition-colors">
                            <User size={32} className="text-brand-navy opacity-80" />
                        </div>

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
                ))}
            </div>

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
