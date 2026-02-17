
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
    // Noli Me Tangere
    { id: "1", name: "Crisostomo Ibarra", role: "Protagonist", novel: "noli", description: "A wealthy young man who returned from Europe to build a school for his countrymen.", aliases: ["Juan Crisostomo Ibarra y Magsalin", "Ibarra", "Juan Ibarra", "Crisostomo Ibarra", "Juan Crisostomo", "Crisostomo", "Juan", "Ibarra y Magsalin", "Juan Crisostomo Magsalin", "Crisostomo Magsalin"] },
    { id: "2", name: "Maria Clara", role: "Protagonist", novel: "noli", description: "Ibarra's fiancée and pests Captain Tiago's daughter; symbol of the Filipina woman.", aliases: ["Maria Clara de los Santos y Alba", "Maria Clara", "Clara", "Maria Clara Santos", "Maria Clara Alba", "Maria Clara de los Santos", "Maria", "Clara de los Santos"] },
    { id: "3", name: "Kapitan Tiago", role: "Influential Figure", novel: "noli", description: "A wealthy landowner and influential figure in San Diego.", aliases: ["Don Santiago de los Santos", "Don Santiago", "Santiago", "Kapitan Tiago", "Tiago", "Santiago de los Santos"] },
    { id: "4", name: "Pia Alba", role: "Mother", novel: "noli", description: "Maria Clara's mother who died in childbirth.", aliases: ["Pia Alba", "Pia"] },
    { id: "5", name: "Padre Damaso", role: "Antagonist", novel: "noli", description: "The former curate of San Diego and Maria Clara's biological father; arrogant and cruel.", aliases: ["Padre Damaso Verdolagas", "Padre Damaso", "Damaso Verdolagas", "Damaso", "Verdolagas"] },
    { id: "6", name: "Padre Salvi", role: "Curate", novel: "noli", description: "The curate who replaced Padre Damaso; secretly lusts after Maria Clara.", aliases: ["Padre Bernardo Salvi", "Padre Salvi", "Bernardo Salvi", "Salvi", "Padre Bernardo"] },
    { id: "7", name: "Elias", role: "Revolutionary", novel: "noli", description: "A fugitive who became Ibarra's friend and savior; sacrificed for the country.", aliases: ["Elias"] },
    { id: "8", name: "Pilosopo Tasio", role: "Wise Old Man", novel: "noli", description: "A wise man often dismissed as crazy by the uneducated.", aliases: ["Pilosopo Tasio", "Tasio", "Don Anastacio", "Anastacio"] },
    { id: "9", name: "Dona Victorina", role: "Social Climber", novel: "both", description: "A woman who tries to be more Spanish than the Spaniards.", aliases: ["Dona Victorina de los Reyes de Espadana", "Dona Victorina", "Victorina", "Victorina de los Reyes", "Victorina de Espadana", "Doña Victorina"] },
    { id: "10", name: "Don Tiburcio", role: "Husband", novel: "noli", description: "Dona Victorina's submissive husband.", aliases: ["Don Tiburcio de Espadana", "Don Tiburcio", "Tiburcio", "Tiburcio de Espadana"] },
    { id: "11", name: "Sisa", role: "Mother", novel: "noli", description: "A tragic mother who went mad searching for her sons.", aliases: ["Sisa", "Narcisa", "Sisa Narcisa", "Narcisa Sisa"] },
    { id: "12", name: "Basilio", role: "Son/Student", novel: "both", description: "Sisa's son who studied hard to become a doctor.", aliases: ["Basilio"] },
    { id: "13", name: "Crispin", role: "Son", novel: "noli", description: "Basilio's younger brother, falsely accused of theft.", aliases: ["Crispin", "Crispin"] },
    { id: "14", name: "Don Rafael Ibarra", role: "Father", novel: "noli", description: "Crisostomo Ibarra's father who died in prison.", aliases: ["Don Rafael Ibarra", "Don Rafael", "Rafael Ibarra", "Rafael"] },
    { id: "15", name: "Don Saturnino", role: "Ancestor", novel: "noli", description: "Ibarra's ancestor.", aliases: ["Don Saturnino", "Saturnino"] },
    { id: "16", name: "Alperes", role: "Official", novel: "noli", description: "Head of the Guardia Civil.", aliases: ["Alperes"] },
    { id: "17", name: "Donya Consolacion", role: "Wife", novel: "noli", description: "The Alperes' abusive wife.", aliases: ["Donya Consolacion", "Consolacion", "Donya"] },
    { id: "18", name: "Teniente Guevarra", role: "Official", novel: "noli", description: "An honest lieutenant of the Guardia Civil.", aliases: ["Teniente Guevarra", "Guevarra", "Lieutenant Guevarra"] },
    { id: "19", name: "Nol Juan", role: "Foreman", novel: "noli", description: "Overseer of Ibarra's school construction.", aliases: ["Nol Juan", "Juan"] },
    { id: "20", name: "Lucas", role: "Conspirator", novel: "noli", description: "A man involved in the plot against Ibarra.", aliases: ["Lucas"] },
    { id: "21", name: "Albino", role: "Seminarian", novel: "noli", description: "A former seminarian.", aliases: ["Albino"] },

    // El Filibusterismo
    { id: "22", name: "Simoun", role: "Protagonist", novel: "fili", description: "The wealthy jeweler who is actually Ibarra in disguise.", aliases: ["Simoun", "Simoun Ibarra", "Crisostomo Ibarra", "Crisostomo", "Ibarra", "Juan Crisostomo Ibarra", "Juan Ibarra"] },
    { id: "23", name: "Isagani", role: "Student/Idealist", novel: "fili", description: "A poet and student with great love for his country.", aliases: ["Isagani"] },
    { id: "24", name: "Kabesang Tales", role: "Victim/Rebel", novel: "fili", description: "A hardworking farmer turned bandit due to injustice.", aliases: ["Kabesang Tales", "Tales", "Kabe", "Ka-Tales"] },
    { id: "25", name: "Paulita Gomez", role: "Student's Love", novel: "fili", description: "A rich heiress and Isagani's sweetheart.", aliases: ["Paulita Gomez", "Paulita", "Gomez"] },
    { id: "26", name: "Juanito Pelaez", role: "Student", novel: "fili", description: "A hunchbacked student and rival of Isagani.", aliases: ["Juanito Pelaez", "Juanito", "Pelaez"] },
    { id: "27", name: "Juli", role: "Daughter", novel: "fili", description: "Kabesang Tales' daughter and Basilio's sweetheart.", aliases: ["Juli"] },
    { id: "28", name: "Padre Florentino", role: "Priest", novel: "fili", description: "A patriotic native priest.", aliases: ["Padre Florentino", "Florentino", "Padre F"] },
    { id: "29", name: "Don Custodio", role: "Official", novel: "fili", description: "A government official known as 'Buena Tinta'.", aliases: ["Don Custodio", "Custodio"] },
    { id: "30", name: "Padre Camorra", role: "Priest", novel: "fili", description: "A lustful friar.", aliases: ["Padre Camorra", "Camorra"] },
    { id: "31", name: "Padre Irene", role: "Priest", novel: "fili", description: "Kapitan Tiago's executor and ally of the students.", aliases: ["Padre Irene", "Irene"] },
    { id: "32", name: "Ben-Zayb", role: "Journalist", novel: "fili", description: "A journalist who distorts the truth.", aliases: ["Ben-Zayb", "Benzayb"] },
    { id: "33", name: "Placido Penitente", role: "Student", novel: "fili", description: "A student who becomes disillusioned with the university.", aliases: ["Placido Penitente", "Placido", "Penitente"] },
    { id: "34", name: "Father Fernandez", role: "Priest", novel: "fili", description: "A Dominican professor who tries to understand students.", aliases: ["Father Fernandez", "Fernandez"] },
    { id: "35", name: "Tandang Selo", role: "Grandfather", novel: "fili", description: "Kabesang Tales' father.", aliases: ["Tandang Selo", "Selo", "Tandang"] },
    { id: "36", name: "Quiroga", role: "Merchant", novel: "fili", description: "A Chinese merchant aspiring to be a consul.", aliases: ["Quiroga"] },
    { id: "37", name: "Hermana Penchang", role: "Devotee", novel: "fili", description: "Juli's pious employer.", aliases: ["Hermana Penchang", "Penchang"] },
    { id: "38", name: "Hermana Bali", role: "Gambler", novel: "fili", description: "A gambler who advises Juli.", aliases: ["Hermana Bali", "Bali"] },
    { id: "39", name: "Father Millon", role: "Professor", novel: "fili", description: "Physics professor.", aliases: ["Father Millon", "Millon"] },
    { id: "40", name: "Tadeo", role: "Student", novel: "fili", description: "A student who rejoices when classes are suspended.", aliases: ["Tadeo"] },
    { id: "41", name: "Mr. Leeds", role: "Showman", novel: "fili", description: "An American showman.", aliases: ["Leeds"] },
    { id: "42", name: "Tano", role: "Son/Guard", novel: "fili", description: "Kabesang Tales' son who became a guard.", aliases: ["Tano"] },
    { id: "43", name: "Pepay", role: "Dancer", novel: "fili", description: "A dancer and Don Custodio's mistress.", aliases: ["Pepay"] },
    { id: "44", name: "Pecson", role: "Student", novel: "fili", description: "A skeptical student.", aliases: ["Pecson"] },
];


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

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                {filteredCharacters.map((char) => (
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
