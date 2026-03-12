import { Sparkles, Search } from "lucide-react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";

interface SuggestionsCardProps {
    suggestions: string[];
    onSuggestionClick?: (query: string) => void;
}

export const SuggestionsCard = ({ suggestions, onSuggestionClick }: SuggestionsCardProps) => {
    const router = useRouter();

    if (!suggestions || suggestions.length === 0) return null;

    const handleSuggestionClick = (query: string) => {
        if (onSuggestionClick) {
            onSuggestionClick(query);
        } else {
            const encodedQuery = encodeURIComponent(query);
            router.push(`/search?q=${encodedQuery}`);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 mb-4 mx-auto w-full max-w-2xl bg-white border border-brand-red/10 rounded-xl overflow-hidden shadow-sm"
        >
            <div className="bg-brand-red/5 px-4 py-3 border-b border-brand-red/10 flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-brand-red" />
                <h3 className="font-semibold text-brand-dark/90 text-sm tracking-wide">
                    Kaugnay na Paghahanap
                </h3>
            </div>
            <div className="p-4 bg-white/50">
                <div className="flex flex-wrap gap-2">
                    {suggestions.map((sug, idx) => (
                        <button
                            key={idx}
                            onClick={() => handleSuggestionClick(sug)}
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-brand-gold/10 hover:bg-brand-gold/20 
                                     border border-brand-gold/20 hover:border-brand-gold/40
                                     text-sm text-brand-dark/80 rounded-full transition-colors font-medium
                                     focus:outline-none focus:ring-2 focus:ring-brand-gold/50"
                        >
                            <Search className="h-3 w-3 text-brand-gold/70" />
                            <span className="capitalize">{sug}</span>
                        </button>
                    ))}
                </div>
            </div>
        </motion.div>
    );
};
