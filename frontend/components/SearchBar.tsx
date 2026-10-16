"use client";

import { useEffect, useRef, useState } from "react";
import { Search, Loader2, Zap, Clock, X } from "lucide-react";
import { SearchBarProps, Suggestion } from "../types";

export function SearchBar({
    variant = 'hero',
    defaultValue = '',
    placeholder = 'Search for themes, characters, or passages...',
    isLoading = false,
    onSearch,
    showSuggestions = true,
}: SearchBarProps) {
    const [query, setQuery] = useState(defaultValue);
    const [isOpen, setIsOpen] = useState(false);
    const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
    const wrapperRef = useRef<HTMLDivElement>(null);

    // Mock suggestions (replace with API call later)
    const MOCK_SUGGESTIONS: Suggestion[] = [
        { text: 'Edukasyon', type: 'semantic' },
        { text: 'Justicia', type: 'semantic' },
        { text: "Simoun's Lamp", type: 'lexical' },
        { text: 'Crisostomo Ibarra', type: 'recent' },
    ];

    useEffect(() => {
        setQuery(defaultValue);
    }, [defaultValue]);

    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        }
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    useEffect(() => {
        if (query.length >= 3 && showSuggestions) {
            const timer = setTimeout(() => {
                setSuggestions(MOCK_SUGGESTIONS);
            }, 300);
            return () => clearTimeout(timer);
        } else {
            setSuggestions([]);
        }
    }, [query, showSuggestions]);

    const handleSubmit = (searchQuery: string) => {
        if (searchQuery.length >= 3) {
            onSearch(searchQuery);
            setIsOpen(false);
        }
    };

    const handleClear = () => {
        setQuery("");
        onSearch("");
        setIsOpen(false);
    };

    return (
        <div ref={wrapperRef} className="relative w-full max-w-xl mx-auto z-50">
            <div className="relative group">
                <input
                    type="search"
                    inputMode="search"
                    enterKeyHint="search"
                    value={query}
                    onChange={(e) => {
                        setQuery(e.target.value);
                        setIsOpen(true);
                    }}
                    onFocus={() => setIsOpen(true)}
                    onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                            e.preventDefault();
                            handleSubmit(query);
                        }
                    }}
                    placeholder={placeholder}
                    className="w-full py-2.5 sm:py-3 min-h-[44px] bg-transparent border-b-2 border-brand-gold/30 dark:border-brand-gold/40 text-brand-navy dark:text-foreground font-serif text-base sm:text-xl placeholder:text-brand-text-light/50 dark:placeholder:text-brand-text-muted/60 focus:outline-none focus:border-brand-gold transition-all duration-500"
                    aria-label="Manaliksik sa mga nobela ni Rizal"
                />

                <div className="absolute right-0 top-1/2 -translate-y-1/2 flex items-center">
                    {query.length > 0 && (
                        <button
                            type="button"
                            onClick={handleClear}
                            className="min-w-[44px] min-h-[44px] flex items-center justify-center text-brand-text-light dark:text-brand-text-muted hover:text-brand-navy dark:hover:text-foreground active:scale-95 transition-all"
                            aria-label="Burahin ang paghahanap"
                        >
                            <X size={18} />
                        </button>
                    )}

                    <button
                        type="button"
                        onClick={() => handleSubmit(query)}
                        className="min-w-[44px] min-h-[44px] flex items-center justify-center text-brand-gold hover:text-brand-accent-hover active:scale-95 transition-all"
                        aria-label="Simulan ang paghahanap"
                    >
                        {isLoading ? (
                            <Loader2 className="animate-spin text-brand-gold" size={22} />
                        ) : (
                            <Search className="group-focus-within:text-brand-gold text-brand-gold/70 transition-colors duration-500" size={22} />
                        )}
                    </button>
                </div>
            </div>

            {isOpen && suggestions.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-2 bg-white/95 dark:bg-brand-paper/98 backdrop-blur-xl rounded-sm shadow-xl border border-brand-gold/20 dark:border-brand-gold/40 overflow-hidden py-1 animate-in fade-in slide-in-from-top-2">
                    {suggestions.map((item, idx) => (
                        <button
                            key={idx}
                            onClick={() => {
                                setQuery(item.text);
                                handleSubmit(item.text);
                            }}
                            className="w-full text-left px-4 py-3 hover:bg-brand-gold/10 dark:hover:bg-brand-gold/15 flex items-center gap-3 transition-colors group"
                        >
                            {item.type === 'semantic' ? (
                                <Zap size={14} className="text-brand-gold group-hover:text-brand-navy dark:group-hover:text-brand-gold" />
                            ) : item.type === 'recent' ? (
                                <Clock size={14} className="text-brand-text-light dark:text-brand-text-muted" />
                            ) : (
                                <Search size={14} className="text-brand-text-light dark:text-brand-text-muted" />
                            )}
                            <span className="font-serif text-brand-text dark:text-foreground">{item.text}</span>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}
