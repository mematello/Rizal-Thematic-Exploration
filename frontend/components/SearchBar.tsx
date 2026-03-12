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
                    className="w-full py-3 bg-transparent border-b-2 border-brand-gold/20 text-brand-text font-serif text-xl placeholder:text-brand-text-light/30 focus:outline-none focus:border-brand-gold transition-all duration-500"
                    aria-label="Manaliksik sa mga nobela ni Rizal"
                />

                <div className="absolute right-0 top-1/2 -translate-y-1/2 flex items-center pr-2">
                    {query.length > 0 && (
                        <button
                            onClick={handleClear}
                            className="p-1 text-brand-text-light hover:text-brand-navy transition-colors mr-2"
                        >
                            <X size={16} />
                        </button>
                    )}

                    {isLoading ? (
                        <Loader2 className="animate-spin text-brand-gold" size={24} />
                    ) : (
                        <Search className="text-brand-gold/50 group-focus-within:text-brand-gold transition-colors duration-500" size={24} />
                    )}
                </div>
            </div>

            {isOpen && suggestions.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-2 bg-white/80 backdrop-blur-xl rounded-sm shadow-lg border border-brand-gold/20 overflow-hidden py-1 animate-in fade-in slide-in-from-top-2">
                    {suggestions.map((item, idx) => (
                        <button
                            key={idx}
                            onClick={() => {
                                setQuery(item.text);
                                handleSubmit(item.text);
                            }}
                            className="w-full text-left px-4 py-3 hover:bg-brand-gold/10 flex items-center gap-3 transition-colors group"
                        >
                            {item.type === 'semantic' ? (
                                <Zap size={14} className="text-brand-gold group-hover:text-brand-navy" />
                            ) : item.type === 'recent' ? (
                                <Clock size={14} className="text-brand-text-light" />
                            ) : (
                                <Search size={14} className="text-brand-text-light" />
                            )}
                            <span className="font-serif text-brand-text">{item.text}</span>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}
