"use client";

import { useEffect, useRef, useState } from "react";
import { Search, Loader2, Zap, Clock } from "lucide-react";
import { SearchBarProps, Suggestion } from "../types";

export function SearchBar({
    variant = 'hero',
    defaultValue = '',
    placeholder = 'Maghanap ng tema, tauhan, o salita...',
    isLoading = false,
    onSearch,
    showSuggestions = true,
}: SearchBarProps) {
    const [query, setQuery] = useState(defaultValue);
    const [isOpen, setIsOpen] = useState(false);
    const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
    const wrapperRef = useRef<HTMLDivElement>(null);

    // Mock suggestions (replace with API call)
    const MOCK_SUGGESTIONS: Suggestion[] = [
        { text: 'Edukasyon bilang susi', type: 'semantic' },
        { text: 'Katarungan para kay Sisa', type: 'semantic' },
        { text: "Simoun's jewelry", type: 'lexical' },
        { text: 'Padre Damaso', type: 'recent' },
    ];

    // Close dropdown when clicking outside
    useEffect(() => {
        function handleClickOutside(event: MouseEvent) {
            if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        }
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    // Load suggestions when typing (debounced)
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

    const heightClass = variant === 'hero' ? 'h-14' : 'h-12';
    const textClass = variant === 'hero' ? 'text-lg' : 'text-base';

    return (
        <div ref={wrapperRef} className="relative w-full max-w-2xl mx-auto z-50">
            {/* Input Field */}
            <div
                className={`
          relative flex items-center bg-white rounded-full
          border-2 border-brand-brown ${heightClass} shadow-md
          transition-all duration-200
          focus-within:ring-4 focus-within:ring-brand-brown/10
          focus-within:border-brand-blue
        `}
            >
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
                    className={`
            w-full h-full bg-transparent px-6 rounded-full
            focus:outline-none font-roboto ${textClass}
            text-brand-text placeholder:text-gray-400 placeholder:italic
          `}
                    aria-label="Search Rizal's novels"
                    aria-autocomplete="list"
                    aria-controls="search-suggestions"
                    aria-expanded={isOpen && suggestions.length > 0}
                />

                {/* Icon / Spinner */}
                <div className="absolute right-4">
                    {isLoading ? (
                        <Loader2 className="animate-spin text-brand-brown" size={20} />
                    ) : (
                        <button
                            onClick={() => handleSubmit(query)}
                            className="
                p-2 hover:bg-gray-100 rounded-full transition-colors
                focus:outline-none focus:ring-2 focus:ring-brand-blue
              "
                            aria-label="Search"
                        >
                            <Search size={20} className="text-brand-brown" />
                        </button>
                    )}
                </div>
            </div>

            {/* Type-ahead Dropdown */}
            {isOpen && suggestions.length > 0 && (
                <div
                    id="search-suggestions"
                    role="listbox"
                    className="
            absolute top-16 left-0 right-0 bg-white rounded-xl
            shadow-xl border border-brand-brown/10 overflow-hidden
            py-2 animate-in fade-in slide-in-from-top-2 duration-200
          "
                >
                    <div className="px-4 py-2 text-xs font-bold text-gray-400 uppercase tracking-wider">
                        Suggestions
                    </div>
                    {suggestions.map((item, idx) => (
                        <button
                            key={idx}
                            role="option"
                            onClick={() => {
                                setQuery(item.text);
                                handleSubmit(item.text);
                            }}
                            className="
                w-full text-left px-4 py-3 hover:bg-brand-cream
                flex items-center gap-3 transition-colors
                focus:outline-none focus:bg-brand-cream
              "
                        >
                            {item.type === 'semantic' ? (
                                <Zap size={16} className="text-semantic-teal" aria-hidden="true" />
                            ) : item.type === 'recent' ? (
                                <Clock size={16} className="text-gray-400" aria-hidden="true" />
                            ) : (
                                <Search size={16} className="text-gray-400" aria-hidden="true" />
                            )}
                            <span className="font-crimson text-lg text-brand-text">{item.text}</span>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}
