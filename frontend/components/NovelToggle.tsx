"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { ChevronDown, BookOpen } from "lucide-react";

type Novel = "noli" | "fili";

interface NovelToggleProps {
    selected: Novel;
    onSelect: (novel: Novel) => void;
}

export function NovelToggle({ selected, onSelect }: NovelToggleProps) {
    const [isOpen, setIsOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    const getLabel = (novel: Novel) => {
        switch (novel) {
            case "noli": return "Noli Me Tangere";
            case "fili": return "El Filibusterismo";
        }
    };

    const toggleOpen = () => setIsOpen(!isOpen);

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    const options: Novel[] = ["noli", "fili"];

    return (
        <div className="relative z-50" ref={containerRef}>
            <button
                onClick={toggleOpen}
                className={cn(
                    "flex items-center gap-2 px-4 py-2 rounded-sm border transition-all duration-500 shadow-sm backdrop-blur-md",
                    "bg-white/60 hover:bg-white/90 text-brand-navy font-serif font-bold text-xs uppercase tracking-widest",
                    "border-brand-gold/10 hover:border-brand-gold/30"
                )}
            >
                <BookOpen size={16} className="text-brand-gold" />
                <span>{getLabel(selected)}</span>
                <ChevronDown
                    size={16}
                    className={cn("text-brand-navy/30 transition-transform duration-500", isOpen && "rotate-180")}
                />
            </button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.95 }}
                        transition={{ duration: 0.15 }}
                        className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl border border-brand-gold/20 overflow-hidden"
                    >
                        <div className="py-1">
                            {options.map((option) => (
                                <button
                                    key={option}
                                    onClick={() => {
                                        onSelect(option);
                                        setIsOpen(false);
                                    }}
                                    className={cn(
                                        "w-full text-left px-4 py-2 text-sm font-serif transition-colors",
                                        selected === option
                                            ? "bg-brand-gold/10 text-brand-navy font-bold"
                                            : "text-brand-text hover:bg-brand-cream hover:text-brand-navy"
                                    )}
                                >
                                    {getLabel(option)}
                                </button>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
