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
                type="button"
                onClick={toggleOpen}
                className={cn(
                    "min-h-[40px] flex items-center gap-2 px-4 py-2 rounded-full md:rounded-sm border transition-all duration-300 shadow-sm backdrop-blur-md active:scale-95",
                    "bg-white/80 hover:bg-white/95 text-brand-navy font-serif font-bold text-xs uppercase tracking-widest",
                    "border-brand-gold/20 hover:border-brand-gold/40"
                )}
                aria-expanded={isOpen}
                aria-label="Piliin ang nobela"
            >
                <BookOpen size={16} className="text-brand-gold" />
                <span>{getLabel(selected)}</span>
                <ChevronDown
                    size={16}
                    className={cn("text-brand-navy/50 transition-transform duration-300", isOpen && "rotate-180")}
                />
            </button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.95 }}
                        transition={{ duration: 0.15 }}
                        className="absolute right-0 mt-2 w-52 bg-white/98 rounded-xl shadow-xl border border-brand-gold/20 overflow-hidden z-50 backdrop-blur-md"
                    >
                        <div className="py-1">
                            {options.map((option) => (
                                <button
                                    key={option}
                                    type="button"
                                    onClick={() => {
                                        onSelect(option);
                                        setIsOpen(false);
                                    }}
                                    className={cn(
                                        "w-full min-h-[44px] text-left px-4 py-2.5 text-xs uppercase font-serif tracking-wider transition-colors flex items-center",
                                        selected === option
                                            ? "bg-brand-gold/15 text-brand-navy font-black"
                                            : "text-brand-text/80 hover:bg-brand-cream hover:text-brand-navy"
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
