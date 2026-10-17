"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { ChevronDown, BookOpen } from "lucide-react";

type Novel = "noli" | "fili" | "both";

interface NovelToggleProps {
    selected: Novel;
    onSelect: (novel: any) => void;
    includeBoth?: boolean;
    className?: string;
}

export function NovelToggle({ selected, onSelect, includeBoth = false, className }: NovelToggleProps) {
    const [isOpen, setIsOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    const getLabel = (novel: Novel) => {
        switch (novel) {
            case "both": return "Lahat ng Nobela";
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

    const options: Novel[] = includeBoth ? ["both", "noli", "fili"] : ["noli", "fili"];

    return (
        <div className={cn("relative z-40", className)} ref={containerRef}>
            <button
                type="button"
                onClick={toggleOpen}
                className={cn(
                    "h-10 min-h-[40px] flex items-center gap-2 px-3.5 py-1.5 rounded-full border transition-all duration-200 shadow-xs backdrop-blur-md active:scale-95 font-serif font-bold text-xs uppercase tracking-wider focus:outline-none focus-visible:ring-2",
                    selected === "fili"
                        ? "bg-white/80 dark:bg-[#201214] hover:bg-white/95 dark:hover:bg-[#2C181B] text-[#8D2D2D] dark:text-[#E06C75] border-[#8D2D2D]/20 dark:border-[#E06C75]/30 hover:border-[#8D2D2D]/40 focus-visible:ring-[#8D2D2D]"
                        : "bg-white/80 dark:bg-[#182232] hover:bg-white/95 dark:hover:bg-[#202C3F] text-brand-navy dark:text-brand-gold border-brand-gold/20 dark:border-brand-gold/30 hover:border-brand-gold/40 focus-visible:ring-brand-gold"
                )}
                aria-expanded={isOpen}
                aria-label="Piliin ang nobela"
            >
                <BookOpen size={15} className={cn("shrink-0", selected === "fili" ? "text-[#8D2D2D] dark:text-[#E06C75]" : "text-brand-gold")} />
                <span className="truncate max-w-[140px] sm:max-w-none">{getLabel(selected)}</span>
                <ChevronDown
                    size={15}
                    className={cn("transition-transform duration-200 shrink-0", selected === "fili" ? "text-[#8D2D2D]/60 dark:text-[#E06C75]/70" : "text-brand-navy/50 dark:text-brand-gold/60", isOpen && "rotate-180")}
                />
            </button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.95 }}
                        transition={{ duration: 0.15 }}
                        className="absolute right-0 mt-2 w-52 bg-white/98 dark:bg-brand-paper/98 rounded-xl shadow-xl border border-brand-gold/20 overflow-hidden z-50 backdrop-blur-md"
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
                                            ? "bg-brand-gold/15 dark:bg-brand-gold/25 text-brand-navy dark:text-brand-gold font-black"
                                            : "text-brand-text/80 dark:text-brand-text hover:bg-brand-cream hover:text-brand-navy dark:hover:text-brand-gold"
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
