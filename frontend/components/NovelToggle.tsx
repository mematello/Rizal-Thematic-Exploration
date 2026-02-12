"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

type Novel = "noli" | "fili" | "both";

interface NovelToggleProps {
    selected: Novel;
    onSelect: (novel: Novel) => void;
}

export function NovelToggle({ selected, onSelect }: NovelToggleProps) {
    return (
        <div className="flex items-center justify-center p-1 bg-brand-paper rounded-full shadow-sm border border-brand-gold/20 w-fit mx-auto my-6">
            <ToggleButton
                label="Noli Me Tangere"
                isActive={selected === "noli"}
                onClick={() => onSelect("noli")}
            />
            <ToggleButton
                label="Both"
                isActive={selected === "both"}
                onClick={() => onSelect("both")}
            />
            <ToggleButton
                label="El Filibusterismo"
                isActive={selected === "fili"}
                onClick={() => onSelect("fili")}
            />
        </div>
    );
}

function ToggleButton({
    label,
    isActive,
    onClick,
}: {
    label: string;
    isActive: boolean;
    onClick: () => void;
}) {
    return (
        <button
            onClick={onClick}
            className={cn(
                "relative px-6 py-2 text-sm font-serif font-medium transition-colors duration-300 rounded-full z-10",
                isActive ? "text-brand-cream" : "text-brand-text hover:text-brand-gold"
            )}
        >
            {isActive && (
                <motion.div
                    layoutId="activeNovel"
                    className="absolute inset-0 bg-brand-navy rounded-full -z-10"
                    transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
            )}
            {label}
        </button>
    );
}
