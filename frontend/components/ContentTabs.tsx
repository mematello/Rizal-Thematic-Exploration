"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

type Tab = "chapters" | "characters" | "themes";

interface ContentTabsProps {
    activeTab: Tab;
    onTabChange: (tab: Tab) => void;
}

export function ContentTabs({ activeTab, onTabChange }: ContentTabsProps) {
    const tabs: { id: Tab; label: string }[] = [
        { id: "chapters", label: "Chapters" },
        { id: "characters", label: "Characters" },
        { id: "themes", label: "Themes" },
    ];

    return (
        <div className="flex items-center justify-center space-x-12 border-b border-brand-gold/30 pb-4 mb-8">
            {tabs.map((tab) => (
                <button
                    key={tab.id}
                    onClick={() => onTabChange(tab.id)}
                    className={cn(
                        "relative text-lg font-serif tracking-wide transition-colors duration-300 pb-2",
                        activeTab === tab.id
                            ? "text-brand-navy font-semibold"
                            : "text-brand-text-light hover:text-brand-navy"
                    )}
                >
                    {tab.label}
                    {activeTab === tab.id && (
                        <motion.div
                            layoutId="activeTab"
                            className="absolute bottom-0 left-0 right-0 h-0.5 bg-brand-gold"
                            transition={{ duration: 0.3 }}
                        />
                    )}
                </button>
            ))}
        </div>
    );
}
