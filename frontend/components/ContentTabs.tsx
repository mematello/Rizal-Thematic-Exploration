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
        <div className="flex items-center justify-center gap-4 sm:gap-8 md:gap-12 border-b border-brand-gold/10 pb-2 sm:pb-4 mb-6 sm:mb-8 overflow-x-auto no-scrollbar">
            {tabs.map((tab) => (
                <button
                    key={tab.id}
                    type="button"
                    onClick={() => onTabChange(tab.id)}
                    className={cn(
                        "relative min-h-[44px] px-2 sm:px-3 flex items-center justify-center font-serif tracking-[0.08em] transition-all duration-300 uppercase text-xs sm:text-sm font-bold active:scale-95",
                        activeTab === tab.id
                            ? "text-brand-navy font-black"
                            : "text-brand-text/50 hover:text-brand-navy"
                    )}
                    aria-selected={activeTab === tab.id}
                    role="tab"
                >
                    <span className="pb-1">{tab.label}</span>
                    {activeTab === tab.id && (
                        <motion.div
                            layoutId="activeTab"
                            className="absolute bottom-0 left-0 right-0 h-0.5 bg-brand-gold"
                            transition={{ type: "spring", damping: 15, stiffness: 200 }}
                        />
                    )}
                </button>
            ))}
        </div>
    );
}
