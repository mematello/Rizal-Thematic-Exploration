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
    <div
      role="tablist"
      aria-label="Mga seksyon ng nilalaman"
      className="w-full max-w-xl mx-auto mb-6 sm:mb-8 px-2 sm:px-0 min-w-0"
    >
      {/* Mobile: Full-width segmented control with 3 equal columns (min-h-[44px]) */}
      {/* Desktop (sm+): Elegant underline tab bar */}
      <div className="grid grid-cols-3 p-1 rounded-xl bg-brand-gold/10 dark:bg-black/40 border border-brand-gold/20 dark:border-brand-gold/30 sm:bg-transparent sm:dark:bg-transparent sm:border-0 sm:border-b sm:border-brand-gold/20 sm:p-0 sm:flex sm:items-center sm:justify-center sm:gap-8 md:gap-12 sm:pb-3 w-full max-w-full min-w-0">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              id={`tab-${tab.id}`}
              aria-controls={`tabpanel-${tab.id}`}
              aria-selected={isActive}
              onClick={() => onTabChange(tab.id)}
              className={cn(
                "relative min-h-[44px] px-2 sm:px-4 py-2 flex items-center justify-center font-serif tracking-[0.08em] transition-all duration-300 uppercase text-xs sm:text-sm font-bold active:scale-95 select-none focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-gold rounded-lg sm:rounded-none",
                isActive
                  ? "bg-brand-navy text-white shadow-sm dark:bg-brand-gold dark:text-[#0F141C] sm:bg-transparent sm:dark:bg-transparent sm:text-brand-navy sm:dark:text-brand-gold sm:font-black sm:shadow-none"
                  : "text-brand-text/60 dark:text-foreground/70 hover:text-brand-navy dark:hover:text-brand-gold"
              )}
            >
              <span className="truncate">{tab.label}</span>
              {isActive && (
                <motion.div
                  layoutId="activeTabUnderline"
                  className="hidden sm:block absolute bottom-0 left-0 right-0 h-0.5 bg-brand-gold"
                  transition={{ type: "spring", damping: 15, stiffness: 200 }}
                />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
