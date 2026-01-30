"use client";

import { clsx } from "clsx";

interface FilterBarProps {
    matchFilter: 'all' | 'noli' | 'fili';
    onMatchFilterChange: (filter: 'all' | 'noli' | 'fili') => void;
    minScore: number;
    onMinScoreChange: (score: number) => void;
}

export function FilterBar({
    matchFilter,
    onMatchFilterChange,
    minScore,
    onMinScoreChange,
}: FilterBarProps) {
    return (
        <div className="w-full bg-white border-y border-brand-brown/10 sticky top-0 z-30 shadow-sm backdrop-blur-sm bg-white/90">
            <div className="max-w-4xl mx-auto px-4 py-2 flex items-center justify-between gap-4 overflow-x-auto no-scrollbar">

                {/* Match Type Filters */}
                <div className="flex items-center gap-1 bg-brand-cream/50 p-1 rounded-lg">
                    {(['all', 'noli', 'fili'] as const).map((filter) => (
                        <button
                            key={filter}
                            onClick={() => onMatchFilterChange(filter)}
                            className={clsx(
                                "px-3 py-1.5 text-xs font-bold uppercase tracking-wide rounded transition-all",
                                matchFilter === filter
                                    ? "bg-brand-brown text-white shadow-sm"
                                    : "text-brand-brown/60 hover:bg-brand-brown/5 hover:text-brand-brown"
                            )}
                        >
                            {{
                                all: 'Lahat',
                                noli: 'Noli',
                                fili: 'Fili'
                            }[filter]}
                        </button>
                    ))}
                </div>

                {/* Score Slider (Desktop only typically, but we'll show it) */}
                <div className="flex items-center gap-3 min-w-[140px]">
                    <span className="text-[10px] uppercase font-bold text-gray-400 whitespace-nowrap">
                        Confidence &gt; {minScore}%
                    </span>
                    <input
                        type="range"
                        min="0"
                        max="90"
                        step="10"
                        value={minScore}
                        onChange={(e) => onMinScoreChange(Number(e.target.value))}
                        className="w-24 h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-brand-brown"
                    />
                </div>
            </div>
        </div>
    );
}
