"use client";

import { useState } from "react";
import { ChevronUp, ChevronDown } from "lucide-react";
import { ResultCardProps } from "../types";
import { ScoreVisualizer } from "./ScoreVisualizer";

export function ResultCard({
    id,
    novel,
    chapter,
    chapterTitle,
    passageHtml,
    contextHtml,
    scores,
    confidenceBadge = false,
    themes = [],
}: ResultCardProps) {
    const [isExpanded, setIsExpanded] = useState(false);

    // Theme configuration
    const theme = {
        noli: {
            border: 'border-l-noli-gold',
            innerBorder: 'border-noli-gold',
            badge: 'bg-noli-gold/10 text-amber-800',
            novelName: 'Noli Me Tangere',
        },
        fili: {
            border: 'border-l-fili-magenta',
            innerBorder: 'border-fili-magenta',
            badge: 'bg-fili-magenta/10 text-pink-800',
            novelName: 'El Filibusterismo',
        },
    }[novel];

    return (
        <article
            id={id}
            className={`
        bg-white rounded-r-lg shadow-sm border-l-4 ${theme.border}
        p-4 mb-4 transition-shadow hover:shadow-md
        scroll-mt-20
      `}
            aria-labelledby={`result-title-${id}`}
        >
            {/* Header */}
            <header className="flex justify-between items-start mb-3">
                <div className="flex-1">
                    <span className="text-xs uppercase tracking-wide text-gray-500 font-roboto">
                        {theme.novelName}
                    </span>
                    <h3
                        id={`result-title-${id}`}
                        className="font-roboto font-bold text-brand-brown text-lg leading-tight mt-1"
                    >
                        Chapter {chapter}: {chapterTitle}
                    </h3>
                </div>

                {confidenceBadge && (
                    <span
                        className={`
              text-[10px] px-2 py-1 rounded-full font-bold
              uppercase tracking-wide ${theme.badge}
            `}
                        aria-label="High confidence result"
                    >
                        High Confidence
                    </span>
                )}
            </header>

            {/* Passage Body */}
            <div
                className="font-crimson text-lg leading-relaxed text-brand-text mb-3"
                dangerouslySetInnerHTML={{ __html: passageHtml }}
                aria-label="Search result passage"
            />

            {/* Theme Section (Editorial Style) */}
            {themes.length > 0 && (
                <div className={`mb-4 pl-4 border-l-2 ${theme.innerBorder}`}>
                    {themes.map((theme) => (
                        <div key={theme.id} className="flex flex-col gap-1">
                            {/* Label */}
                            <span className="text-[10px] font-bold uppercase tracking-widest text-stone-400">
                                Theme
                            </span>

                            {/* Title */}
                            <h4 className="text-base font-bold text-stone-800">
                                {theme.label}
                            </h4>

                            {/* Explanation (Quote Style) */}
                            {theme.explanation && (
                                <p className="font-crimson text-sm text-stone-600 italic leading-relaxed mt-1">
                                    "{theme.explanation}"
                                </p>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {/* Context Expansion */}
            <div
                className={`
          grid transition-all duration-300 ease-out
          ${isExpanded ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'}
        `}
            >
                <div className="overflow-hidden">
                    <div
                        className="
              mt-3 pt-3 border-t border-dashed border-gray-200
              font-crimson text-gray-600 text-base italic
              bg-gray-50 p-3 rounded
            "
                        dangerouslySetInnerHTML={{ __html: contextHtml }}
                        aria-label="Surrounding context"
                    />
                </div>
            </div>

            {/* Footer Actions */}
            <div className="mt-3 flex items-center justify-between">
                <button
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="
            flex items-center gap-1 text-xs font-bold text-brand-blue
            hover:underline uppercase tracking-wide
            focus:outline-none focus:ring-2 focus:ring-brand-blue focus:ring-offset-2
            rounded px-2 py-1
          "
                    aria-expanded={isExpanded}
                    aria-controls={`context-${id}`}
                >
                    {isExpanded ? (
                        <>
                            <ChevronUp size={14} />
                            Hide Context
                        </>
                    ) : (
                        <>
                            <ChevronDown size={14} />
                            Show Context
                        </>
                    )}
                </button>
            </div>

            {/* Score Visualizer */}
            <ScoreVisualizer semantic={scores.semantic} lexical={scores.lexical} />
        </article>
    );
}
