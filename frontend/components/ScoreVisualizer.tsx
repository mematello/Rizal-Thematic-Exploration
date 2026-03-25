import { ScoreVisualizerProps } from "../types";

export function ScoreVisualizer({ semantic, lexical, char = 0, compact = false }: ScoreVisualizerProps & { char?: number, compact?: boolean }) {
    return (
        <div
            className={compact ? "flex items-center gap-4" : "mt-4 pt-3 border-t border-brand-brown/10 space-y-3"}
            aria-label={`Semantic match ${semantic}%, Lexical match ${lexical}%, Entity match ${char}%`}
        >
            {/* Semantic Score */}
            <div className="flex items-center gap-3">
                <span className="w-16 text-[10px] uppercase tracking-wider text-semantic-teal font-bold font-roboto">
                    Kahulugan
                </span>
                <div className="flex-1 h-1 bg-black/5 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-semantic-teal rounded-full transition-all duration-700"
                        style={{ width: `${semantic}%` }}
                        role="progressbar"
                        aria-valuenow={semantic}
                    />
                </div>
                <span className="w-8 text-right text-[10px] font-bold font-roboto text-semantic-teal">
                    {semantic}%
                </span>
            </div>

            {/* Lexical Score */}
            <div className="flex items-center gap-3">
                <span className="w-16 text-[10px] uppercase tracking-wider text-lexical-text font-bold font-roboto">
                    Salita
                </span>
                <div className="flex-1 h-1 bg-black/5 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-lexical-amber rounded-full transition-all duration-700"
                        style={{ width: `${lexical}%` }}
                        role="progressbar"
                        aria-valuenow={lexical}
                    />
                </div>
                <span className="w-8 text-right text-[10px] font-bold font-roboto text-lexical-text">
                    {lexical}%
                </span>
            </div>

            {/* Character Score */}
            <div className="flex items-center gap-3">
                <span className={`w-16 text-[10px] uppercase tracking-wider font-bold font-roboto ${char === -1 ? 'text-brand-text/40' : 'text-brand-blue'}`}>
                    Tauhan
                </span>
                <div className="flex-1 h-1 bg-black/5 rounded-full overflow-hidden">
                    <div
                        className={`h-full rounded-full transition-all duration-700 ${char === -1 ? 'bg-transparent' : 'bg-brand-blue'}`}
                        style={{ width: `${char === -1 ? 0 : char}%` }}
                        role="progressbar"
                        aria-valuenow={char === -1 ? 0 : char}
                    />
                </div>
                <span className={`w-8 text-right text-[10px] font-bold font-roboto ${char === -1 ? 'text-brand-text/40' : 'text-brand-blue'}`}>
                    {char === -1 ? 'N/A' : `${char}%`}
                </span>
            </div>
        </div>
    );
}
