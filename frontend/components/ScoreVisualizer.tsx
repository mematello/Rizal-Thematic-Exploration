import { ScoreVisualizerProps } from "../types";

export function ScoreVisualizer({ 
    semantic, 
    lexical, 
    char = 0, 
    ratio = 0,
    compact = false 
}: ScoreVisualizerProps & { char?: number, ratio?: number, compact?: boolean }) {
    
    // Calculate total based on Sanggunian weights: 
    // Jaccard (Lexical) 30%, Semantic 40%, Tauhan (Char) 20%, Ratio 10%
    const charVal = char === -1 ? 0 : char;
    const totalScore = Math.round(
        (lexical * 0.30) + 
        (semantic * 0.40) + 
        (charVal * 0.20) + 
        (ratio * 0.10)
    );

    return (
        <div className={compact ? "space-y-2" : "mt-4 pt-3 border-t border-brand-brown/10 space-y-3"}>
            {!compact && (
                <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-brand-brown/60">Antas ng Pagkakatulad</span>
                    <span className="text-sm font-black text-brand-brown">{totalScore}%</span>
                </div>
            )}

            {/* Semantic Score (40%) */}
            <ScoreRow 
                label="Kahulugan" 
                value={semantic} 
                weight={40} 
                colorClass="bg-semantic-teal" 
                textClass="text-semantic-teal" 
            />

            {/* Lexical Score (30%) */}
            <ScoreRow 
                label="Salita" 
                value={lexical} 
                weight={30} 
                colorClass="bg-lexical-amber" 
                textClass="text-lexical-text" 
            />

            {/* Character Score (20%) */}
            <ScoreRow 
                label="Tauhan" 
                value={char === -1 ? 0 : char} 
                weight={20} 
                colorClass="bg-brand-blue" 
                textClass="text-brand-blue"
                isNA={char === -1}
            />

            {/* Ratio Score (10%) */}
            <ScoreRow 
                label="Ratio" 
                value={ratio} 
                weight={10} 
                colorClass="bg-purple-500" 
                textClass="text-purple-600" 
            />
        </div>
    );
}

function ScoreRow({ label, value, weight, colorClass, textClass, isNA = false }: { 
    label: string, 
    value: number, 
    weight: number, 
    colorClass: string, 
    textClass: string,
    isNA?: boolean
}) {
    return (
        <div className="flex items-center gap-3">
            <span className={`w-16 text-[9px] uppercase tracking-wider font-bold font-roboto ${isNA ? 'text-brand-text/40' : textClass}`}>
                {label}
            </span>
            <div className="flex-1 h-1 bg-black/5 rounded-full overflow-hidden">
                <div
                    className={`h-full rounded-full transition-all duration-700 ${isNA ? 'bg-transparent' : colorClass}`}
                    style={{ width: `${isNA ? 0 : value}%` }}
                    role="progressbar"
                    aria-valuenow={isNA ? 0 : value}
                />
            </div>
            <div className="flex items-center gap-1 w-12 justify-end">
                <span className={`text-[10px] font-bold font-roboto ${isNA ? 'text-brand-text/40' : textClass}`}>
                    {isNA ? 'N/A' : `${value}%`}
                </span>
                <span className="text-[8px] opacity-40 font-medium">({weight}%)</span>
            </div>
        </div>
    );
}
