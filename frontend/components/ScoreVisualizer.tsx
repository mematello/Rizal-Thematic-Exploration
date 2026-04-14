import { ScoreVisualizerProps } from "../types";

export function ScoreVisualizer({ 
    semantic, 
    lexical, 
    char,
    ratio,
    compact = false 
}: ScoreVisualizerProps & { char?: number, ratio?: number, compact?: boolean }) {
    
    const showCharRatio = char !== undefined && ratio !== undefined;
    const isCharNA = char === -1;
    const charVal = isCharNA ? 0 : (char ?? 0);
    const ratioVal = ratio ?? 0;
    
    // Total score calculation
    let totalScore = 0;
    if (showCharRatio) {
        if (isCharNA) {
            totalScore = Math.round(
                (lexical * (40/90)) + 
                (semantic * (40/90)) + 
                (ratioVal * (10/90))
            );
        } else {
            totalScore = Math.round(
                (lexical * 0.40) + 
                (semantic * 0.40) + 
                (charVal * 0.10) + 
                (ratioVal * 0.10)
            );
        }
    } else {
        // Search results: only Kahulugan + Salita, equal weight
        totalScore = Math.round((semantic + lexical) / 2);
    }

    return (
        <div className={compact ? "space-y-2" : "mt-4 pt-3 border-t border-brand-brown/10 space-y-3"}>
            {!compact && (
                <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-brand-brown/60">Antas ng Pagkakatulad</span>
                    <span className="text-sm font-black text-brand-brown">{totalScore}%</span>
                </div>
            )}

            {/* Semantic Score */}
            <ScoreRow 
                label="Kahulugan" 
                value={semantic} 
                weight={showCharRatio ? (isCharNA ? 44 : 40) : 50} 
                colorClass="bg-semantic-teal" 
                textClass="text-semantic-teal" 
            />

            {/* Lexical Score */}
            <ScoreRow 
                label="Salita" 
                value={lexical} 
                weight={showCharRatio ? (isCharNA ? 44 : 40) : 50} 
                colorClass="bg-lexical-amber" 
                textClass="text-lexical-text" 
            />

            {/* Tauhan and Posisyon — only shown in Sanggunian/reference view */}
            {showCharRatio && (
                <>
                    <ScoreRow 
                        label="Tauhan" 
                        value={charVal} 
                        weight={10} 
                        colorClass="bg-brand-blue" 
                        textClass="text-brand-blue"
                        isNA={isCharNA}
                    />
                    <ScoreRow 
                        label="Posisyon" 
                        value={ratioVal} 
                        weight={isCharNA ? 12 : 10} 
                        colorClass="bg-purple-500" 
                        textClass="text-purple-600" 
                    />
                </>
            )}
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
            <div className="flex items-center gap-1 w-16 justify-end whitespace-nowrap">
                <span className={`text-[10px] font-bold font-roboto ${isNA ? 'text-brand-text/40' : textClass}`}>
                    {isNA ? 'N/A' : `${value}%`}
                </span>
                <span className="text-[8px] opacity-40 font-medium">({weight}%)</span>
            </div>
        </div>
    );
}
