import { ScoreVisualizerProps } from "../types";

export function ScoreVisualizer({ semantic, lexical }: ScoreVisualizerProps) {
    return (
        <div
            className="mt-4 pt-3 border-t border-brand-brown/10 space-y-2"
            aria-label={`Semantic match ${semantic}%, Lexical match ${lexical}%`}
        >
            {/* Semantic Score */}
            <div className="flex items-center gap-3">
                <span className="w-16 text-[10px] uppercase tracking-wider text-semantic-teal font-bold font-roboto">
                    Meaning
                </span>
                <div className="flex-1 h-1.5 bg-black/5 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-semantic-teal rounded-full transition-all duration-500"
                        style={{ width: `${semantic}%` }}
                        role="progressbar"
                        aria-valuenow={semantic}
                        aria-valuemin={0}
                        aria-valuemax={100}
                    />
                </div>
                <span className="w-8 text-right text-xs font-bold font-roboto text-semantic-teal">
                    {semantic}%
                </span>
            </div>

            {/* Lexical Score */}
            <div className="flex items-center gap-3">
                <span className="w-16 text-[10px] uppercase tracking-wider text-lexical-text font-bold font-roboto">
                    Word
                </span>
                <div className="flex-1 h-1.5 bg-black/5 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-lexical-amber rounded-full transition-all duration-500"
                        style={{ width: `${lexical}%` }}
                        role="progressbar"
                        aria-valuenow={lexical}
                        aria-valuemin={0}
                        aria-valuemax={100}
                    />
                </div>
                <span className="w-8 text-right text-xs font-bold font-roboto text-lexical-text">
                    {lexical}%
                </span>
            </div>
        </div>
    );
}
