export interface Suggestion {
    text: string;
    type: 'semantic' | 'lexical' | 'recent';
}

export interface SearchBarProps {
    variant?: 'hero' | 'persistent';
    defaultValue?: string;
    placeholder?: string;
    isLoading?: boolean;
    onSearch: (query: string) => void;
    showSuggestions?: boolean;
}

export interface ResultCardProps {
    id: string;
    novel: 'noli' | 'fili';
    chapter: number;
    chapterTitle: string;
    passageHtml: string;
    contextHtml: string;
    scores: {
        semantic: number;
        lexical: number;
        final: number;
    };
    confidenceBadge?: boolean;
    themes?: { id: string; label: string }[];
}

export interface ScoreVisualizerProps {
    semantic: number;
    lexical: number;
}
