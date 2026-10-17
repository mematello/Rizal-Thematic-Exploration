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
    isShort?: boolean;
    sentenceIndex?: number;
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
    conceptMatchType?: 'strong' | 'partial';
    confidenceBadge?: boolean;
    themes?: {
        id: string;
        label: string;
        score: number;
        explanation?: string;
    }[];
}

export interface ScoreVisualizerProps {
    semantic: number;
    lexical: number;
    char?: number;
    ratio?: number;
}

export interface ThemeMatch {
    id?: string;
    label?: string;
    score?: number;
    explanation?: string;
}

export interface ChapterContent {
    id: number;
    is_short?: boolean;
    sentence_index: number;
    sentence_text: string;
    themes?: ThemeMatch[];
}

export interface SanggunianReference {
    has_reference: boolean;
    passage_ids?: number[];
    reference_text?: string;
    alignment_status?: string;
    score?: number;
    semantic_score?: number;
    lexical_score?: number;
    char_score?: number;
    ratio_score?: number;
    matched_characters?: string[];
    buod_sentence_index?: number;
    full_sentence_indices?: number[];
    full_is_short?: boolean[];
}

export interface ChapterSanggunianResponse {
    book: string;
    chapter_number: number;
    mode: string;
    total_sentences: number;
    references: Record<string, SanggunianReference>;
}

