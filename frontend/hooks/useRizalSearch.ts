import { useQuery } from '@tanstack/react-query';
import { useModeStore } from '../store/modeStore';
import { ResultCardProps } from '../types';

interface SearchResponse {
    results: {
        noli: ResultCardProps[];
        fili: ResultCardProps[];
    };
    metadata: {
        totalTime?: number;
        queryAnalysis?: any;
        result_mode?: string;
        reason?: string;
        has_lexical_hits?: boolean;
        suggestions?: string[];
    };
}

async function fetchSearchResults(query: string, mode: 'buod' | 'full'): Promise<SearchResponse> {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
    const response = await fetch(`${apiUrl}/api/v1/search?q=${encodeURIComponent(query)}&source_type=${mode}`);

    if (!response.ok) {
        throw new Error('Search failed');
    }

    const data = await response.json();

    // Transform backend data to match ResultCardProps
    const transformResults = (results: any[], novel: 'noli' | 'fili') => {
        if (!results) return [];
        return results.map(item => ({
            id: String(item.id),
            novel: novel,
            chapter: item.chapter_number,
            chapterTitle: item.chapter_title,
            passageHtml: item.sentence_text,
            context: (item.context || []).map((s: any) => ({
                id: s.id,
                text: s.text,
                is_center: s.is_center,
            })),
            scores: item.scores,
            conceptMatchType: item.concept_match_type,
            sentenceIndex: item.sentence_index,
            themes: item.themes,
            confidenceBadge: item.scores?.final > 85
        }));
    };

    return {
        results: {
            noli: transformResults(data.results.noli, 'noli'),
            fili: transformResults(data.results.elfili || [], 'fili')
        },
        metadata: {
            totalTime: data.metadata?.totalTime || 0,
            queryAnalysis: data.metadata?.queryAnalysis || null,
            result_mode: data.metadata?.result_mode || "lexical",
            reason: data.metadata?.reason || "",
            has_lexical_hits: data.metadata?.has_lexical_hits ?? true,
            suggestions: data.metadata?.suggestions || []
        }
    };
}

export function useRizalSearch(query: string) {
    const { mode } = useModeStore();

    return useQuery({
        queryKey: ['search', query, mode],
        queryFn: () => fetchSearchResults(query, mode),
        enabled: query.length >= 3,
        staleTime: 0, // Always fresh — sort is applied in display layer
    });
}
