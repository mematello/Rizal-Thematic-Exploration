import { useQuery } from '@tanstack/react-query';
import { useModeStore } from '../store/modeStore';
import { ResultCardProps } from '../types';

interface SearchResponse {
    results: {
        noli: ResultCardProps[];
        fili: ResultCardProps[];
    };
    metadata: {
        totalTime: number;
        queryAnalysis: any;
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
            contextHtml: item.context_text,
            scores: item.scores,
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
            totalTime: 0,
            queryAnalysis: null
        }
    };
}

export function useRizalSearch(query: string) {
    const { mode } = useModeStore();
    
    return useQuery({
        queryKey: ['search', query, mode],
        queryFn: () => fetchSearchResults(query, mode),
        enabled: query.length >= 3,
        staleTime: 1000 * 60 * 5, // 5 minutes
    });
}
