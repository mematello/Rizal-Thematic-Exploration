import { useQuery } from '@tanstack/react-query';
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

async function fetchSearchResults(query: string): Promise<SearchResponse> {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
    const response = await fetch(`${apiUrl}/api/v1/search?q=${encodeURIComponent(query)}`);

    if (!response.ok) {
        throw new Error('Search failed');
    }

    const data = await response.json();

    // Transform backend data to match ResultCardProps
    const transformResults = (results: any[], novel: 'noli' | 'fili') => {
        return results.map(item => ({
            id: String(item.id),
            novel: novel,
            chapter: item.chapter_number,
            chapterTitle: item.chapter_title,
            passageHtml: item.sentence_text,
            contextHtml: item.context_text,
            scores: item.scores,
            themes: item.themes,
            confidenceBadge: item.scores.final > 85
        }));
    };

    return {
        results: {
            noli: transformResults(data.results.noli, 'noli'),
            fili: transformResults(data.results.elfili || [], 'fili') // Backend sends 'elfili', frontend expects 'fili'
        },
        metadata: {
            totalTime: 0,
            queryAnalysis: null
        }
    };
}

export function useRizalSearch(query: string) {
    return useQuery({
        queryKey: ['search', query],
        queryFn: () => fetchSearchResults(query),
        enabled: query.length >= 3,
        staleTime: 1000 * 60 * 5, // 5 minutes
    });
}
