import { useQueries } from '@tanstack/react-query';
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

async function fetchSearchResults(query: string, mode: 'buod' | 'full', novel: string): Promise<SearchResponse> {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
    const response = await fetch(`${apiUrl}/api/v1/search?q=${encodeURIComponent(query)}&source_type=${mode}&novel=${novel}`);

    if (!response.ok) {
        throw new Error('Search failed');
    }

    const data = await response.json();

    // Transform backend data to match ResultCardProps
    const transformResults = (results: any[], targetNovel: 'noli' | 'fili') => {
        if (!results) return [];
        return results.map(item => ({
            id: String(item.id),
            novel: targetNovel,
            chapter: item.chapter_number,
            chapterTitle: item.chapter_title,
            passageHtml: item.sentence_text,
            contextHtml: item.context_text,
            scores: item.scores,
            conceptMatchType: item.concept_match_type,
            sentenceIndex: item.sentence_index,
            themes: item.themes,
            confidenceBadge: item.scores?.final > 85
        }));
    };

    return {
        results: {
            noli: transformResults(data.results?.noli || [], 'noli'),
            fili: transformResults(data.results?.elfili || [], 'fili')
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

export function useRizalSearch(query: string, requiredNovels: ('noli' | 'fili')[] = ['noli', 'fili']) {
    const { mode } = useModeStore();

    const queryResults = useQueries({
        queries: requiredNovels.map(novel => ({
            queryKey: ['search', query, mode, novel],
            queryFn: () => fetchSearchResults(query, mode, novel === 'fili' ? 'elfili' : 'noli'),
            enabled: query.length >= 3,
            staleTime: 1000 * 60 * 5, // 5 minutes
        }))
    });

    const isLoading = queryResults.some(r => r.isLoading);
    const error = queryResults.find(r => r.error)?.error;

    // Merge data from both potentially independent queries
    const mergedData: SearchResponse = {
        results: { noli: [], fili: [] },
        metadata: {
            totalTime: 0,
            queryAnalysis: null,
            result_mode: 'lexical',
            reason: '',
            has_lexical_hits: true,
            suggestions: []
        }
    };

    queryResults.forEach((qRes) => {
        if (qRes.data) {
            if (qRes.data.results.noli.length > 0) mergedData.results.noli = qRes.data.results.noli;
            if (qRes.data.results.fili.length > 0) mergedData.results.fili = qRes.data.results.fili;
            if (mergedData.metadata.totalTime === 0 && qRes.data.metadata.totalTime) {
                mergedData.metadata = qRes.data.metadata;
            }
        }
    });

    return { data: mergedData, isLoading, error };
}
