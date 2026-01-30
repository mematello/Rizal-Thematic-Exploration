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
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    const response = await fetch(`${apiUrl}/api/v1/search?q=${encodeURIComponent(query)}`);

    if (!response.ok) {
        throw new Error('Search failed');
    }

    return response.json();
}

export function useRizalSearch(query: string) {
    return useQuery({
        queryKey: ['search', query],
        queryFn: () => fetchSearchResults(query),
        enabled: query.length >= 3,
        staleTime: 1000 * 60 * 5, // 5 minutes
    });
}
