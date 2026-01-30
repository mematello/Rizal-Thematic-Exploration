# API Integration Guide

## Overview

This document explains how to connect the Next.js frontend to the FastAPI backend, including data transformation, error handling, and caching strategies.

---

## Table of Contents

1. [API Client Setup](#api-client-setup)
2. [Data Transformation](#data-transformation)
3. [React Query Integration](#react-query-integration)
4. [Error Handling](#error-handling)
5. [Highlighting Logic](#highlighting-logic)
6. [Type Safety](#type-safety)

---

## API Client Setup

### Base Configuration

```typescript
// lib/api/client.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
  ) {
    super(message);
    this.name = 'APIError';
  }
}

export async function fetchAPI<T>(
  endpoint: string,
  options?: RequestInit,
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  try {
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new APIError(
        data.message || 'API request failed',
        response.status,
        data.code,
      );
    }

    return data as T;
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    // Network error or JSON parse error
    throw new APIError(
      error instanceof Error ? error.message : 'Unknown error',
      0,
    );
  }
}
```

---

## Data Transformation

### Backend Response Types

```typescript
// types/api.ts

// What the backend returns
export interface BackendSearchResponse {
  status: 'success' | 'blocked';
  results?: {
    noli: BackendNovelResults;
    elfili: BackendNovelResults;
  };
  next_queries?: string[];
  query_time_ms?: number;
  // Error fields
  reason?: 'domain_incoherent' | 'too_short' | 'stops_only';
  message?: string;
  suggestions?: string[];
}

export interface BackendNovelResults {
  items: BackendResultItem[];
  metadata: {
    count: number;
    avg_score: number;
  };
}

export interface BackendResultItem {
  id: string;
  chapter_number: number;
  chapter_title: string;
  sentence_text: string;
  scores: {
    semantic: number;  // 0-100
    lexical: number;   // 0-100
    final: number;     // 0-100
  };
  match_type: 'exact' | 'partial_lexical' | 'semantic';
  context?: {
    prev: string[];
    next: string[];
  };
  themes?: Array<{
    id: string;
    label: string;
    confidence: number;
  }>;
}
```

### Frontend Component Types

```typescript
// types/search.ts (what components expect)
export interface ResultCardProps {
  id: string;
  novel: 'noli' | 'fili';
  chapter: number;
  chapterTitle: string;
  passageHtml: string;      // TRANSFORMED: highlighted HTML
  contextHtml: string;       // TRANSFORMED: formatted context
  semanticScore: number;
  lexicalScore: number;
  confidenceBadge: boolean;  // DERIVED: final > 85
  themes?: ThemeTag[];
}
```

### Transformation Function

```typescript
// lib/transformers/searchResults.ts

import type { BackendResultItem } from '@/types/api';
import type { ResultCardProps } from '@/types/search';

/**
 * Applies highlighting to text based on query and scores
 */
function applyHighlights(
  text: string,
  query: string,
  semanticScore: number,
  lexicalScore: number,
): string {
  let highlightedText = text;

  // A. Lexical Highlighting (Yellow Background)
  // Split query into words, ignore short stopwords
  const queryWords = query
    .toLowerCase()
    .split(/\s+/)
    .filter((word) => word.length > 2); // Ignore "sa", "ng", "at", etc.

  // Highlight each word (case-insensitive)
  queryWords.forEach((word) => {
    const regex = new RegExp(`(${escapeRegex(word)})`, 'gi');
    highlightedText = highlightedText.replace(
      regex,
      '<span class="lexical-match">$1</span>',
    );
  });

  // B. Semantic Highlighting (Teal Underline)
  // If semantic score is high (>75), underline the entire phrase
  // This indicates the match is conceptual, not just keyword-based
  if (semanticScore > 75 && lexicalScore < 50) {
    // Only underline if it's primarily semantic (not already highlighted lexically)
    highlightedText = `<span class="semantic-match">${highlightedText}</span>`;
  }

  return highlightedText;
}

/**
 * Formats context (prev/next sentences) as HTML
 */
function formatContext(context?: { prev: string[]; next: string[] }): string {
  if (!context || (context.prev.length === 0 && context.next.length === 0)) {
    return '';
  }

  const parts: string[] = [];

  if (context.prev.length > 0) {
    parts.push(
      `<div class="mb-2"><strong>Before:</strong> ${context.prev.join(' ')}</div>`,
    );
  }

  if (context.next.length > 0) {
    parts.push(
      `<div><strong>After:</strong> ${context.next.join(' ')}</div>`,
    );
  }

  return parts.join('');
}

/**
 * Transforms backend data to frontend component props
 */
export function transformBackendToResultCard(
  item: BackendResultItem,
  query: string,
  novel: 'noli' | 'fili',
): ResultCardProps {
  return {
    id: item.id || `${novel}-${item.chapter_number}-${Date.now()}`,
    novel,
    chapter: item.chapter_number,
    chapterTitle: item.chapter_title || `Chapter ${item.chapter_number}`,
    
    // The magic: Apply highlights
    passageHtml: applyHighlights(
      item.sentence_text,
      query,
      item.scores.semantic,
      item.scores.lexical,
    ),
    
    // Format context
    contextHtml: formatContext(item.context),
    
    // Round scores to integers
    semanticScore: Math.round(item.scores.semantic),
    lexicalScore: Math.round(item.scores.lexical),
    
    // High confidence badge if final score > 85
    confidenceBadge: item.scores.final > 85,
    
    // Pass through themes
    themes: item.themes?.map((t) => ({
      id: t.id,
      label: t.label,
      confidence: t.confidence,
    })),
  };
}

/**
 * Helper: Escape special regex characters
 */
function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Transform complete search response
 */
export function transformSearchResponse(
  response: BackendSearchResponse,
  query: string,
): {
  noli: ResultCardProps[];
  fili: ResultCardProps[];
  nextQueries: string[];
  queryTime: number;
} {
  if (response.status !== 'success' || !response.results) {
    return {
      noli: [],
      fili: [],
      nextQueries: [],
      queryTime: 0,
    };
  }

  return {
    noli: response.results.noli.items.map((item) =>
      transformBackendToResultCard(item, query, 'noli'),
    ),
    fili: response.results.elfili.items.map((item) =>
      transformBackendToResultCard(item, query, 'fili'),
    ),
    nextQueries: response.next_queries || [],
    queryTime: response.query_time_ms || 0,
  };
}
```

---

## React Query Integration

### API Functions

```typescript
// lib/api/search.ts

import { fetchAPI } from './client';
import type { BackendSearchResponse } from '@/types/api';

export interface SearchParams {
  query: string;
  top_k?: number;
  filters?: {
    match_type?: 'all' | 'exact' | 'semantic';
    min_score?: number;
  };
  include_context?: boolean;
}

export async function searchRizal(
  params: SearchParams,
): Promise<BackendSearchResponse> {
  return fetchAPI<BackendSearchResponse>('/api/v1/search', {
    method: 'POST',
    body: JSON.stringify({
      query: params.query,
      top_k: params.top_k || 10,
      filters: params.filters || {},
      include_context: params.include_context ?? true,
    }),
  });
}

export async function getThemes() {
  return fetchAPI<ThemeData[]>('/api/v1/themes', {
    method: 'GET',
  });
}

export async function getSuggestions(query: string) {
  return fetchAPI<string[]>(`/api/v1/suggestions?q=${encodeURIComponent(query)}`, {
    method: 'GET',
  });
}
```

### Custom Hooks

```typescript
// hooks/useRizalSearch.ts

import { useQuery } from '@tanstack/react-query';
import { searchRizal } from '@/lib/api/search';
import { transformSearchResponse } from '@/lib/transformers/searchResults';
import type { SearchParams } from '@/lib/api/search';

export function useRizalSearch(params: SearchParams) {
  return useQuery({
    queryKey: ['search', params.query, params.filters],
    queryFn: async () => {
      const response = await searchRizal(params);
      
      // Handle blocked queries
      if (response.status === 'blocked') {
        throw new Error(response.message || 'Query blocked');
      }
      
      // Transform data for frontend
      return transformSearchResponse(response, params.query);
    },
    enabled: params.query.length >= 3, // Only search if query is long enough
    staleTime: 5 * 60 * 1000,          // Consider data fresh for 5 minutes
    retry: 2,                           // Retry failed requests twice
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  });
}
```

### Usage in Components

```typescript
// app/search/page.tsx

'use client';

import { useSearchParams } from 'next/navigation';
import { useRizalSearch } from '@/hooks/useRizalSearch';
import { ResultCard } from '@/components/ResultCard';
import { SkeletonLoader } from '@/components/SkeletonLoader';
import { ErrorState } from '@/components/ErrorState';

export default function SearchPage() {
  const searchParams = useSearchParams();
  const query = searchParams.get('q') || '';

  const { data, isLoading, error } = useRizalSearch({
    query,
    filters: {
      match_type: 'all',
      min_score: 0,
    },
  });

  if (isLoading) {
    return <SkeletonLoader />;
  }

  if (error) {
    return <ErrorState message={error.message} />;
  }

  if (!data) {
    return null;
  }

  return (
    <div className="grid lg:grid-cols-2 gap-6">
      {/* Noli Column */}
      <div>
        <h2 className="text-xl font-bold mb-4">Noli Me Tangere</h2>
        {data.noli.length === 0 ? (
          <EmptyNovelState novel="noli" query={query} />
        ) : (
          data.noli.map((result) => (
            <ResultCard key={result.id} {...result} />
          ))
        )}
      </div>

      {/* Fili Column */}
      <div>
        <h2 className="text-xl font-bold mb-4">El Filibusterismo</h2>
        {data.fili.map((result) => (
          <ResultCard key={result.id} {...result} />
        ))}
      </div>
    </div>
  );
}
```

---

## Error Handling

### Error Types

```typescript
// types/errors.ts

export type ErrorCode =
  | 'NETWORK_ERROR'
  | 'VALIDATION_ERROR'
  | 'DOMAIN_INCOHERENT'
  | 'QUERY_TOO_SHORT'
  | 'RATE_LIMITED'
  | 'SERVER_ERROR';

export interface AppError {
  code: ErrorCode;
  message: string;
  suggestions?: string[];
  retryable: boolean;
}
```

### Error Handling Hook

```typescript
// hooks/useErrorHandler.ts

import { APIError } from '@/lib/api/client';
import type { AppError } from '@/types/errors';

export function useErrorHandler() {
  const handleError = (error: unknown): AppError => {
    if (error instanceof APIError) {
      // Backend returned an error
      if (error.status === 429) {
        return {
          code: 'RATE_LIMITED',
          message: 'Too many requests. Please wait a moment and try again.',
          retryable: true,
        };
      }

      if (error.status === 422) {
        return {
          code: 'VALIDATION_ERROR',
          message: error.message,
          retryable: false,
        };
      }

      if (error.status >= 500) {
        return {
          code: 'SERVER_ERROR',
          message: 'Server error. Our team has been notified.',
          retryable: true,
        };
      }

      // Blocked query (domain incoherent, etc.)
      return {
        code: error.code as ErrorCode,
        message: error.message,
        retryable: false,
      };
    }

    // Network error
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return {
        code: 'NETWORK_ERROR',
        message: 'Network error. Please check your connection.',
        retryable: true,
      };
    }

    // Unknown error
    return {
      code: 'SERVER_ERROR',
      message: 'An unexpected error occurred.',
      retryable: true,
    };
  };

  return { handleError };
}
```

### Error Display Component

```typescript
// components/ErrorState.tsx

import { AlertCircle } from 'lucide-react';
import type { AppError } from '@/types/errors';

export function ErrorState({ error }: { error: AppError }) {
  return (
    <div className="max-w-md mx-auto mt-12 p-6 bg-red-50 border border-red-200 rounded-lg">
      <div className="flex items-start gap-3">
        <AlertCircle className="text-red-600 mt-0.5" size={20} />
        <div>
          <h3 className="font-bold text-red-900 mb-1">Search Error</h3>
          <p className="text-red-800 text-sm mb-3">{error.message}</p>

          {error.suggestions && error.suggestions.length > 0 && (
            <div>
              <p className="text-sm font-semibold text-red-900 mb-2">
                Try these instead:
              </p>
              <ul className="space-y-1">
                {error.suggestions.map((suggestion, i) => (
                  <li key={i}>
                    <button
                      onClick={() => {
                        // Navigate to search with suggestion
                        window.location.href = `/search?q=${encodeURIComponent(suggestion)}`;
                      }}
                      className="text-sm text-red-700 hover:underline"
                    >
                      → {suggestion}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {error.retryable && (
            <button
              onClick={() => window.location.reload()}
              className="mt-3 text-sm font-bold text-red-700 hover:text-red-900"
            >
              Try Again
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
```

---

## Highlighting Logic

### Advanced Highlighting (Optional Enhancement)

If the backend can send `matched_spans` (character indices), use this:

```typescript
// lib/transformers/highlighting.ts

interface MatchedSpan {
  start: number;
  end: number;
  type: 'lexical' | 'semantic';
}

/**
 * Apply highlights using character indices from backend
 */
export function applySpanHighlights(
  text: string,
  spans: MatchedSpan[],
): string {
  // Sort spans by start position
  const sortedSpans = [...spans].sort((a, b) => a.start - b.start);

  let result = '';
  let lastIndex = 0;

  sortedSpans.forEach((span) => {
    // Add text before this span
    result += text.substring(lastIndex, span.start);

    // Add highlighted span
    const spanText = text.substring(span.start, span.end);
    const className = span.type === 'lexical' ? 'lexical-match' : 'semantic-match';
    result += `<span class="${className}">${spanText}</span>`;

    lastIndex = span.end;
  });

  // Add remaining text
  result += text.substring(lastIndex);

  return result;
}
```

### Fallback: Regex-Based Highlighting (Current Implementation)

The current `applyHighlights` function in the transformation section works for MVP. It:
1. Highlights exact keyword matches (lexical)
2. Underlines entire sentence if semantic score is high

**Limitation**: Can't highlight specific semantic matches within a sentence.

**Solution for Production**: Update backend to send `matched_spans` with character indices.

---

## Type Safety

### Zod Schemas for Runtime Validation

```typescript
// lib/validators/search.ts

import { z } from 'zod';

export const searchParamsSchema = z.object({
  query: z.string().min(3, 'Query must be at least 3 characters'),
  top_k: z.number().min(1).max(20).optional(),
  filters: z.object({
    match_type: z.enum(['all', 'exact', 'semantic']).optional(),
    min_score: z.number().min(0).max(100).optional(),
  }).optional(),
  include_context: z.boolean().optional(),
});

export const backendResponseSchema = z.object({
  status: z.enum(['success', 'blocked']),
  results: z.object({
    noli: z.object({
      items: z.array(z.any()), // Define full schema
      metadata: z.object({
        count: z.number(),
        avg_score: z.number(),
      }),
    }),
    elfili: z.object({
      items: z.array(z.any()),
      metadata: z.object({
        count: z.number(),
        avg_score: z.number(),
      }),
    }),
  }).optional(),
  next_queries: z.array(z.string()).optional(),
  query_time_ms: z.number().optional(),
  message: z.string().optional(),
  suggestions: z.array(z.string()).optional(),
});

// Usage in API client
export async function searchRizal(params: unknown) {
  const validatedParams = searchParamsSchema.parse(params);
  const response = await fetchAPI('/api/v1/search', {
    method: 'POST',
    body: JSON.stringify(validatedParams),
  });
  return backendResponseSchema.parse(response);
}
```

---

## Caching Strategy

### React Query Configuration

```typescript
// app/providers.tsx

'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { useState } from 'react';

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 5 * 60 * 1000,        // 5 minutes
            cacheTime: 10 * 60 * 1000,       // 10 minutes
            retry: 2,
            refetchOnWindowFocus: false,     // Don't refetch when tab regains focus
            refetchOnReconnect: true,        // Do refetch when internet reconnects
          },
        },
      }),
  );

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {process.env.NODE_ENV === 'development' && <ReactQueryDevtools />}
    </QueryClientProvider>
  );
}
```

### Prefetching (Optional)

```typescript
// app/page.tsx (Home)

import { dehydrate, HydrationBoundary, QueryClient } from '@tanstack/react-query';
import { getThemes } from '@/lib/api/search';

export default async function HomePage() {
  // Prefetch themes on server
  const queryClient = new QueryClient();
  
  await queryClient.prefetchQuery({
    queryKey: ['themes'],
    queryFn: getThemes,
  });

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      {/* Client component will have themes pre-loaded */}
      <HomeContent />
    </HydrationBoundary>
  );
}
```

---

## Testing API Integration

### Mock API Responses

```typescript
// __tests__/mocks/searchResponse.ts

export const mockSearchResponse: BackendSearchResponse = {
  status: 'success',
  results: {
    noli: {
      items: [
        {
          id: 'noli-4-1',
          chapter_number: 4,
          chapter_title: 'Erehe at Filibustero',
          sentence_text: 'Ang edukasyon ay susi sa pag-unlad.',
          scores: {
            semantic: 85,
            lexical: 92,
            final: 88,
          },
          match_type: 'exact',
          context: {
            prev: ['Nauna rito...'],
            next: ['Kasunod nito...'],
          },
        },
      ],
      metadata: {
        count: 1,
        avg_score: 88,
      },
    },
    elfili: {
      items: [],
      metadata: {
        count: 0,
        avg_score: 0,
      },
    },
  },
  next_queries: ['Edukasyon at reporma', 'Pag-asa ng kabataan'],
  query_time_ms: 285,
};
```

### Integration Tests

```typescript
// __tests__/api/search.test.ts

import { rest } from 'msw';
import { setupServer } from 'msw/node';
import { searchRizal } from '@/lib/api/search';
import { mockSearchResponse } from '../mocks/searchResponse';

const server = setupServer(
  rest.post('http://localhost:8000/api/v1/search', (req, res, ctx) => {
    return res(ctx.json(mockSearchResponse));
  }),
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('searchRizal', () => {
  it('fetches and transforms search results', async () => {
    const result = await searchRizal({ query: 'edukasyon' });
    
    expect(result.status).toBe('success');
    expect(result.results?.noli.items).toHaveLength(1);
    expect(result.results?.noli.items[0].chapter_title).toBe('Erehe at Filibustero');
  });

  it('handles blocked queries', async () => {
    server.use(
      rest.post('http://localhost:8000/api/v1/search', (req, res, ctx) => {
        return res(
          ctx.json({
            status: 'blocked',
            reason: 'domain_incoherent',
            message: 'Query blocked',
            suggestions: ['Try this instead'],
          }),
        );
      }),
    );

    const result = await searchRizal({ query: 'nonsense' });
    expect(result.status).toBe('blocked');
    expect(result.suggestions).toContain('Try this instead');
  });
});
```

---

## Environment Setup

### Environment Variables

```bash
# .env.local (frontend)
NEXT_PUBLIC_API_URL=http://localhost:8000

# .env.production
NEXT_PUBLIC_API_URL=https://api.rizal-explorer.com
```

### API Base URL Logic

```typescript
// lib/api/client.ts

function getAPIBaseURL(): string {
  // Server-side: use internal URL if available (faster)
  if (typeof window === 'undefined') {
    return process.env.API_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  }
  
  // Client-side: use public URL
  return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
}

export const API_BASE_URL = getAPIBaseURL();
```

---

## CORS Configuration (Backend)

Ensure FastAPI backend allows requests from your frontend:

```python
# backend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",           # Local development
        "https://rizal-explorer.com",      # Production
        "https://*.vercel.app",            # Vercel preview deployments
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
```

---

## Summary Checklist

- [ ] API client with proper error handling
- [ ] Data transformation utilities
- [ ] React Query hooks for search, themes, suggestions
- [ ] Error handling hook and component
- [ ] Highlighting logic (lexical + semantic)
- [ ] Zod schemas for validation
- [ ] Mock API responses for testing
- [ ] Environment variables configured
- [ ] CORS enabled on backend

---

## Next Steps

1. Read **06_IMPLEMENTATION_PLAN.md** for development roadmap
2. Set up API client and test with backend
3. Implement transformation functions
4. Create React Query hooks
5. Test integration with real backend data