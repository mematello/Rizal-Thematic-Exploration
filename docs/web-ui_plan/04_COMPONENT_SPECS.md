# Component Specifications

## Component Library Overview

This document provides detailed specifications for every component in the Rizal Thematic Exploration System. Each section includes TypeScript interfaces, implementation details, accessibility requirements, and usage examples.

---

## Table of Contents

1. [ResultCard](#resultcard) - The core search result component
2. [SearchBar](#searchbar) - Hero and persistent search input
3. [ScoreVisualizer](#scorevisualizer) - Dual score progress bars
4. [SkeletonLoader](#skeletonloader) - Loading state
5. [FilterBar](#filterbar) - Search filters
6. [ThemeCard](#themecard) - Explore page theme tiles
7. [EmptyNovelState](#emptynovelstate) - No results state
8. [TabSwitcher](#tabswitcher) - Mobile novel tabs
9. [LoadMoreButton](#loadmorebutton) - Pagination control
10. [MethodologyTabs](#methodologytabs) - Student/Researcher tabs

---

## ResultCard

### Purpose
Display a single search result with highlighted text, scores, and expandable context.

### Visual Specifications

**Desktop**:
- Width: 100% of column
- Padding: 16px
- Border-left: 4px solid (gold for Noli, magenta for Fili)
- Margin-bottom: 16px

**Mobile**:
- Same as desktop (full-width)

### TypeScript Interface

```typescript
// types/search.ts
export interface ResultCardProps {
  id: string;
  novel: 'noli' | 'fili';
  chapter: number;
  chapterTitle: string;
  passageHtml: string;        // Pre-highlighted HTML
  contextHtml: string;         // Surrounding sentences
  semanticScore: number;       // 0-100
  lexicalScore: number;        // 0-100
  confidenceBadge?: boolean;   // Show "High Confidence" badge
  themes?: ThemeTag[];         // Associated themes
}

interface ThemeTag {
  id: string;
  label: string;
  confidence: number;  // 0-1
}
```

### Implementation

```typescript
// components/ResultCard.tsx
import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { ScoreVisualizer } from './ScoreVisualizer';
import type { ResultCardProps } from '@/types/search';

export function ResultCard({
  id,
  novel,
  chapter,
  chapterTitle,
  passageHtml,
  contextHtml,
  semanticScore,
  lexicalScore,
  confidenceBadge = false,
  themes = [],
}: ResultCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Theme configuration
  const theme = {
    noli: {
      border: 'border-l-noli-gold',
      badge: 'bg-noli-gold/10 text-amber-800',
      novelName: 'Noli Me Tangere',
    },
    fili: {
      border: 'border-l-fili-magenta',
      badge: 'bg-fili-magenta/10 text-pink-800',
      novelName: 'El Filibusterismo',
    },
  }[novel];

  return (
    <article
      id={id}
      className={`
        bg-white rounded-r-lg shadow-sm border-l-4 ${theme.border}
        p-4 mb-4 transition-shadow hover:shadow-md
        scroll-mt-20
      `}
      aria-labelledby={`result-title-${id}`}
    >
      {/* Header */}
      <header className="flex justify-between items-start mb-3">
        <div className="flex-1">
          <span className="text-xs uppercase tracking-wide text-gray-500 font-roboto">
            {theme.novelName}
          </span>
          <h3
            id={`result-title-${id}`}
            className="font-roboto font-bold text-brand-brown text-lg leading-tight mt-1"
          >
            Chapter {chapter}: {chapterTitle}
          </h3>
        </div>

        {confidenceBadge && (
          <span
            className={`
              text-[10px] px-2 py-1 rounded-full font-bold
              uppercase tracking-wide ${theme.badge}
            `}
            aria-label="High confidence result"
          >
            High Confidence
          </span>
        )}
      </header>

      {/* Passage Body */}
      <div
        className="font-crimson text-lg leading-relaxed text-brand-text mb-3"
        dangerouslySetInnerHTML={{ __html: passageHtml }}
        aria-label="Search result passage"
      />

      {/* Theme Tags */}
      {themes.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3" aria-label="Related themes">
          {themes.map((theme) => (
            <span
              key={theme.id}
              className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded"
            >
              {theme.label}
            </span>
          ))}
        </div>
      )}

      {/* Context Expansion */}
      <div
        className={`
          grid transition-all duration-300 ease-out
          ${isExpanded ? 'grid-rows-[1fr]' : 'grid-rows-[0fr]'}
        `}
      >
        <div className="overflow-hidden">
          <div
            className="
              mt-3 pt-3 border-t border-dashed border-gray-200
              font-crimson text-gray-600 text-base italic
              bg-gray-50 p-3 rounded
            "
            dangerouslySetInnerHTML={{ __html: contextHtml }}
            aria-label="Surrounding context"
          />
        </div>
      </div>

      {/* Footer Actions */}
      <div className="mt-3 flex items-center justify-between">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="
            flex items-center gap-1 text-xs font-bold text-brand-blue
            hover:underline uppercase tracking-wide
            focus:outline-none focus:ring-2 focus:ring-brand-blue focus:ring-offset-2
            rounded px-2 py-1
          "
          aria-expanded={isExpanded}
          aria-controls={`context-${id}`}
        >
          {isExpanded ? (
            <>
              <ChevronUp size={14} />
              Hide Context
            </>
          ) : (
            <>
              <ChevronDown size={14} />
              Show Context
            </>
          )}
        </button>
      </div>

      {/* Score Visualizer */}
      <ScoreVisualizer semantic={semanticScore} lexical={lexicalScore} />
    </article>
  );
}
```

### Accessibility Checklist

- [x] Semantic HTML (`<article>`, `<header>`, `<h3>`)
- [x] Unique `id` and `aria-labelledby` for each card
- [x] `aria-expanded` on context toggle button
- [x] `aria-label` for screen readers on passage/context
- [x] Focus ring on interactive elements
- [x] Keyboard navigable (Tab to button, Enter to expand)

### Usage Example

```typescript
// app/search/page.tsx
import { ResultCard } from '@/components/ResultCard';

const results = [
  {
    id: 'noli-4-1',
    novel: 'noli' as const,
    chapter: 4,
    chapterTitle: 'Erehe at Filibustero',
    passageHtml: '<span class="lexical-match">Edukasyon</span> ang susi...',
    contextHtml: 'Nauna rito... Kasunod nito...',
    semanticScore: 85,
    lexicalScore: 92,
    confidenceBadge: true,
  },
];

return (
  <div>
    {results.map((result) => (
      <ResultCard key={result.id} {...result} />
    ))}
  </div>
);
```

---

## SearchBar

### Purpose
Primary input for search queries. Two variants: hero (large, on homepage) and persistent (smaller, on results page).

### Visual Specifications

**Hero Variant**:
- Height: 56px
- Max-width: 672px (2xl)
- Border: 2px solid Intramuros Brown
- Shadow: md

**Persistent Variant**:
- Height: 48px
- Max-width: 672px
- Border: 2px solid Intramuros Brown
- Shadow: sm

### TypeScript Interface

```typescript
export interface SearchBarProps {
  variant?: 'hero' | 'persistent';
  defaultValue?: string;
  placeholder?: string;
  isLoading?: boolean;
  onSearch: (query: string) => void;
  showSuggestions?: boolean;
}

interface Suggestion {
  text: string;
  type: 'semantic' | 'lexical' | 'recent';
}
```

### Implementation

```typescript
// components/SearchBar.tsx
import { useState, useEffect, useRef } from 'react';
import { Search, Loader2, Clock, Zap } from 'lucide-react';
import type { SearchBarProps, Suggestion } from '@/types/search';

export function SearchBar({
  variant = 'hero',
  defaultValue = '',
  placeholder = 'Maghanap ng tema, tauhan, o salita...',
  isLoading = false,
  onSearch,
  showSuggestions = true,
}: SearchBarProps) {
  const [query, setQuery] = useState(defaultValue);
  const [isOpen, setIsOpen] = useState(false);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const wrapperRef = useRef<HTMLDivElement>(null);

  // Mock suggestions (replace with API call)
  const MOCK_SUGGESTIONS: Suggestion[] = [
    { text: 'Edukasyon bilang susi', type: 'semantic' },
    { text: 'Katarungan para kay Sisa', type: 'semantic' },
    { text: "Simoun's jewelry", type: 'lexical' },
    { text: 'Padre Damaso', type: 'recent' },
  ];

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Load suggestions when typing (debounced)
  useEffect(() => {
    if (query.length >= 3 && showSuggestions) {
      const timer = setTimeout(() => {
        setSuggestions(MOCK_SUGGESTIONS);
      }, 300);
      return () => clearTimeout(timer);
    } else {
      setSuggestions([]);
    }
  }, [query, showSuggestions]);

  const handleSubmit = (searchQuery: string) => {
    if (searchQuery.length >= 3) {
      onSearch(searchQuery);
      setIsOpen(false);
    }
  };

  const heightClass = variant === 'hero' ? 'h-14' : 'h-12';
  const textClass = variant === 'hero' ? 'text-lg' : 'text-base';

  return (
    <div ref={wrapperRef} className="relative w-full max-w-2xl mx-auto z-50">
      {/* Input Field */}
      <div
        className={`
          relative flex items-center bg-white rounded-full
          border-2 border-brand-brown ${heightClass} shadow-md
          transition-all duration-200
          focus-within:ring-4 focus-within:ring-brand-brown/10
          focus-within:border-brand-blue
        `}
      >
        <input
          type="search"
          inputMode="search"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault();
              handleSubmit(query);
            }
          }}
          placeholder={placeholder}
          className={`
            w-full h-full bg-transparent px-6 rounded-full
            focus:outline-none font-roboto ${textClass}
            text-brand-text placeholder:text-gray-400 placeholder:italic
          `}
          aria-label="Search Rizal's novels"
          aria-autocomplete="list"
          aria-controls="search-suggestions"
          aria-expanded={isOpen && suggestions.length > 0}
        />

        {/* Icon / Spinner */}
        <div className="absolute right-4">
          {isLoading ? (
            <Loader2 className="animate-spin text-brand-brown" size={20} />
          ) : (
            <button
              onClick={() => handleSubmit(query)}
              className="
                p-2 hover:bg-gray-100 rounded-full transition-colors
                focus:outline-none focus:ring-2 focus:ring-brand-blue
              "
              aria-label="Search"
            >
              <Search size={20} className="text-brand-brown" />
            </button>
          )}
        </div>
      </div>

      {/* Type-ahead Dropdown */}
      {isOpen && suggestions.length > 0 && (
        <div
          id="search-suggestions"
          role="listbox"
          className="
            absolute top-16 left-0 right-0 bg-white rounded-xl
            shadow-xl border border-brand-brown/10 overflow-hidden
            py-2 animate-in fade-in slide-in-from-top-2 duration-200
          "
        >
          <div className="px-4 py-2 text-xs font-bold text-gray-400 uppercase tracking-wider">
            Suggestions
          </div>
          {suggestions.map((item, idx) => (
            <button
              key={idx}
              role="option"
              onClick={() => {
                setQuery(item.text);
                handleSubmit(item.text);
              }}
              className="
                w-full text-left px-4 py-3 hover:bg-brand-cream
                flex items-center gap-3 transition-colors
                focus:outline-none focus:bg-brand-cream
              "
            >
              {item.type === 'semantic' ? (
                <Zap size={16} className="text-semantic-teal" aria-hidden="true" />
              ) : item.type === 'recent' ? (
                <Clock size={16} className="text-gray-400" aria-hidden="true" />
              ) : (
                <Search size={16} className="text-gray-400" aria-hidden="true" />
              )}
              <span className="font-crimson text-lg text-brand-text">{item.text}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
```

### Accessibility Checklist

- [x] `inputMode="search"` for mobile keyboards
- [x] `aria-label` on input
- [x] `aria-autocomplete="list"` and `aria-controls`
- [x] `role="listbox"` and `role="option"` for suggestions
- [x] Keyboard navigation (Arrow keys to select suggestions)
- [x] Focus ring on search button

### Usage Example

```typescript
// app/page.tsx (Home)
import { SearchBar } from '@/components/SearchBar';
import { useRouter } from 'next/navigation';

export default function HomePage() {
  const router = useRouter();

  return (
    <main className="min-h-screen flex items-center justify-center bg-brand-cream">
      <SearchBar
        variant="hero"
        onSearch={(query) => router.push(`/search?q=${encodeURIComponent(query)}`)}
      />
    </main>
  );
}
```

---

## ScoreVisualizer

### Purpose
Display semantic and lexical scores as dual progress bars.

### Visual Specifications

- Two horizontal bars (4px height each)
- Semantic: Teal color
- Lexical: Amber color
- Labels: 10px uppercase, bold
- Values: 12px monospace, right-aligned

### TypeScript Interface

```typescript
export interface ScoreVisualizerProps {
  semantic: number;  // 0-100
  lexical: number;   // 0-100
}
```

### Implementation

```typescript
// components/ScoreVisualizer.tsx
import type { ScoreVisualizerProps } from '@/types/search';

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
            aria-label={`Semantic score: ${semantic}%`}
          />
        </div>
        <span className="w-8 text-right text-xs font-mono text-semantic-teal">
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
            className="h-full bg-noli-gold rounded-full transition-all duration-500"
            style={{ width: `${lexical}%` }}
            role="progressbar"
            aria-valuenow={lexical}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`Lexical score: ${lexical}%`}
          />
        </div>
        <span className="w-8 text-right text-xs font-mono text-lexical-text">
          {lexical}%
        </span>
      </div>
    </div>
  );
}
```

### Accessibility Checklist

- [x] `role="progressbar"` with `aria-valuenow/min/max`
- [x] Overall `aria-label` describing both scores
- [x] Visual AND text representation (not color-only)

---

## SkeletonLoader

### Purpose
Display loading state while search results are being fetched.

### Visual Specifications

- 3 skeleton cards
- Organic paragraph widths (90%, 85%, 60%)
- Pulse animation (2s duration)
- Random loading tip displayed

### Implementation

```typescript
// components/SkeletonLoader.tsx
import { useState } from 'react';

const LOADING_TIPS = [
  'Did you know? El Filibusterismo was finished in Ghent, Belgium.',
  'Analyzing semantic connections in 19th-century Tagalog...',
  'Consulting the archives for themes of Justice...',
  'Searching for key metaphors used by Ibarra...',
  'Rizal originally wrote Noli Me Tangere in Spanish.',
];

export function SkeletonLoader() {
  // Use useState initializer for one-time random selection
  const [tip] = useState(() => 
    LOADING_TIPS[Math.floor(Math.random() * LOADING_TIPS.length)]
  );

  return (
    <div className="w-full max-w-2xl mx-auto mt-8 px-4" role="status" aria-busy="true">
      <p className="text-center text-sm font-crimson italic text-gray-600 mb-6 animate-pulse">
        {tip}
      </p>

      {/* 3 Skeleton Cards */}
      {[1, 2, 3].map((i) => (
        <div
          key={i}
          className="
            bg-white rounded-r-lg border-l-4 border-gray-200
            p-4 mb-4 shadow-sm
            animate-[pulse_2s_cubic-bezier(0.4,0,0.6,1)_infinite]
          "
        >
          {/* Header Skeleton */}
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-2" />
          <div className="h-6 bg-gray-300 rounded w-3/4 mb-4" />

          {/* Body Paragraph Skeleton (Organic Widths) */}
          <div className="space-y-2 mb-4">
            <div className="h-4 bg-gray-200 rounded w-full" />
            <div className="h-4 bg-gray-200 rounded w-[92%]" />
            <div className="h-4 bg-gray-200 rounded w-[96%]" />
            <div className="h-4 bg-gray-200 rounded w-[60%]" />
          </div>

          {/* Footer Skeleton */}
          <div className="mt-4 pt-3 border-t border-gray-100 flex gap-4">
            <div className="h-2 bg-gray-200 rounded w-full" />
          </div>
        </div>
      ))}

      <span className="sr-only">Loading search results...</span>
    </div>
  );
}
```

---

## FilterBar

### Purpose
Allow users to filter search results by match type and minimum score.

### Visual Specifications

- Sticky positioning (below search header)
- Horizontal scroll on mobile
- Chip-style buttons
- Active state: filled with brand color

### Implementation

```typescript
// components/FilterBar.tsx
import { SlidersHorizontal, Check } from 'lucide-react';

export interface FilterBarProps {
  activeFilter: 'all' | 'exact' | 'semantic';
  minScore: number;
  onFilterChange: (filter: 'all' | 'exact' | 'semantic') => void;
  onMinScoreChange: (score: number) => void;
}

export function FilterBar({
  activeFilter,
  minScore,
  onFilterChange,
  onMinScoreChange,
}: FilterBarProps) {
  const chipClass = (filter: 'all' | 'exact' | 'semantic') =>
    activeFilter === filter
      ? 'bg-brand-brown text-white shadow-sm'
      : 'bg-white border border-brand-brown/20 text-brand-brown hover:bg-brand-brown/5';

  return (
    <div
      className="
        flex items-center gap-3 overflow-x-auto no-scrollbar py-2
        sticky top-0 bg-brand-cream/95 backdrop-blur-sm
        border-b border-brand-brown/10 z-10
      "
      role="toolbar"
      aria-label="Search filters"
    >
      <div className="flex items-center gap-2 text-brand-text text-sm font-bold mr-2">
        <SlidersHorizontal size={16} aria-hidden="true" />
        <span>Filter:</span>
      </div>

      {/* Match Type Chips */}
      <button
        onClick={() => onFilterChange('all')}
        className={`
          whitespace-nowrap px-4 py-1.5 rounded-full text-xs font-bold
          ${chipClass('all')} flex items-center gap-2
          focus:outline-none focus:ring-2 focus:ring-brand-blue focus:ring-offset-2
        `}
        aria-pressed={activeFilter === 'all'}
      >
        {activeFilter === 'all' && <Check size={12} aria-hidden="true" />}
        All Matches
      </button>

      <button
        onClick={() => onFilterChange('exact')}
        className={`
          whitespace-nowrap px-4 py-1.5 rounded-full text-xs font-bold
          ${chipClass('exact')}
          focus:outline-none focus:ring-2 focus:ring-brand-blue focus:ring-offset-2
        `}
        aria-pressed={activeFilter === 'exact'}
      >
        {activeFilter === 'exact' && <Check size={12} aria-hidden="true" />}
        Exact Only
      </button>

      <button
        onClick={() => onFilterChange('semantic')}
        className={`
          whitespace-nowrap px-4 py-1.5 rounded-full text-xs font-bold
          ${chipClass('semantic')}
          focus:outline-none focus:ring-2 focus:ring-brand-blue focus:ring-offset-2
        `}
        aria-pressed={activeFilter === 'semantic'}
      >
        {activeFilter === 'semantic' && <Check size={12} aria-hidden="true" />}
        Semantic Only
      </button>

      {/* Divider */}
      <div className="w-px h-6 bg-brand-brown/20 mx-1" aria-hidden="true" />

      {/* Min Score Toggle */}
      <button
        onClick={() => onMinScoreChange(minScore === 0 ? 80 : 0)}
        className={`
          whitespace-nowrap px-4 py-1.5 rounded-full text-xs font-bold
          ${minScore > 0 ? chipClass('all') : chipClass('semantic')}
          focus:outline-none focus:ring-2 focus:ring-brand-blue focus:ring-offset-2
        `}
        aria-pressed={minScore > 0}
      >
        {minScore > 0 && <Check size={12} aria-hidden="true" />}
        Min. Score {minScore > 0 ? `${minScore}%` : 'Off'}
      </button>
    </div>
  );
}
```

---

## ThemeCard

### Purpose
Display theme tiles on /explore page for thematic discovery.

### Visual Specifications

- Aspect ratio: 1:1.2 (portrait)
- Icon: 48px line-art
- Hover: Lift animation (-4px translateY)
- Grid: 2 cols mobile, 4 cols desktop

### Implementation

```typescript
// components/ThemeCard.tsx
import { Scale, Heart, Book, ShieldAlert, type LucideIcon } from 'lucide-react';

export interface ThemeCardProps {
  id: string;
  titleFilipino: string;
  subtitleEnglish: string;
  iconName: 'justice' | 'love' | 'education' | 'violence';
  onClick: () => void;
}

const ICON_MAP: Record<string, LucideIcon> = {
  justice: Scale,
  love: Heart,
  education: Book,
  violence: ShieldAlert,
};

export function ThemeCard({
  id,
  titleFilipino,
  subtitleEnglish,
  iconName,
  onClick,
}: ThemeCardProps) {
  const Icon = ICON_MAP[iconName];

  return (
    <article
      onClick={onClick}
      className="
        group cursor-pointer bg-white rounded-xl shadow-sm
        border border-brand-brown/10 p-6 aspect-[5/6]
        flex flex-col items-center justify-center text-center
        transition-all duration-300 hover:shadow-lg hover:-translate-y-1
        focus-within:ring-2 focus-within:ring-brand-blue focus-within:ring-offset-2
      "
      tabIndex={0}
      role="button"
      aria-label={`Explore ${titleFilipino} theme`}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      }}
    >
      <div className="mb-4 text-brand-brown group-hover:text-brand-blue transition-colors">
        <Icon strokeWidth={1.5} size={48} aria-hidden="true" />
      </div>

      <h3 className="font-roboto font-bold text-lg text-brand-text mb-1">
        {titleFilipino}
      </h3>

      <p className="font-crimson italic text-gray-500 text-sm">
        {subtitleEnglish}
      </p>
    </article>
  );
}
```

### Usage Example

```typescript
// app/explore/page.tsx
import { ThemeCard } from '@/components/ThemeCard';
import { useRouter } from 'next/navigation';

const THEMES = [
  {
    id: 'justice',
    titleFilipino: 'Katarungan',
    subtitleEnglish: 'Justice',
    iconName: 'justice' as const,
  },
  // ... more themes
];

export default function ExplorePage() {
  const router = useRouter();

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-6 p-6">
      {THEMES.map((theme) => (
        <ThemeCard
          key={theme.id}
          {...theme}
          onClick={() => router.push(`/search?theme=${theme.id}`)}
        />
      ))}
    </div>
  );
}
```

---

## Global CSS (Required)

Add these styles to `app/globals.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  /* Highlighting Styles for ResultCard */
  .lexical-match {
    background-color: #FFF59D;
    color: #261612;
    padding: 2px 0;
    font-weight: 600;
  }

  .semantic-match {
    text-decoration: underline;
    text-decoration-color: #00695C;
    text-decoration-thickness: 2px;
    text-underline-offset: 3px;
    text-decoration-skip-ink: none;
  }

  /* Hide scrollbar but keep functionality */
  .no-scrollbar::-webkit-scrollbar {
    display: none;
  }
  .no-scrollbar {
    -ms-overflow-style: none;
    scrollbar-width: none;
  }
}
```

---

## Component Testing Strategy

### Unit Tests (Vitest + React Testing Library)

```typescript
// __tests__/ResultCard.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ResultCard } from '@/components/ResultCard';

describe('ResultCard', () => {
  const mockProps = {
    id: 'test-1',
    novel: 'noli' as const,
    chapter: 4,
    chapterTitle: 'Test Chapter',
    passageHtml: 'Test passage',
    contextHtml: 'Test context',
    semanticScore: 85,
    lexicalScore: 92,
  };

  it('renders chapter title', () => {
    render(<ResultCard {...mockProps} />);
    expect(screen.getByText(/Chapter 4: Test Chapter/)).toBeInTheDocument();
  });

  it('expands context when clicked', async () => {
    render(<ResultCard {...mockProps} />);
    const button = screen.getByRole('button', { name: /Show Context/i });
    
    await userEvent.click(button);
    
    expect(button).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('Test context')).toBeVisible();
  });

  it('displays correct scores', () => {
    render(<ResultCard {...mockProps} />);
    expect(screen.getByText('85%')).toBeInTheDocument(); // Semantic
    expect(screen.getByText('92%')).toBeInTheDocument(); // Lexical
  });
});
```

---

## Performance Optimization

### Code Splitting

```typescript
// app/search/page.tsx
import dynamic from 'next/dynamic';

// Lazy load FilterBar (only needed on search page)
const FilterBar = dynamic(() => import('@/components/FilterBar'), {
  loading: () => <div className="h-12 bg-gray-100 animate-pulse" />,
});
```

### Memoization

```typescript
// components/ResultCard.tsx
import { memo } from 'react';

export const ResultCard = memo(function ResultCard({ ... }: ResultCardProps) {
  // Component implementation
}, (prevProps, nextProps) => {
  // Custom comparison: only re-render if id or scores change
  return (
    prevProps.id === nextProps.id &&
    prevProps.semanticScore === nextProps.semanticScore &&
    prevProps.lexicalScore === nextProps.lexicalScore
  );
});
```

---

## Next Steps

1. Read **05_API_INTEGRATION.md** for backend data transformation
2. Read **06_IMPLEMENTATION_PLAN.md** for step-by-step development
3. Start implementing components in priority order:
   - ResultCard → SearchBar → ScoreVisualizer → SkeletonLoader