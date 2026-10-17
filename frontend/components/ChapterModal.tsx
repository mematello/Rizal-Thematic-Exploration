"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef, useState, useMemo } from "react";
import { X, Maximize2, Minimize2, ChevronLeft, ChevronRight, Users, BookOpen, Quote } from "lucide-react";
import { CHARACTERS, Character } from "@/lib/characterData";
import { ItemModal } from "@/components/ItemModal";
import { useModeStore } from "@/store/modeStore";
import { ScoreVisualizer } from "@/components/ScoreVisualizer";
import { ChapterContent, ThemeMatch, ChapterSanggunianResponse } from "@/types";
import { useQuery, useQueries, useQueryClient } from "@tanstack/react-query";
import { fetchChapters, fetchChapterContent, fetchChapterSanggunian } from "@/components/ChapterGrid";
import { formatChapterTitle, cleanChapterTitle } from "@/lib/utils";

export interface BuodMatch {
    buodSentenceId: number;
    sentence_index: number;
    text: string;
    score: number;
    alignment_status?: string;
    matched_characters?: string[];
    semantic_score?: number;
    lexical_score?: number;
    char_score?: number;
    ratio_score?: number;
    buod_chapter?: number;
}

export function getMatchingBuodChapters(book: string, fullChapter: number): number[] {
    const isNoli = book.toLowerCase().includes("noli");
    if (isNoli && fullChapter === 63) {
        return [63, 64];
    }
    return [fullChapter];
}


interface ChapterModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    chapterNumber: number;
    book: string;
    content: ChapterContent[];
    isLoading: boolean;
    highlightSentenceIndex?: number;
    /** Map View only: anchor sentence + context window to highlight with two-level styling */
    highlightContext?: { anchor: number; contextMin: number; contextMax: number } | null;
    onNavigate?: (book: string, chapter: number) => void;
}

interface LoadedChapter {
    chapterNumber: number;
    title: string;
    content: ChapterContent[];
    isLoading: boolean;
}

export function ChapterModal({
    isOpen,
    onClose,
    title,
    chapterNumber,
    book,
    content,
    isLoading,
    highlightSentenceIndex,
    highlightContext,
    onNavigate,
}: ChapterModalProps) {
    const queryClient = useQueryClient();
    const modalRef = useRef<HTMLDivElement>(null);
    const highlightRef = useRef<HTMLSpanElement>(null);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [showCharacters, setShowCharacters] = useState(false);
    const [showThemes, setShowThemes] = useState(false);
    const [characterKeywords, setCharacterKeywords] = useState<string[]>([]);
    const { mode } = useModeStore();
    const [modeError, setModeError] = useState<string | null>(null);
    const modeSwitchInProgress = useRef(false);
    const prevModeRef = useRef(mode);

    // Inline character modal state
    const [selectedChar, setSelectedChar] = useState<Character | null>(null);
    const [charAppearances, setCharAppearances] = useState<{
        book: string; chapter_number: number; chapter_title: string;
        score: number; preview_text?: string; sentence_index?: number;
    }[]>([]);
    const [charLoading, setCharLoading] = useState(false);
    const [isCharModalOpen, setIsCharModalOpen] = useState(false);
    const [charSortBy, setCharSortBy] = useState<'number' | 'relevance'>('number');

    const [showReference, setShowReference] = useState(false);
    const [refFullscreen, setRefFullscreen] = useState(false);
    const [selectedSentenceForRef, setSelectedSentenceForRef] = useState<number | null>(null);

    // Reference state
    const [selectedReference, setSelectedReference] = useState<{
        sentence_text: string;
        chapter_number: number;
        chapter_title: string;
        book: string;
        mode: string;
        alignment_status?: string;
        matched_characters?: string[];
        score: number;
        semantic_score: number;
        lexical_score: number;
        char_score: number;
        ratio_score: number;
        buod_sentence_index?: number;
        full_sentence_indices?: number[];
        full_is_short?: boolean[];
    } | null>(null);
    const [isRefLoading, setIsRefLoading] = useState(false);
    const [refError, setRefError] = useState<string | null>(null);

    // Reference results cache
    const [referenceResults, setReferenceResults] = useState<Record<number, any>>({});
    const [isLoadingRefs, setIsLoadingRefs] = useState(false);
    const [selectedSourceSentence, setSelectedSourceSentence] = useState<ChapterContent | null>(null);
    const [selectedFullMatches, setSelectedFullMatches] = useState<BuodMatch[] | null>(null);

    const handleReferenceClick = async (sentenceText: string, targetChapter: number) => {
        setIsRefLoading(true);
        setRefError(null);
        setSelectedReference(null);

        try {
            const targetMode = mode === 'buod' ? 'full' : 'buod';
            const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
            const apiUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;

            const res = await fetch(`${apiUrl}/api/v1/chapters/reference`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sentence_text: sentenceText,
                    book: book,
                    chapter_number: targetChapter,
                    target_mode: targetMode
                })
            });

            if (!res.ok) {
                if (res.status === 404) throw new Error("Walang nakitang kaugnay na pangungusap sa kabilang bersyon.");
                throw new Error("Failed to fetch reference");
            }

            const data = await res.json();
            setSelectedReference(data);
        } catch (err: any) {
            console.error("Reference fetch error:", err);
            setRefError(err.message || "Error fetching reference");
        } finally {
            setIsRefLoading(false);
        }
    };

    // Fullscreen multi-chapter state
    const [loadedChapters, setLoadedChapters] = useState<LoadedChapter[]>([]);
    const [chapterMetadata, setChapterMetadata] = useState<Record<number, string>>({});
    const [isLoadingMore, setIsLoadingMore] = useState(false);
    const bottomSentinelRef = useRef<HTMLDivElement>(null);
    const topSentinelRef = useRef<HTMLDivElement>(null);
    const scrollAreaRef = useRef<HTMLDivElement>(null);

    // State for tracking active chapter in view
    const [activeChapterInView, setActiveChapterInView] = useState<number>(chapterNumber);

    // Sync active chapter with prop when NOT in fullscreen (background sync)
    useEffect(() => {
        if (!isFullscreen) {
            setActiveChapterInView(chapterNumber);
        }
    }, [chapterNumber, isFullscreen]);


    // ── Sanggunian Cross-Reference Chapter Fetching & Prefetching ───────────
    const currentChapterForSanggunian = isFullscreen ? activeChapterInView : chapterNumber;

    // 1. React Query for active chapter sanggunian (Buod mode)
    const { data: chapterSanggunianData, isError: isSanggunianError } = useQuery<ChapterSanggunianResponse>({
        queryKey: ['sanggunian', book, currentChapterForSanggunian, mode],
        queryFn: () => fetchChapterSanggunian(book, currentChapterForSanggunian, mode),
        staleTime: 1000 * 60 * 60 * 24, // 24 hours
        enabled: isOpen && showReference && mode === 'buod',
    });

    // ── Full-mode Inverse Sanggunian: Map<fullSentenceId, BuodMatch[]> ───
    const matchingBuodChapters = useMemo(() => {
        if (mode !== 'full') return [];
        const chapters = isFullscreen ? loadedChapters.map(c => c.chapterNumber) : [currentChapterForSanggunian];
        const set = new Set<number>();
        chapters.forEach(ch => {
            getMatchingBuodChapters(book, ch).forEach(b => set.add(b));
        });
        return Array.from(set).sort((a, b) => a - b);
    }, [mode, isFullscreen, loadedChapters, currentChapterForSanggunian, book]);

    const sanggunianQueries = useQueries({
        queries: matchingBuodChapters.map(ch => ({
            queryKey: ['sanggunian', book, ch, 'buod'],
            queryFn: () => fetchChapterSanggunian(book, ch, 'buod'),
            staleTime: 1000 * 60 * 60 * 24,
            enabled: isOpen && showReference && mode === 'full',
        }))
    });

    const buodContentQueries = useQueries({
        queries: matchingBuodChapters.map(ch => ({
            queryKey: ['chapter', book, ch, 'buod'],
            queryFn: () => fetchChapterContent(book, ch, 'buod'),
            staleTime: 1000 * 60 * 60 * 24,
            enabled: isOpen && showReference && mode === 'full',
        }))
    });

    const isFullInverseLoading = mode === 'full' && showReference && (
        sanggunianQueries.some(q => q.isLoading) || buodContentQueries.some(q => q.isLoading)
    );
    const isFullInverseError = mode === 'full' && showReference && (
        sanggunianQueries.some(q => q.isError) || buodContentQueries.some(q => q.isError)
    );

    const fullInverseMap = useMemo(() => {
        if (mode !== 'full' || !showReference) return new Map<number, BuodMatch[]>();

        const map = new Map<number, BuodMatch[]>();
        matchingBuodChapters.forEach((bCh, idx) => {
            const sangData = sanggunianQueries[idx]?.data as ChapterSanggunianResponse | undefined;
            const buodContent = buodContentQueries[idx]?.data as ChapterContent[] | undefined;

            if (!sangData?.references || !buodContent) return;

            const buodSentenceMap = new Map<number, ChapterContent>();
            buodContent.forEach(s => buodSentenceMap.set(s.id, s));

            Object.entries(sangData.references).forEach(([sidStr, ref]) => {
                const sid = Number(sidStr);
                if (!ref.has_reference || !ref.passage_ids) return;
                const buodSentence = buodSentenceMap.get(sid);
                if (!buodSentence) return;

                const matchItem: BuodMatch = {
                    buodSentenceId: buodSentence.id,
                    sentence_index: buodSentence.sentence_index,
                    text: buodSentence.sentence_text,
                    score: ref.score ?? 0,
                    alignment_status: ref.alignment_status,
                    matched_characters: ref.matched_characters,
                    semantic_score: ref.semantic_score,
                    lexical_score: ref.lexical_score,
                    char_score: ref.char_score,
                    ratio_score: ref.ratio_score,
                    buod_chapter: bCh,
                };

                ref.passage_ids.forEach((pid: number) => {
                    const existing = map.get(pid);
                    if (existing) {
                        if (!existing.some(m => m.buodSentenceId === matchItem.buodSentenceId)) {
                            existing.push(matchItem);
                        }
                    } else {
                        map.set(pid, [matchItem]);
                    }
                });
            });
        });

        // Sort matches for each full sentence by sentence_index ascending
        map.forEach(matches => {
            matches.sort((a, b) => {
                if (a.buod_chapter !== b.buod_chapter) {
                    return (a.buod_chapter || 0) - (b.buod_chapter || 0);
                }
                return a.sentence_index - b.sentence_index;
            });
        });

        return map;
    }, [mode, showReference, matchingBuodChapters, sanggunianQueries, buodContentQueries]);

    // 2. Prefetch on modal open
    useEffect(() => {
        if (!isOpen) return;

        if (mode === 'full') {
            const buods = getMatchingBuodChapters(book, currentChapterForSanggunian);
            buods.forEach(bCh => {
                queryClient.prefetchQuery({
                    queryKey: ['sanggunian', book, bCh, 'buod'],
                    queryFn: () => fetchChapterSanggunian(book, bCh, 'buod'),
                    staleTime: 1000 * 60 * 60 * 24,
                });
                queryClient.prefetchQuery({
                    queryKey: ['chapter', book, bCh, 'buod'],
                    queryFn: () => fetchChapterContent(book, bCh, 'buod'),
                    staleTime: 1000 * 60 * 60 * 24,
                });
            });
            return;
        }

        // Prefetch active chapter in buod mode
        queryClient.prefetchQuery({
            queryKey: ['sanggunian', book, currentChapterForSanggunian, mode],
            queryFn: () => fetchChapterSanggunian(book, currentChapterForSanggunian, mode),
            staleTime: 1000 * 60 * 60 * 24,
        });

        // Prefetch adjacent chapters in buod mode
        if (mode === 'buod') {
            const isNoli = book.toLowerCase().includes('noli');
            const totalChapters = isNoli ? 64 : 39;
            const prevCh = currentChapterForSanggunian > 1 ? currentChapterForSanggunian - 1 : null;
            const nextCh = currentChapterForSanggunian < totalChapters ? currentChapterForSanggunian + 1 : null;

            if (prevCh) {
                queryClient.prefetchQuery({
                    queryKey: ['sanggunian', book, prevCh, mode],
                    queryFn: () => fetchChapterSanggunian(book, prevCh, mode),
                    staleTime: 1000 * 60 * 60 * 24,
                });
            }
            if (nextCh) {
                queryClient.prefetchQuery({
                    queryKey: ['sanggunian', book, nextCh, mode],
                    queryFn: () => fetchChapterSanggunian(book, nextCh, mode),
                    staleTime: 1000 * 60 * 60 * 24,
                });
            }
        }
    }, [isOpen, book, currentChapterForSanggunian, mode, queryClient]);

    // 3. Populate referenceResults from chapter query data (Buod mode)
    useEffect(() => {
        if (mode !== 'buod' || !chapterSanggunianData?.references) return;

        setReferenceResults(prev => {
            const next = { ...prev };
            Object.entries(chapterSanggunianData.references).forEach(([sidStr, ref]) => {
                const sid = Number(sidStr);
                next[sid] = ref;
                if (ref.has_reference && ref.passage_ids) {
                    ref.passage_ids.forEach((pid: number) => {
                        next[pid] = ref;
                    });
                }
            });
            return next;
        });
    }, [chapterSanggunianData, mode]);

    // 4. Fallback: If chapter endpoint fails in buod mode, fall back to per-sentence endpoint
    useEffect(() => {
        if (showReference && mode === 'buod' && isSanggunianError && !isLoadingRefs) {
            const fetchRefsFallback = async () => {
                setIsLoadingRefs(true);
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
                const sentences = isFullscreen ? loadedChapters.flatMap(c => c.content) : content;
                const results: Record<number, any> = { ...referenceResults };
                const toFetch = sentences.filter(s => !results[s.id]);
                
                const batchSize = 5;
                for (let i = 0; i < toFetch.length; i += batchSize) {
                    const batch = toFetch.slice(i, i + batchSize);
                    await Promise.all(batch.map(async (s) => {
                        try {
                            const res = await fetch(`${apiUrl}/api/v1/sentences/${s.id}/sanggunian`);
                            if (res.ok) {
                                const data = await res.json();
                                results[s.id] = data;
                                if (data.has_reference && data.passage_ids) {
                                    data.passage_ids.forEach((pid: number) => {
                                        results[pid] = data;
                                    });
                                }
                            } else {
                                results[s.id] = { has_reference: false };
                            }
                        } catch {
                            results[s.id] = { has_reference: false };
                        }
                    }));
                    setReferenceResults({ ...results });
                }
                setIsLoadingRefs(false);
            };
            fetchRefsFallback();
        }
    }, [showReference, mode, isSanggunianError, isFullscreen, loadedChapters, content]);

    // Precise Scroll Spy Logic for Fullscreen
    useEffect(() => {
        if (!isFullscreen || loadedChapters.length === 0) return;

        const observers: IntersectionObserver[] = [];

        loadedChapters.forEach((chap) => {
            const element = document.getElementById(`chapter-view-${chap.chapterNumber}`);
            if (element) {
                const observer = new IntersectionObserver(
                    (entries) => {
                        entries.forEach((entry) => {
                            if (entry.isIntersecting) {
                                setActiveChapterInView(chap.chapterNumber);
                            }
                        });
                    },
                    {
                        rootMargin: "-10% 0px -60% 0px",
                        threshold: 0
                    }
                );
                observer.observe(element);
                observers.push(observer);
            }
        });

        return () => {
            observers.forEach(obs => obs.disconnect());
        };
    }, [isFullscreen, loadedChapters]);

    // Fetch metadata once
    useEffect(() => {
        if (isOpen) {
            const fetchMetadata = async () => {
                try {
                    const data = await queryClient.fetchQuery({
                        queryKey: ["chapters", mode],
                        queryFn: () => fetchChapters(mode),
                        staleTime: 1000 * 60 * 60 * 24,
                    });
                    const map: Record<number, string> = {};
                    data.filter((c: any) => (book.toLowerCase() === 'noli' ? c.book.toLowerCase() === 'noli' || c.book === 'Noli Me Tangere' : c.book.toLowerCase() === 'elfili' || c.book === 'El Filibusterismo'))
                        .forEach((c: any) => {
                            map[c.chapter_number] = formatChapterTitle(c.chapter_title);
                        });
                    setChapterMetadata(map);
                } catch (err) {
                    console.error("Error fetching metadata:", err);
                }
            };
            fetchMetadata();
        }
    }, [isOpen, book, mode]);

    // Initialize loadedChapters when entering fullscreen or when initial content changes
    useEffect(() => {
        if (isFullscreen) {
            setLoadedChapters([{
                chapterNumber,
                title,
                content: content || [],
                isLoading: false
            }]);

            setTimeout(() => {
                const element = document.getElementById(`chapter-view-${chapterNumber}`);
                if (element) {
                    element.scrollIntoView({ behavior: 'auto', block: 'start' });
                }
            }, 100);
        }
    }, [isFullscreen, chapterNumber, title, content]);

    // Real-time mode fetching
    useEffect(() => {
        if (!isOpen || !book || !chapterNumber) return;
        if (prevModeRef.current === mode) return;

        let isMounted = true;
        const abortController = new AbortController();

        const fetchNewModeData = async () => {
            if (modeSwitchInProgress.current) return;
            modeSwitchInProgress.current = true;
            setModeError(null);

            try {
                const targetChapter = isFullscreen ? activeChapterInView : chapterNumber;
                const data = await queryClient.fetchQuery({
                    queryKey: ["chapter", book, targetChapter, mode],
                    queryFn: () => fetchChapterContent(book, targetChapter, mode),
                    staleTime: 1000 * 60 * 60 * 24,
                });

                if (isMounted) {
                    setLoadedChapters(prev => prev.map(chap =>
                        chap.chapterNumber === targetChapter ? { ...chap, content: data, isLoading: false } : chap
                    ));
                    // If not fullscreen, content is managed by page.tsx, so we don't manually set it here.
                    // But for robustness, we update prevModeRef
                }
            } catch (err: any) {
                if (err.name === 'AbortError') return;
                console.error("Mode switch fetch error:", err);
                if (isMounted) {
                    setModeError(err.message || 'Error occurred while switching modes.');
                }
            } finally {
                if (isMounted) {
                    modeSwitchInProgress.current = false;
                    prevModeRef.current = mode;
                }
            }
        };

        fetchNewModeData();

        return () => {
            isMounted = false;
            abortController.abort();
        };
    }, [mode, isOpen, book, chapterNumber, activeChapterInView, isFullscreen]);

    // Fetch characters for this chapter
    useEffect(() => {
        if (isOpen && book && chapterNumber) {
            const allKeywords = CHARACTERS.flatMap(c => [c.name, ...(c.aliases || [])]);
            const uniqueKeywords = Array.from(new Set(allKeywords)).sort((a, b) => b.length - a.length);
            setCharacterKeywords(uniqueKeywords);
        }
    }, [isOpen, book, chapterNumber]);

    // Close on escape key
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                if (refFullscreen) {
                    setRefFullscreen(false);
                    setSelectedSentenceForRef(null);
                    setSelectedReference(null);
                } else if (isFullscreen) {
                    setIsFullscreen(false);
                } else {
                    onClose();
                }
            }
        };
        if (isOpen) {
            window.addEventListener("keydown", handleKeyDown);
            document.body.style.overflow = "hidden";
        }
        return () => {
            window.removeEventListener("keydown", handleKeyDown);
            document.body.style.overflow = "unset";
        };
    }, [isOpen, isFullscreen, refFullscreen, onClose]);

    // Scroll to highlighted sentence
    useEffect(() => {
        if (isOpen && !isLoading && highlightSentenceIndex !== undefined && highlightRef.current) {
            setTimeout(() => {
                highlightRef.current?.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center',
                });
            }, 300);
        }
    }, [isOpen, isLoading, highlightSentenceIndex]);

    const handleBackdropClick = (e: React.MouseEvent) => {
        if (!isFullscreen && modalRef.current && !modalRef.current.contains(e.target as Node)) {
            onClose();
        }
    };

    const isNoli = book.toLowerCase() === "noli";
    const modalNovel = isNoli ? 'noli' : 'fili';
    const accentColor = isNoli ? "text-noli-accent" : "text-fili-accent";

    const NOLI_CHAPTERS = 64;
    const FILI_CHAPTERS = 39;

    const handlePrevChapter = () => {
        if (!onNavigate) return;
        const maxChapters = isNoli ? NOLI_CHAPTERS : FILI_CHAPTERS;
        if (chapterNumber <= 1) onNavigate(book, maxChapters);
        else onNavigate(book, chapterNumber - 1);
    };

    const handleNextChapter = () => {
        if (!onNavigate) return;
        const maxChapters = isNoli ? NOLI_CHAPTERS : FILI_CHAPTERS;
        if (chapterNumber >= maxChapters) onNavigate(book, 1);
        else onNavigate(book, chapterNumber + 1);
    };

    const fetchSpecificChapter = async (num: number, overrideMode?: 'buod' | 'full', signal?: AbortSignal): Promise<ChapterContent[]> => {
        const targetMode = overrideMode || mode;
        return queryClient.fetchQuery({
            queryKey: ["chapter", book, num, targetMode],
            queryFn: () => fetchChapterContent(book, num, targetMode),
            staleTime: 1000 * 60 * 60 * 24,
        });
    };

    const loadNextChapter = async () => {
        if (isLoadingMore || loadedChapters.length === 0) return;
        const lastChapter = loadedChapters[loadedChapters.length - 1];
        const nextNum = lastChapter.chapterNumber + 1;
        const maxChapters = isNoli ? NOLI_CHAPTERS : FILI_CHAPTERS;
        if (nextNum > maxChapters) return;
        setIsLoadingMore(true);
        try {
            const content = await fetchSpecificChapter(nextNum);
            setLoadedChapters(prev => [...prev, {
                chapterNumber: nextNum,
                title: chapterMetadata[nextNum] || `Kabanata ${nextNum}`,
                content,
                isLoading: false
            }]);
        } catch (err) {}
        setIsLoadingMore(false);
    };

    const loadPrevChapter = async () => {
        if (isLoadingMore || loadedChapters.length === 0) return;
        const firstChapter = loadedChapters[0];
        const prevNum = firstChapter.chapterNumber - 1;
        if (prevNum < 1) return;
        setIsLoadingMore(true);
        const container = scrollAreaRef.current;
        const prevScrollHeight = container?.scrollHeight || 0;
        const prevScrollTop = container?.scrollTop || 0;
        try {
            const content = await fetchSpecificChapter(prevNum);
            setLoadedChapters(prev => [{
                chapterNumber: prevNum,
                title: chapterMetadata[prevNum] || `Kabanata ${prevNum}`,
                content,
                isLoading: false
            }, ...prev]);
            requestAnimationFrame(() => {
                if (container) {
                    const newScrollHeight = container.scrollHeight;
                    container.scrollTop = prevScrollTop + (newScrollHeight - prevScrollHeight);
                }
            });
        } catch (err) {}
        setTimeout(() => setIsLoadingMore(false), 200);
    };

    useEffect(() => {
        if (!isFullscreen) return;
        const bottomObserver = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) loadNextChapter();
        }, { threshold: 0.1 });
        let topObserver: IntersectionObserver | null = null;
        const timer = setTimeout(() => {
            topObserver = new IntersectionObserver((entries) => {
                if (entries[0].isIntersecting) loadPrevChapter();
            }, { threshold: 0.1 });
            if (topSentinelRef.current) topObserver.observe(topSentinelRef.current);
        }, 1000);
        if (bottomSentinelRef.current) bottomObserver.observe(bottomSentinelRef.current);
        return () => {
            bottomObserver.disconnect();
            if (topObserver) topObserver.disconnect();
            clearTimeout(timer);
        };
    }, [isFullscreen, loadedChapters, isLoadingMore]);

    const fetchCharAppearances = async (char: Character, sort: 'number' | 'relevance') => {
        setCharLoading(true);
        try {
            const searchTerm = [char.name, ...(char.aliases || [])].join(",");
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
            const res = await fetch(`${apiUrl}/api/v1/characters/chapters?name=${encodeURIComponent(searchTerm)}&sort_by=${sort}&mode=${mode}`);
            if (res.ok) setCharAppearances(await res.json());
        } catch (err) {}
        setCharLoading(false);
    };

    const handleCharMarkClick = async (clickedText: string) => {
        const match = CHARACTERS.find(c => c.name.toLowerCase() === clickedText.toLowerCase() || (c.aliases || []).some(a => a.toLowerCase() === clickedText.toLowerCase()));
        if (!match) return;
        setSelectedChar(match);
        setIsCharModalOpen(true);
        await fetchCharAppearances(match, 'number');
    };

    const handleCharModalClose = () => {
        setIsCharModalOpen(false);
        setTimeout(() => { setSelectedChar(null); setCharAppearances([]); }, 300);
    };

    const highlightText = (text: string) => {
        if (!showCharacters || characterKeywords.length === 0) return text;
        const pattern = characterKeywords.map(k => {
            const escaped = k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            return `(?<![\\wÀ-ÖØ-öø-ÿ])${escaped}(?![\\wÀ-ÖØ-öø-ÿ])`;
        }).join('|');
        const regex = new RegExp(`(${pattern})`, 'gi');
        const parts = text.split(regex);
        const markClass = isNoli
            ? "bg-amber-200 dark:bg-amber-400/30 text-brand-navy dark:text-amber-100 hover:bg-amber-300 dark:hover:bg-amber-400/45"
            : "bg-rose-200 dark:bg-rose-500/30 text-[#2C1810] dark:text-rose-100 hover:bg-rose-300 dark:hover:bg-rose-500/45";

        return parts.map((part, index) => {
            const isKeyword = characterKeywords.some(k => k.toLowerCase() === part.toLowerCase());
            if (isKeyword) {
                return (
                    <mark key={index} onClick={(e) => { e.stopPropagation(); handleCharMarkClick(part); }} className={`${markClass} font-semibold px-1 rounded cursor-pointer hover:shadow-sm transition-all underline decoration-dotted decoration-brand-navy/40 dark:decoration-foreground/40`}>
                        {part}
                    </mark>
                );
            }
            return part;
        });
    };

    const { data: allChapters } = useQuery<any[]>({
        queryKey: ["chapters", mode],
        queryFn: () => fetchChapters(mode),
        staleTime: 1000 * 60 * 60 * 24,
    });

    const normalizedBook = (book === 'elfili' ? 'fili' : book).toLowerCase();
    const chapterTitleFromDb = allChapters?.find((c: any) => {
        const b = (c.book === 'elfili' ? 'fili' : c.book || '').toLowerCase();
        return b === normalizedBook && c.chapter_number === chapterNumber;
    })?.chapter_title || allChapters?.find((c: any) => {
        const b = (c.book === 'elfili' ? 'fili' : c.book || '').toLowerCase();
        return b === normalizedBook && c.chapter_number === chapterNumber;
    })?.title;

    const rawTitle = title && !title.toLowerCase().startsWith("kabanata") ? title : (chapterTitleFromDb || title);
    const cleanedTitle = cleanChapterTitle(rawTitle, chapterNumber);

    const activeTitleFromDb = allChapters?.find((c: any) => {
        const b = (c.book === 'elfili' ? 'fili' : c.book || '').toLowerCase();
        return b === normalizedBook && c.chapter_number === activeChapterInView;
    })?.chapter_title || allChapters?.find((c: any) => {
        const b = (c.book === 'elfili' ? 'fili' : c.book || '').toLowerCase();
        return b === normalizedBook && c.chapter_number === activeChapterInView;
    })?.title;

    const rawActiveTitle = chapterMetadata[activeChapterInView] || (activeChapterInView === chapterNumber ? rawTitle : activeTitleFromDb);
    const cleanedActiveTitle = cleanChapterTitle(rawActiveTitle, activeChapterInView);

    return (
        <>
            <AnimatePresence>
                {isOpen && (
                    <div id="chapter-modal" data-testid="chapter-modal" data-novel={modalNovel} className={`fixed inset-0 z-50 flex items-center justify-center ${isFullscreen ? 'p-0 h-[100dvh]' : 'p-2 sm:p-4 md:p-6'} overscroll-contain`}>
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={handleBackdropClick} className="absolute inset-0 bg-black/75 dark:bg-black/85 backdrop-blur-sm" />
                        <motion.div id="modal-shell" ref={modalRef} initial={{ opacity: 0, scale: 0.95, y: 20 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.95, y: 20 }} transition={{ type: "spring", damping: 25, stiffness: 300 }} className={`relative ${isFullscreen ? 'w-full h-[100dvh] bg-[#FAFAFA] dark:bg-[var(--modal-canvas-bg)]' : 'w-full max-w-4xl max-h-[92dvh] bg-brand-paper dark:bg-[var(--modal-shell-bg)] rounded-lg border border-brand-gold/15 dark:border-[var(--modal-border)]'} flex flex-col shadow-2xl overflow-hidden`}>
                            
                            {!isFullscreen && (
                                <div className="flex items-start justify-between p-3.5 sm:p-4 md:p-6 border-b border-brand-gold/10 dark:border-[var(--modal-border)] bg-brand-cream dark:bg-[var(--modal-header-bg)]">
                                    <div className="flex-1 pr-2">
                                        <span className={`text-[10px] font-bold tracking-[0.2em] uppercase ${accentColor}`}>
                                            {isNoli ? "Noli Me Tangere" : "El Filibusterismo"}
                                        </span>
                                        <h2 className="text-lg sm:text-xl md:text-2xl font-serif text-brand-navy dark:text-foreground mt-0.5 sm:mt-1 font-bold line-clamp-2">
                                            {cleanedTitle || `Kabanata ${chapterNumber}`}
                                        </h2>
                                        {cleanedTitle ? (
                                            <p className="text-brand-text/70 dark:text-brand-text-muted font-serif italic mt-0.5 sm:mt-1 text-xs sm:text-sm">
                                                Kabanata {chapterNumber}
                                            </p>
                                        ) : null}
                                    </div>
                                    <div className="flex items-center gap-1">
                                        <button type="button" onClick={() => setIsFullscreen(!isFullscreen)} className="min-w-[44px] min-h-[44px] flex items-center justify-center hover:bg-black/5 dark:hover:bg-white/10 active:scale-95 rounded-full transition-all text-brand-text/60 dark:text-foreground/70 hover:text-brand-navy dark:hover:text-brand-gold" title="Enter fullscreen" aria-label="I-fullscreen ang kabanata"><Maximize2 size={18} /></button>
                                        <button type="button" onClick={onClose} className="min-w-[44px] min-h-[44px] flex items-center justify-center hover:bg-black/5 dark:hover:bg-white/10 active:scale-95 rounded-full transition-all text-brand-text/60 dark:text-foreground/70 hover:text-brand-navy dark:hover:text-brand-gold" aria-label="Isara ang kabanata"><X size={22} /></button>
                                    </div>
                                </div>
                            )}

                            {!isFullscreen && (
                                <div className="flex flex-col sm:flex-row gap-2 sm:gap-4 p-3 sm:p-4 bg-brand-cream/50 dark:bg-[var(--modal-sub-bar-bg)] border-b border-brand-gold/10 dark:border-[var(--modal-border)] items-start sm:items-center justify-between">
                                    <div className="flex gap-2 overflow-x-auto no-scrollbar max-w-full py-0.5">
                                        <button type="button" onClick={() => setShowCharacters(!showCharacters)} className={`min-h-[44px] flex items-center gap-2 px-3.5 sm:px-4 py-2 rounded-lg sm:rounded-md transition-all whitespace-nowrap active:scale-95 ${showCharacters ? 'bg-brand-gold text-[#1B263B] font-bold shadow-sm' : 'bg-white/70 dark:bg-[var(--modal-button-bg)] text-brand-text dark:text-foreground/80 hover:bg-brand-gold/20 dark:hover:bg-brand-gold/20'}`}><Users size={16} /><span className="text-xs sm:text-sm font-bold uppercase tracking-wider">Tauhan</span></button>
                                        <button type="button" onClick={() => setShowReference(!showReference)} className={`min-h-[44px] flex items-center gap-2 px-3.5 sm:px-4 py-2 rounded-lg sm:rounded-md transition-all whitespace-nowrap active:scale-95 ${showReference ? 'bg-brand-gold text-[#1B263B] font-bold shadow-sm' : 'bg-white/70 dark:bg-[var(--modal-button-bg)] text-brand-text dark:text-foreground/80 hover:bg-brand-gold/20 dark:hover:bg-brand-gold/20'}`}>
                                            <BookOpen size={16} />
                                            <span className="text-xs sm:text-sm font-bold uppercase tracking-wider">Sanggunian</span>
                                            {showReference && isFullInverseLoading && (
                                                <span className="inline-block w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin ml-1" />
                                            )}
                                        </button>
                                    </div>
                                </div>
                            )}

                            <div className={`flex w-full min-h-0 flex-1 ${isFullscreen ? 'flex-row' : 'flex-col'}`}>
                                {isFullscreen && (
                                    <div className="hidden lg:flex w-80 flex-col justify-center h-full border-r border-brand-gold/10 dark:border-[var(--modal-border)] bg-[#F5F2EB]/50 dark:bg-[var(--modal-sidebar-bg)] p-12 relative pt-[max(3rem,env(safe-area-inset-top))]">
                                        <div className="absolute top-6 left-6 flex gap-2">
                                            <button onClick={() => setIsFullscreen(false)} className="min-w-[44px] min-h-[44px] flex items-center justify-center hover:bg-black/5 dark:hover:bg-white/10 rounded-full text-brand-text/40 dark:text-foreground/60 hover:text-brand-navy dark:hover:text-brand-gold transition-colors" title="Exit fullscreen"><Minimize2 size={20} /></button>
                                            <button onClick={onClose} className="min-w-[44px] min-h-[44px] flex items-center justify-center hover:bg-black/5 dark:hover:bg-white/10 rounded-full text-brand-text/40 dark:text-foreground/60 hover:text-brand-navy dark:hover:text-brand-gold transition-colors" title="Close"><X size={20} /></button>
                                        </div>
                                        <div className="flex flex-col space-y-8">
                                            <div>
                                                <span className={`text-xs font-bold tracking-[0.3em] uppercase ${accentColor} block mb-4`}>
                                                    {isNoli ? "Noli Me Tangere" : "El Filibusterismo"}
                                                </span>
                                                <h1 className="text-3xl lg:text-4xl font-serif text-brand-navy dark:text-foreground font-black leading-tight">
                                                    {cleanedActiveTitle || `Kabanata ${activeChapterInView}`}
                                                </h1>
                                                {cleanedActiveTitle ? (
                                                    <>
                                                        <div className="w-12 h-1 bg-brand-gold/30 my-6" />
                                                        <p className="text-base lg:text-lg font-serif italic text-brand-text/70 dark:text-brand-text-muted leading-relaxed">
                                                            Kabanata {activeChapterInView}
                                                        </p>
                                                    </>
                                                ) : null}
                                            </div>
                                            <div className="space-y-4 pt-4 border-t border-brand-gold/10 dark:border-brand-gold/20">
                                                <button onClick={() => setShowCharacters(!showCharacters)} className={`w-full text-left text-sm font-bold uppercase tracking-widest transition-colors ${showCharacters ? 'text-brand-gold font-black' : 'text-brand-text/50 dark:text-foreground/60 hover:text-brand-navy dark:hover:text-brand-gold'}`}>Tauhan</button>
                                                <button onClick={() => setShowReference(!showReference)} className={`w-full text-left text-sm font-bold uppercase tracking-widest transition-colors flex items-center justify-between ${showReference ? 'text-brand-gold font-black' : 'text-brand-text/50 dark:text-foreground/60 hover:text-brand-navy dark:hover:text-brand-gold'}`}>
                                                    <span>Sanggunian</span>
                                                    {showReference && isFullInverseLoading && (
                                                        <span className="inline-block w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin ml-1" />
                                                    )}
                                                </button>
                                                <p className="text-[10px] text-brand-text/40 dark:text-foreground/40 font-serif italic">I-click ang pangalan ng tauhan para makita ang Paksa</p>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                <div
                                    id="modal-reading-canvas"
                                    ref={scrollAreaRef}
                                    tabIndex={0}
                                    role="region"
                                    aria-label="Teksto ng kabanata"
                                    className={`flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-brand-gold/30 scrollbar-track-transparent bg-brand-paper dark:bg-[var(--modal-canvas-bg)] relative overscroll-contain focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-gold focus-visible:ring-inset ${isFullscreen ? '' : 'p-4 sm:p-6 md:p-14'}`}
                                >
                                    {isFullscreen && (
                                        <div className="lg:hidden sticky top-0 z-30 bg-brand-paper/95 dark:bg-[var(--modal-shell-bg)]/95 backdrop-blur border-b border-brand-gold/10 dark:border-[var(--modal-border)] px-4 pt-[max(0.75rem,env(safe-area-inset-top))] pb-3 flex flex-col gap-2">
                                            <div className="flex justify-between items-center">
                                                <div className="min-w-0 pr-2">
                                                    <span className={`text-[9px] font-bold uppercase tracking-wider ${accentColor} block truncate`}>
                                                        {isNoli ? "Noli Me Tangere" : "El Filibusterismo"}
                                                    </span>
                                                    <div className="text-sm sm:text-base font-serif font-bold text-brand-navy dark:text-foreground truncate">
                                                        {cleanedActiveTitle || `Kabanata ${activeChapterInView}`}
                                                    </div>
                                                    {cleanedActiveTitle ? (
                                                        <span className="text-[10px] text-brand-text/60 dark:text-foreground/60 block truncate">
                                                            Kabanata {activeChapterInView}
                                                        </span>
                                                    ) : null}
                                                </div>
                                                <div className="flex items-center gap-1 shrink-0">
                                                    <button type="button" onClick={() => setIsFullscreen(false)} className="min-w-[44px] min-h-[44px] flex items-center justify-center hover:bg-black/5 dark:hover:bg-white/10 active:scale-95 rounded-full text-brand-text/60 dark:text-foreground/70 hover:text-brand-navy dark:hover:text-brand-gold transition-all" title="Exit fullscreen" aria-label="I-exit ang fullscreen"><Minimize2 size={18} /></button>
                                                    <button type="button" onClick={onClose} className="min-w-[44px] min-h-[44px] flex items-center justify-center hover:bg-black/5 dark:hover:bg-white/10 active:scale-95 rounded-full text-brand-text/60 dark:text-foreground/70 hover:text-brand-navy dark:hover:text-brand-gold transition-all" title="Close" aria-label="Isara ang kabanata"><X size={20} /></button>
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-2 pt-1 border-t border-brand-gold/10 dark:border-[var(--modal-border)]">
                                                <button type="button" onClick={() => setShowCharacters(!showCharacters)} className={`min-h-[36px] flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-bold uppercase tracking-wider transition-all active:scale-95 ${showCharacters ? 'bg-brand-gold text-[#1B263B] font-bold shadow-sm' : 'bg-brand-cream/80 dark:bg-[var(--modal-button-bg)] text-brand-text/70 dark:text-foreground/80'}`}><Users size={14} /><span>Tauhan</span></button>
                                                <button type="button" onClick={() => setShowReference(!showReference)} className={`min-h-[36px] flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-bold uppercase tracking-wider transition-all active:scale-95 ${showReference ? 'bg-brand-gold text-[#1B263B] font-bold shadow-sm' : 'bg-brand-cream/80 dark:bg-[var(--modal-button-bg)] text-brand-text/70 dark:text-foreground/80'}`}>
                                                    <BookOpen size={14} />
                                                    <span>Sanggunian</span>
                                                    {showReference && isFullInverseLoading && (
                                                        <span className="inline-block w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin ml-1" />
                                                    )}
                                                </button>
                                            </div>
                                        </div>
                                    )}

                                    {isLoading && !isFullscreen ? (
                                        <div className="space-y-4 animate-pulse p-6 md:p-14">{[...Array(8)].map((_, i) => (<div key={i} className="h-4 bg-brand-gold/10 rounded w-full last:w-3/4" />))}</div>
                                    ) : (
                                        <div className={`${isFullscreen ? 'max-w-3xl mx-auto px-4 sm:px-8 py-6 sm:py-20 pb-[max(4rem,env(safe-area-inset-bottom))]' : 'max-w-3xl mx-auto mt-4'}`}>
                                            {isFullscreen && <div ref={topSentinelRef} className="h-40 flex items-center justify-center">{isLoadingMore && <div className="animate-pulse text-brand-gold font-serif italic text-sm">Kinakarga ang nakaraang kabanata...</div>}</div>}
                                            {modeError && (<div className="mb-8 p-4 bg-red-50 border border-red-200 rounded-md text-red-800 flex items-center justify-between"><span className="text-sm">{modeError}</span><button onClick={() => setModeError(null)} className="text-sm font-bold underline hover:no-underline ml-4 whitespace-nowrap">Dismiss</button></div>)}

                                            {(isFullscreen ? loadedChapters : [{ chapterNumber, title, content, isLoading }]).map((chap) => (
                                                <div key={`${chap.chapterNumber}`} id={`chapter-view-${chap.chapterNumber}`} className={isFullscreen ? "mb-40 min-h-[80vh]" : ""}>
                                                    {isFullscreen && (
                                                        <div className="mb-12 text-center opacity-80">
                                                            <span className={`text-[10px] font-bold uppercase tracking-[0.2em] ${accentColor} block mb-1`}>
                                                                {isNoli ? "Noli Me Tangere" : "El Filibusterismo"}
                                                            </span>
                                                            <h2 className="text-2xl sm:text-3xl font-serif font-bold text-brand-navy dark:text-foreground">
                                                                {cleanChapterTitle(chap.title, chap.chapterNumber) || `Kabanata ${chap.chapterNumber}`}
                                                            </h2>
                                                            {cleanChapterTitle(chap.title, chap.chapterNumber) ? (
                                                                <>
                                                                    <p className="text-xs sm:text-sm font-serif italic text-brand-text/60 dark:text-brand-text-muted mt-1">
                                                                        Kabanata {chap.chapterNumber}
                                                                    </p>
                                                                    <div className="w-8 h-0.5 bg-brand-gold/20 mx-auto mt-4" />
                                                                </>
                                                            ) : null}
                                                        </div>
                                                    )}
                                                    <div lang="tl" className="font-serif text-brand-text dark:text-foreground/90 leading-[2.1] sm:leading-[2.2] text-left [hyphens:auto] [-webkit-hyphens:auto] [text-wrap:pretty] space-y-6 sm:space-y-8 text-lg md:text-xl">
                                                        {chap.content.reduce((paragraphs: ChapterContent[][], sentence) => {
                                                            if (paragraphs.length === 0 || paragraphs[paragraphs.length - 1].length >= 6) paragraphs.push([sentence]);
                                                            else paragraphs[paragraphs.length - 1].push(sentence);
                                                            return paragraphs;
                                                        }, []).map((para, paraIdx) => (
                                                            <p key={`${paraIdx}-${showCharacters}-${showReference}`} className={paraIdx === 0 ? "drop-cap indent-0" : "indent-0"}>
                                                                {para.map((sentence) => {
                                                                    const isHighlighted = !isFullscreen && sentence.sentence_index === highlightSentenceIndex;
                                                                    // Map View only: context sentences surrounding the anchor (lighter highlight)
                                                                    const isContextSentence = !isHighlighted && !isFullscreen &&
                                                                        highlightContext != null &&
                                                                        sentence.sentence_index >= highlightContext.contextMin &&
                                                                        sentence.sentence_index <= highlightContext.contextMax;
                                                                    
                                                                    // Choose highlight class:
                                                                    // - Map View anchor: strong wash (highlightContext present + is anchor)
                                                                    // - Grid View anchor: clean novel wash (highlightContext absent)
                                                                    // - Map View context: lighter wash
                                                                    // - Unhighlighted: no class
                                                                    const anchorClass = isHighlighted
                                                                        ? (highlightContext
                                                                            ? (isNoli
                                                                                ? 'bg-amber-300/70 dark:bg-amber-400/35 font-bold border-b-2 border-amber-600 dark:border-amber-300 shadow-sm text-brand-navy dark:text-amber-50'
                                                                                : 'bg-rose-300/70 dark:bg-rose-500/35 font-bold border-b-2 border-rose-700 dark:border-rose-300 shadow-sm text-[#2C1810] dark:text-rose-50')
                                                                            : (isNoli
                                                                                ? 'bg-amber-200/60 dark:bg-amber-400/25 font-bold border-b-2 border-amber-500 dark:border-amber-400 text-brand-navy dark:text-amber-100'
                                                                                : 'bg-rose-200/60 dark:bg-rose-500/25 font-bold border-b-2 border-rose-600 dark:border-rose-400 text-[#2C1810] dark:text-rose-100'))
                                                                        : '';
                                                                    
                                                                    const contextClass = isContextSentence
                                                                        ? (isNoli
                                                                            ? 'bg-amber-100/80 dark:bg-amber-400/15 border-b-2 border-amber-400/70 dark:border-amber-400/40 text-brand-text dark:text-foreground'
                                                                            : 'bg-rose-100/80 dark:bg-rose-500/15 border-b-2 border-rose-400/70 dark:border-rose-400/40 text-brand-text dark:text-foreground')
                                                                        : '';

                                                                    const hoverClass = isNoli
                                                                        ? 'hover:bg-amber-500/10 dark:hover:bg-amber-400/15'
                                                                        : 'hover:bg-rose-500/10 dark:hover:bg-rose-400/15';
                                                                    
                                                                    let refRender = null;
                                                                    if (showReference) {
                                                                        if (mode === 'buod' && referenceResults[sentence.id]?.has_reference) {
                                                                            const refData = referenceResults[sentence.id];
                                                                            const isLowScore = refData.score < 0.70;
                                                                            const superscriptColor = isLowScore 
                                                                                ? "text-red-700 dark:text-red-300 cursor-pointer hover:text-red-900 ml-0.5 font-bold text-xs bg-red-500/15 px-1 rounded transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-red-500"
                                                                                : "text-amber-800 dark:text-brand-gold cursor-pointer hover:text-brand-navy ml-0.5 font-bold text-xs bg-brand-gold/15 dark:bg-brand-gold/20 px-1 rounded transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-brand-gold";
                                                                            refRender = (
                                                                                <sup
                                                                                    role="button"
                                                                                    tabIndex={0}
                                                                                    aria-label={`Sanggunian para sa pangungusap ${sentence.sentence_index}`}
                                                                                    onClick={(e) => { 
                                                                                        e.stopPropagation(); 
                                                                                        setSelectedSentenceForRef(sentence.sentence_index); 
                                                                                        setSelectedSourceSentence(sentence);
                                                                                        setSelectedFullMatches(null);
                                                                                        setRefFullscreen(true); 
                                                                                        setSelectedReference({
                                                                                            sentence_text: refData.reference_text,
                                                                                            chapter_number: chap.chapterNumber,
                                                                                            chapter_title: chap.title,
                                                                                            book: book,
                                                                                            mode: 'full',
                                                                                            alignment_status: refData.alignment_status,
                                                                                            matched_characters: refData.matched_characters,
                                                                                            score: refData.score,
                                                                                            semantic_score: refData.semantic_score,
                                                                                            lexical_score: refData.lexical_score,
                                                                                            char_score: refData.char_score,
                                                                                            ratio_score: refData.ratio_score,
                                                                                            buod_sentence_index: refData.buod_sentence_index,
                                                                                            full_sentence_indices: refData.full_sentence_indices,
                                                                                            full_is_short: refData.full_is_short
                                                                                        });
                                                                                    }}
                                                                                    onKeyDown={(e) => {
                                                                                        if (e.key === "Enter" || e.key === " ") {
                                                                                            e.preventDefault();
                                                                                            e.stopPropagation();
                                                                                            (e.currentTarget as HTMLElement).click();
                                                                                        }
                                                                                    }}
                                                                                    className={superscriptColor}
                                                                                >[{sentence.sentence_index}]</sup>
                                                                            );
                                                                        } else if (mode === 'full') {
                                                                            const matches = fullInverseMap.get(sentence.id);
                                                                            if (matches && matches.length > 0) {
                                                                                const bestScore = Math.max(...matches.map(m => m.score));
                                                                                const isLowScore = bestScore < 0.70;
                                                                                const superscriptColor = isLowScore 
                                                                                    ? "text-red-700 dark:text-red-300 cursor-pointer hover:text-red-900 ml-0.5 font-bold text-xs bg-red-500/15 px-1 rounded transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-red-500"
                                                                                    : "text-amber-800 dark:text-brand-gold cursor-pointer hover:text-brand-navy ml-0.5 font-bold text-xs bg-brand-gold/15 dark:bg-brand-gold/20 px-1 rounded transition-colors focus:outline-none focus-visible:ring-1 focus-visible:ring-brand-gold";
                                                                                refRender = (
                                                                                    <sup
                                                                                        role="button"
                                                                                        tabIndex={0}
                                                                                        aria-label={`Sanggunian para sa pangungusap ${sentence.sentence_index}`}
                                                                                        onClick={(e) => { 
                                                                                            e.stopPropagation(); 
                                                                                            setSelectedSentenceForRef(sentence.sentence_index); 
                                                                                            setSelectedSourceSentence(sentence);
                                                                                            setSelectedFullMatches(matches);
                                                                                            setSelectedReference(null);
                                                                                            setRefFullscreen(true); 
                                                                                        }}
                                                                                        onKeyDown={(e) => {
                                                                                            if (e.key === "Enter" || e.key === " ") {
                                                                                            e.preventDefault();
                                                                                            e.stopPropagation();
                                                                                            (e.currentTarget as HTMLElement).click();
                                                                                            }
                                                                                        }}
                                                                                        className={superscriptColor}
                                                                                    >[{sentence.sentence_index}]</sup>
                                                                                );
                                                                            }
                                                                        }
                                                                    }
                                                                    
                                                                    return (
                                                                        <span key={sentence.sentence_index} ref={isHighlighted ? highlightRef : null} className={`${hoverClass} transition-all duration-200 rounded px-0.5 relative inline ${anchorClass} ${contextClass}`}>
                                                                            {highlightText(sentence.sentence_text)}
                                                                            {refRender}
                                                                            {" "}
                                                                        </span>
                                                                    );
                                                                })}
                                                            </p>
                                                        ))}
                                                    </div>
                                                </div>
                                            ))}
                                            {isFullscreen && <div ref={bottomSentinelRef} className="h-40 flex items-center justify-center">{isLoadingMore && <div className="animate-pulse text-brand-gold font-serif italic text-sm">Dinatlo ang susunod na kabanata...</div>}</div>}
                                        </div>
                                    )}
                                </div>
                            </div>

                            {!isFullscreen && (
                                <div className="flex items-center justify-between p-3 sm:p-4 border-t border-brand-gold/10 dark:border-[var(--modal-border)] bg-brand-cream dark:bg-[var(--modal-header-bg)] pb-[max(0.75rem,env(safe-area-inset-bottom))]">
                                    <button onClick={handlePrevChapter} className="min-h-[44px] flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 rounded-md bg-white dark:bg-[var(--modal-button-bg)] text-brand-text dark:text-foreground hover:bg-brand-gold/10 dark:hover:bg-brand-gold/20 transition-all border border-brand-gold/10 dark:border-[var(--modal-border)] text-xs font-bold uppercase tracking-widest"><ChevronLeft size={16} /><span>Nakaraan</span></button>
                                    <div className={`hidden sm:block text-[10px] ${isNoli ? 'text-[#5C4033] dark:text-brand-gold' : 'text-[#8D2D2D] dark:text-[#E5A823]'} font-serif uppercase tracking-[0.3em] font-bold`}>Leksikal at Semantikong Pag-aaral</div>
                                    <button onClick={handleNextChapter} className="min-h-[44px] flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 rounded-md bg-white dark:bg-[var(--modal-button-bg)] text-brand-text dark:text-foreground hover:bg-brand-gold/10 dark:hover:bg-brand-gold/20 transition-all border border-brand-gold/10 dark:border-[var(--modal-border)] text-xs font-bold uppercase tracking-widest"><span>Susunod</span><ChevronRight size={16} /></button>
                                </div>
                            )}
                        </motion.div>


                        <AnimatePresence>
                            {refFullscreen && selectedSentenceForRef !== null && (
                                <motion.div id="sanggunian-overlay" data-novel={modalNovel} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-[100] bg-white dark:bg-[var(--modal-canvas-bg)] flex flex-col h-[100dvh] overscroll-contain text-brand-navy dark:text-foreground">
                                    <div className="border-b border-brand-gold/30 dark:border-[var(--modal-border)] bg-brand-cream/50 dark:bg-[var(--modal-header-bg)] pt-[max(0.75rem,env(safe-area-inset-top))] pb-3 sm:py-6 px-4 sm:px-6">
                                        <div className="max-w-7xl mx-auto text-center space-y-2 relative">
                                            <h2 className="text-xl sm:text-2xl font-serif text-brand-navy dark:text-foreground font-bold">Kaugnay na Sanggunian</h2>
                                            <p className="text-sm sm:text-lg text-brand-text dark:text-brand-text-muted">{isNoli ? "Noli Me Tangere" : "El Filibusterismo"} · Kabanata {chapterNumber}</p>
                                            <div className="flex items-center justify-center gap-2 pt-1 sm:pt-2">
                                                <span className="px-2.5 sm:px-3 py-0.5 sm:py-1 bg-brand-navy dark:bg-brand-gold text-white dark:text-brand-navy text-[9px] sm:text-[10px] font-bold uppercase rounded-full">Mula sa {mode === 'buod' ? 'Buod' : 'Buong Kwento'}</span>
                                                <span className="text-brand-gold">→</span>
                                                <span className="px-2.5 sm:px-3 py-0.5 sm:py-1 bg-brand-gold text-white dark:text-brand-navy text-[9px] sm:text-[10px] font-bold uppercase rounded-full">Patungo sa {mode === 'buod' ? 'Buong Kwento' : 'Buod'}</span>
                                            </div>
                                            <button type="button" onClick={() => { setRefFullscreen(false); setSelectedSentenceForRef(null); setSelectedReference(null); setSelectedSourceSentence(null); setSelectedFullMatches(null); }} className="absolute top-0 right-0 min-w-[44px] min-h-[44px] flex items-center justify-center hover:bg-black/5 dark:hover:bg-white/5 active:scale-95 rounded-full text-brand-navy dark:text-brand-gold transition-all" aria-label="Isara ang sanggunian"><X size={22} /></button>
                                        </div>
                                    </div>
                                    {/* Mobile View (< lg): Single unified scroll view with prominent comparison cards */}
                                    <div
                                        tabIndex={0}
                                        role="region"
                                        aria-label="Paghahambing ng sanggunian"
                                        className="flex-1 overflow-y-auto p-4 space-y-5 overscroll-contain lg:hidden pb-[max(3rem,env(safe-area-inset-bottom))] bg-brand-cream/30 dark:bg-[var(--modal-canvas-bg)] focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-gold focus-visible:ring-inset"
                                    >
                                        {/* Source sentence card - fully visible and prominent */}
                                        <div className="p-4 sm:p-5 bg-white dark:bg-[var(--modal-card-bg)] rounded-xl border border-brand-gold/30 dark:border-[var(--modal-card-border)] shadow-sm space-y-3">
                                            <div className="flex items-center justify-between border-b border-brand-gold/10 dark:border-[var(--modal-border)] pb-2">
                                                <span className="text-[11px] font-bold uppercase tracking-widest text-brand-navy/80 dark:text-brand-gold/90 flex items-center gap-1.5">
                                                    <Quote size={13} className="text-brand-gold shrink-0" />
                                                    Pinagmulang Teksto ({mode === 'buod' ? 'Buod' : 'Buong Kwento'})
                                                </span>
                                                <span className="text-[10px] font-bold text-brand-accent/70 uppercase tracking-wider">Kabanata {chapterNumber}</span>
                                            </div>
                                            <p className="font-serif text-brand-navy dark:text-foreground text-base sm:text-lg italic leading-relaxed text-left">
                                                &quot;{selectedSourceSentence?.sentence_text || chapContentOnRef()?.sentence_text}&quot;
                                            </p>
                                        </div>

                                        {/* Directional Connector */}
                                        <div className="flex items-center justify-center gap-2 text-[10px] font-bold uppercase tracking-widest text-brand-gold/80 px-2">
                                            <div className="h-px bg-brand-gold/20 flex-1" />
                                            <span>Nahanap na Katugma</span>
                                            <div className="h-px bg-brand-gold/20 flex-1" />
                                        </div>

                                        {/* Matched reference content */}
                                        {isRefLoading ? (
                                            <div className="flex flex-col items-center py-12 space-y-4 bg-white dark:bg-[var(--modal-card-bg)] rounded-xl border border-brand-gold/20 dark:border-[var(--modal-card-border)] p-6">
                                                <div className="w-10 h-10 border-2 border-brand-gold border-t-transparent rounded-full animate-spin" />
                                                <p className="text-sm font-serif italic text-brand-text/60 dark:text-brand-text-muted text-center">Sinisiyasat ng AI ang kabilang bersyon...</p>
                                            </div>
                                        ) : refError ? (
                                            <div className="py-8 px-6 bg-red-50 dark:bg-red-950/40 text-red-600 dark:text-red-300 rounded-xl border border-red-200 dark:border-red-900/40 text-center space-y-2">
                                                <X size={24} className="mx-auto text-red-400" />
                                                <p className="font-serif text-sm">{refError}</p>
                                            </div>
                                        ) : mode === 'full' ? (
                                            selectedFullMatches && selectedFullMatches.length > 0 && (
                                                <div className="space-y-5">
                                                    {selectedFullMatches.map((match, mIdx) => (
                                                        <div key={match.buodSentenceId || mIdx} className="space-y-5">
                                                            {/* Target sentence card */}
                                                            <div className="p-4 sm:p-5 bg-white dark:bg-[var(--modal-card-bg)] rounded-xl border border-brand-gold/30 dark:border-[var(--modal-card-border)] shadow-sm space-y-3">
                                                                <div className="flex justify-between items-center border-b border-brand-gold/10 dark:border-[var(--modal-border)] pb-2">
                                                                    <span className="text-[11px] font-bold text-brand-navy/80 dark:text-brand-gold/90 uppercase tracking-widest flex items-center gap-1.5">
                                                                        <BookOpen size={13} className="text-brand-gold shrink-0" />
                                                                        Lokasyon sa Buod {match.buod_chapter && match.buod_chapter !== chapterNumber ? `(Kabanata ${match.buod_chapter})` : ''}
                                                                    </span>
                                                                    {match.alignment_status && (
                                                                        <span className={`text-[9px] uppercase font-bold px-2 py-0.5 rounded-full ${match.alignment_status === 'precise' ? 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300' : 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300'}`}>
                                                                            {match.alignment_status}
                                                                        </span>
                                                                    )}
                                                                </div>
                                                                <p className="font-serif text-brand-navy dark:text-foreground text-base sm:text-lg leading-relaxed text-left">
                                                                    &quot;{match.text}&quot;
                                                                </p>
                                                                {match.matched_characters && match.matched_characters.length > 0 && (
                                                                    <div className="pt-3 border-t border-brand-gold/10 dark:border-[var(--modal-border)]">
                                                                        <span className="text-[10px] font-bold text-brand-navy/80 dark:text-brand-gold/90 uppercase tracking-wider block mb-2">Tauhan sa Sanggunian</span>
                                                                        <div className="flex flex-wrap gap-1.5">
                                                                            {match.matched_characters.map(c => (
                                                                                <span key={c} className="bg-brand-gold/10 text-brand-navy dark:text-brand-gold px-2 py-0.5 rounded text-xs font-bold">{c}</span>
                                                                            ))}
                                                                        </div>
                                                                    </div>
                                                                )}
                                                                <div className="pt-3 border-t border-brand-gold/10 dark:border-[var(--modal-border)]">
                                                                    <span className="text-[10px] font-bold text-brand-navy/80 dark:text-brand-gold/90 uppercase tracking-wider block mb-1">Posisyon sa CSV</span>
                                                                    <div className="bg-brand-navy/5 dark:bg-black/20 p-3 rounded-lg text-xs text-slate-600 dark:text-slate-300 font-medium space-y-1">
                                                                        <div><strong className="text-brand-navy dark:text-brand-gold">Pinagmulang Teksto:</strong> Pangungusap {selectedSourceSentence?.sentence_index || selectedSentenceForRef}</div>
                                                                        <div><strong className="text-brand-navy dark:text-brand-gold">Sanggunian:</strong> Pangungusap {match.sentence_index}{match.buod_chapter && match.buod_chapter !== chapterNumber ? ` (Buod Kabanata ${match.buod_chapter})` : ''}</div>
                                                                    </div>
                                                                </div>
                                                            </div>

                                                            {/* Similarity score card */}
                                                            <div className="p-4 sm:p-5 bg-white dark:bg-[var(--modal-card-bg)] rounded-xl border border-brand-gold/30 dark:border-[var(--modal-card-border)] shadow-sm space-y-4">
                                                                <div className="flex justify-between items-end">
                                                                    <span className="text-[11px] font-bold text-brand-navy dark:text-brand-gold uppercase tracking-widest">Antas ng Pagkakatulad (Alignment Score)</span>
                                                                    <span className={`text-2xl font-serif font-black ${match.score < 0.70 ? 'text-red-500/90' : 'text-brand-navy dark:text-brand-gold'}`}>
                                                                        {Math.round(match.score * 100)}%
                                                                    </span>
                                                                </div>
                                                                <ScoreVisualizer 
                                                                    semantic={match.semantic_score !== undefined ? Math.round(match.semantic_score * 100) : 0} 
                                                                    lexical={match.lexical_score !== undefined ? Math.round(match.lexical_score * 100) : 0}
                                                                    char={match.char_score === undefined || match.char_score === -1 ? -1 : Math.round(match.char_score * 100)}
                                                                    ratio={match.ratio_score !== undefined ? Math.round(match.ratio_score * 100) : 0}
                                                                />
                                                                <p className="text-[10px] text-brand-text/50 leading-relaxed italic">
                                                                    Ang porsyento ay kalkulado gamit ang Hybrid Scoring: 40% Kahulugan, 40% Salita, 10% Tauhan, at 10% Posisyon.
                                                                </p>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            )
                                        ) : selectedReference && (
                                            <div className="space-y-5">
                                                {/* Target sentence card */}
                                                <div className="p-4 sm:p-5 bg-white dark:bg-[var(--modal-card-bg)] rounded-xl border border-brand-gold/30 dark:border-[var(--modal-card-border)] shadow-sm space-y-3">
                                                    <div className="flex justify-between items-center border-b border-brand-gold/10 dark:border-[var(--modal-border)] pb-2">
                                                        <span className="text-[11px] font-bold text-brand-navy/80 dark:text-brand-gold/90 uppercase tracking-widest flex items-center gap-1.5">
                                                            <BookOpen size={13} className="text-brand-gold shrink-0" />
                                                            Lokasyon sa Buong Kwento
                                                        </span>
                                                        {selectedReference.alignment_status && (
                                                            <span className={`text-[9px] uppercase font-bold px-2 py-0.5 rounded-full ${selectedReference.alignment_status === 'precise' ? 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300' : 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300'}`}>
                                                                {selectedReference.alignment_status}
                                                            </span>
                                                        )}
                                                    </div>
                                                    <p className="font-serif text-brand-navy dark:text-foreground text-base sm:text-lg leading-relaxed text-left">
                                                        &quot;{selectedReference.sentence_text}&quot;
                                                    </p>
                                                    {selectedReference.matched_characters && selectedReference.matched_characters.length > 0 && (
                                                        <div className="pt-3 border-t border-brand-gold/10 dark:border-[var(--modal-border)]">
                                                            <span className="text-[10px] font-bold text-brand-navy/80 dark:text-brand-gold/90 uppercase tracking-wider block mb-2">Tauhan sa Sanggunian</span>
                                                            <div className="flex flex-wrap gap-1.5">
                                                                {selectedReference.matched_characters.map(c => (
                                                                    <span key={c} className="bg-brand-gold/10 text-brand-navy dark:text-brand-gold px-2 py-0.5 rounded text-xs font-bold">{c}</span>
                                                                ))}
                                                            </div>
                                                        </div>
                                                    )}
                                                    {selectedReference.buod_sentence_index !== undefined && (
                                                        <div className="pt-3 border-t border-brand-gold/10 dark:border-[var(--modal-border)]">
                                                            <span className="text-[10px] font-bold text-brand-navy/80 dark:text-brand-gold/90 uppercase tracking-wider block mb-1">Posisyon sa CSV</span>
                                                            <div className="bg-brand-navy/5 dark:bg-black/20 p-3 rounded-lg text-xs text-slate-600 dark:text-slate-300 font-medium space-y-1">
                                                                <div><strong className="text-brand-navy dark:text-brand-gold">Pinagmulang Teksto:</strong> Pangungusap {selectedReference.buod_sentence_index}</div>
                                                                <div><strong className="text-brand-navy dark:text-brand-gold">Sanggunian:</strong> Pangungusap {selectedReference.full_sentence_indices?.map((idx, i) => `${idx}${selectedReference.full_is_short?.[i] ? '*' : ''}`).join(', ')}</div>
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>

                                                {/* Similarity score card */}
                                                <div className="p-4 sm:p-5 bg-white dark:bg-[var(--modal-card-bg)] rounded-xl border border-brand-gold/30 dark:border-[var(--modal-card-border)] shadow-sm space-y-4">
                                                    <div className="flex justify-between items-end">
                                                        <span className="text-[11px] font-bold text-brand-navy dark:text-brand-gold uppercase tracking-widest">Antas ng Pagkakatulad</span>
                                                        <span className={`text-2xl font-serif font-black ${selectedReference.score < 0.70 ? 'text-red-500/90' : 'text-brand-navy dark:text-brand-gold'}`}>
                                                            {Math.round(selectedReference.score * 100)}%
                                                        </span>
                                                    </div>
                                                    <ScoreVisualizer 
                                                        semantic={Math.round(selectedReference.semantic_score * 100)} 
                                                        lexical={Math.round(selectedReference.lexical_score * 100)}
                                                        char={selectedReference.char_score === -1 ? -1 : Math.round(selectedReference.char_score * 100)}
                                                        ratio={Math.round(selectedReference.ratio_score * 100)}
                                                    />
                                                    <p className="text-[10px] text-brand-text/50 dark:text-brand-text-muted leading-relaxed italic">
                                                        Ang porsyento ay kalkulado gamit ang Hybrid Scoring: 40% Kahulugan, 40% Salita, 10% Tauhan, at 10% Posisyon.
                                                    </p>
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    {/* Desktop View (>= lg): Split Screen side-by-side */}
                                    <div className="hidden lg:flex flex-1 overflow-hidden flex-row">
                                        <div
                                            tabIndex={0}
                                            role="region"
                                            aria-label="Pinagmulang teksto ng sanggunian"
                                            className="flex-1 overflow-y-auto p-8 bg-brand-paper/30 dark:bg-[var(--modal-canvas-bg)] focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-gold focus-visible:ring-inset"
                                        >
                                            <div className="max-w-xl mx-auto space-y-6">
                                                <h3 className="text-xs font-bold uppercase tracking-widest text-brand-navy/80 dark:text-brand-gold/90 border-b border-brand-gold/10 dark:border-[var(--modal-border)] pb-2">Pinagmulang Teksto</h3>
                                                <p className="font-serif text-brand-text dark:text-foreground/90 leading-loose text-left text-xl italic">&quot;{selectedSourceSentence?.sentence_text || chapContentOnRef()?.sentence_text}&quot;</p>
                                            </div>
                                        </div>
                                        <div
                                            tabIndex={0}
                                            role="region"
                                            aria-label="Mga kaugnay na sanggunian"
                                            className="w-[500px] border-l border-brand-gold/30 dark:border-[var(--modal-border)] bg-white dark:bg-[var(--modal-shell-bg)] overflow-y-auto p-10 pb-[max(2rem,env(safe-area-inset-bottom))] focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-gold focus-visible:ring-inset"
                                        >
                                            <h3 className="text-xs font-bold uppercase tracking-widest text-brand-navy dark:text-brand-gold mb-8">Nahanap na Kaugnayan</h3>
                                            {isRefLoading ? (<div className="flex flex-col items-center py-20 space-y-4"><div className="w-10 h-10 border-2 border-brand-gold border-t-transparent rounded-full animate-spin" /><p className="text-sm font-serif italic text-brand-text/60 dark:text-brand-text-muted text-center">Sinisiyasat ng AI ang kabilang bersyon...</p></div>) : refError ? (<div className="py-10 text-center space-y-4"><div className="w-12 h-12 bg-red-50 dark:bg-red-950/40 text-red-400 rounded-full flex items-center justify-center mx-auto"><X size={24} /></div><p className="text-red-600 dark:text-red-300 font-serif">{refError}</p></div>) : mode === 'full' ? (
                                                selectedFullMatches && selectedFullMatches.length > 0 && (
                                                    <div className="space-y-8">
                                                        {selectedFullMatches.map((match, mIdx) => (
                                                            <div key={match.buodSentenceId || mIdx} className="space-y-6">
                                                                <div className="p-6 bg-brand-cream/20 dark:bg-[var(--modal-sub-bar-bg)] rounded-xl border border-brand-gold/10 dark:border-[var(--modal-border)]">
                                                                    <div className="mb-4 flex justify-between items-start">
                                                                        <div>
                                                                            <span className="text-[10px] font-bold text-brand-navy/80 dark:text-brand-gold/90 uppercase tracking-tighter">
                                                                                Lokasyon sa Buod {match.buod_chapter && match.buod_chapter !== chapterNumber ? `(Kabanata ${match.buod_chapter})` : ''}
                                                                            </span>
                                                                        </div>
                                                                        {match.alignment_status && (
                                                                            <span className={`text-[9px] uppercase font-bold px-2 py-0.5 rounded-full ${match.alignment_status === 'precise' ? 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300' : 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300'}`}>
                                                                                {match.alignment_status}
                                                                            </span>
                                                                        )}
                                                                    </div>
                                                                    <p className="text-xl font-serif leading-relaxed text-brand-text dark:text-foreground/90">&quot;{match.text}&quot;</p>
                                                                    
                                                                    {match.matched_characters && match.matched_characters.length > 0 && (
                                                                        <div className="mt-6 pt-4 border-t border-brand-gold/10 dark:border-[var(--modal-border)]">
                                                                            <span className="text-[10px] font-bold text-brand-navy/80 dark:text-brand-gold/90 uppercase tracking-tighter block mb-2">Tauhan sa Sanggunian</span>
                                                                            <div className="flex flex-wrap gap-2">
                                                                                {match.matched_characters.map(c => (
                                                                                    <span key={c} className="bg-brand-gold/10 text-brand-navy dark:text-brand-gold px-2 py-1 rounded text-xs font-bold">{c}</span>
                                                                                ))}
                                                                            </div>
                                                                        </div>
                                                                    )}

                                                                    <div className="mt-6 pt-4 border-t border-brand-gold/10 dark:border-[var(--modal-border)]">
                                                                        <span className="text-[10px] font-bold text-brand-navy/80 dark:text-brand-gold/90 uppercase tracking-tighter block mb-2">Posisyon sa CSV</span>
                                                                        <div className="bg-brand-navy/5 dark:bg-black/20 p-4 rounded-lg flex items-center justify-between shadow-sm border border-brand-navy/10 dark:border-white/10">
                                                                            <div className="flex flex-col text-sm text-slate-600 dark:text-slate-300 font-medium">
                                                                                <div><strong className="text-brand-navy dark:text-brand-gold">Pinagmulang Teksto:</strong> Pangungusap {selectedSourceSentence?.sentence_index || selectedSentenceForRef}</div>
                                                                                <div><strong className="text-brand-navy dark:text-brand-gold">Sanggunian:</strong> Pangungusap {match.sentence_index}{match.buod_chapter && match.buod_chapter !== chapterNumber ? ` (Buod Kabanata ${match.buod_chapter})` : ''}</div>
                                                                            </div>
                                                                        </div>
                                                                    </div>
                                                                </div>

                                                                <div className="p-6 bg-white dark:bg-[var(--modal-card-bg)] rounded-xl border border-brand-gold/10 dark:border-[var(--modal-card-border)] shadow-sm space-y-4">
                                                                    <div className="flex justify-between items-end">
                                                                        <span className="text-[10px] font-bold text-brand-navy dark:text-brand-gold uppercase tracking-widest">Antas ng Pagkakatulad (Alignment Score)</span>
                                                                        <span className={`text-2xl font-serif font-black ${match.score < 0.70 ? 'text-red-500/90' : 'text-brand-navy dark:text-brand-gold'}`}>
                                                                            {Math.round(match.score * 100)}%
                                                                        </span>
                                                                    </div>
                                                                    
                                                                    <ScoreVisualizer 
                                                                        semantic={match.semantic_score !== undefined ? Math.round(match.semantic_score * 100) : 0} 
                                                                        lexical={match.lexical_score !== undefined ? Math.round(match.lexical_score * 100) : 0}
                                                                        char={match.char_score === undefined || match.char_score === -1 ? -1 : Math.round(match.char_score * 100)}
                                                                        ratio={match.ratio_score !== undefined ? Math.round(match.ratio_score * 100) : 0}
                                                                    />
                                                                    
                                                                    <p className="text-[10px] text-brand-text/40 dark:text-brand-text-muted leading-relaxed italic mt-4">
                                                                        Ang porsyento ay kalkulado gamit ang Hybrid Scoring: 40% Kahulugan, 40% Salita, 10% Tauhan, at 10% Posisyon.
                                                                    </p>
                                                                </div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                )
                                            ) : selectedReference && (
                                                <div className="space-y-8">
                                                    <div className="p-6 bg-brand-cream/20 dark:bg-[var(--modal-sub-bar-bg)] rounded-xl border border-brand-gold/10 dark:border-[var(--modal-border)]">
                                                        <div className="mb-4 flex justify-between items-start">
                                                            <div>
                                                                <span className="text-[10px] font-bold text-brand-navy/80 dark:text-brand-gold/90 uppercase tracking-tighter">Lokasyon sa Buong Kwento</span>
                                                            </div>
                                                            {selectedReference.alignment_status && (
                                                                <span className={`text-[9px] uppercase font-bold px-2 py-0.5 rounded-full ${selectedReference.alignment_status === 'precise' ? 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300' : 'bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300'}`}>
                                                                    {selectedReference.alignment_status}
                                                                </span>
                                                            )}
                                                        </div>
                                                        <p className="text-xl font-serif leading-relaxed text-brand-text dark:text-foreground/90">&quot;{selectedReference.sentence_text}&quot;</p>
                                                        
                                                        {selectedReference.matched_characters && selectedReference.matched_characters.length > 0 && (
                                                            <div className="mt-6 pt-4 border-t border-brand-gold/10 dark:border-[var(--modal-border)]">
                                                                <span className="text-[10px] font-bold text-brand-navy/80 dark:text-brand-gold/90 uppercase tracking-tighter block mb-2">Tauhan sa Sanggunian</span>
                                                                <div className="flex flex-wrap gap-2">
                                                                    {selectedReference.matched_characters.map(c => (
                                                                        <span key={c} className="bg-brand-gold/10 text-brand-navy dark:text-brand-gold px-2 py-1 rounded text-xs font-bold">{c}</span>
                                                                    ))}
                                                                </div>
                                                            </div>
                                                        )}

                                                        {selectedReference.buod_sentence_index !== undefined && (
                                                            <div className="mt-6 pt-4 border-t border-brand-gold/10 dark:border-[var(--modal-border)]">
                                                                <span className="text-[10px] font-bold text-brand-navy/80 dark:text-brand-gold/90 uppercase tracking-tighter block mb-2">Posisyon sa CSV</span>
                                                                <div className="bg-brand-navy/5 dark:bg-black/20 p-4 rounded-lg flex items-center justify-between shadow-sm border border-brand-navy/10 dark:border-white/10">
                                                                    <div className="flex flex-col text-sm text-slate-600 dark:text-slate-300 font-medium">
                                                                        <div><strong className="text-brand-navy dark:text-brand-gold">Pinagmulang Teksto:</strong> Pangungusap {selectedReference.buod_sentence_index}</div>
                                                                        <div><strong className="text-brand-navy dark:text-brand-gold">Sanggunian:</strong> Pangungusap {selectedReference.full_sentence_indices?.map((idx, i) => `${idx}${selectedReference.full_is_short?.[i] ? '*' : ''}`).join(', ')}</div>
                                                                    </div>
                                                                </div>                                               </div>
                                                        )}
                                                    </div>

                                                    <div className="p-6 bg-white dark:bg-[var(--modal-card-bg)] rounded-xl border border-brand-gold/10 dark:border-[var(--modal-card-border)] shadow-sm space-y-4">
                                                        <div className="flex justify-between items-end">
                                                            <span className="text-[10px] font-bold text-brand-navy dark:text-brand-gold uppercase tracking-widest">Antas ng Pagkakatulad</span>
                                                            <span className={`text-2xl font-serif font-black ${selectedReference.score < 0.70 ? 'text-red-500/90' : 'text-brand-navy dark:text-brand-gold'}`}>
                                                                {Math.round(selectedReference.score * 100)}%
                                                            </span>
                                                        </div>
                                                        
                                                        <ScoreVisualizer 
                                                            semantic={Math.round(selectedReference.semantic_score * 100)} 
                                                            lexical={Math.round(selectedReference.lexical_score * 100)}
                                                            char={selectedReference.char_score === -1 ? -1 : Math.round(selectedReference.char_score * 100)}
                                                            ratio={Math.round(selectedReference.ratio_score * 100)}
                                                        />
                                                        
                                                        <p className="text-[10px] text-brand-text/40 dark:text-brand-text-muted leading-relaxed italic mt-4">
                                                            Ang porsyento ay kalkulado gamit ang Hybrid Scoring: 40% Kahulugan, 40% Salita, 10% Tauhan, at 10% Posisyon.
                                                        </p>
                                                    </div>
                                                    <div className="pt-6 border-t border-brand-gold/10 dark:border-[var(--modal-border)]"><p className="text-xs text-brand-text/50 dark:text-brand-text-muted leading-relaxed italic">Ang resultang ito ay nabuo sa pamamagitan ng Triple-Signal Segmentation at Hybrid Scoring na may Dynamic Position Window.</p></div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                )}
            </AnimatePresence>

            {selectedChar && (<ItemModal isOpen={isCharModalOpen} onClose={handleCharModalClose} title={selectedChar.name} subtitle={selectedChar.role} type="character" chapterAppearances={charAppearances} isLoading={charLoading} selectedNovel={isNoli ? 'noli' : 'fili'} onNavigate={(navBook, navChapter) => { handleCharModalClose(); onNavigate?.(navBook, navChapter); }} onSort={async (sMode) => { if (selectedChar) await fetchCharAppearances(selectedChar, sMode); }} sortBy={charSortBy} />)}
        </>
    );

    function chapContentOnRef() {
        return (isFullscreen ? loadedChapters.flatMap(c => c.content) : content).find(s => s.sentence_index === selectedSentenceForRef);
    }
}
