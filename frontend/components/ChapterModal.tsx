"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { X, Maximize2, Minimize2, ChevronLeft, ChevronRight, Users, BookOpen } from "lucide-react";
import { CHARACTERS, Character } from "@/lib/characterData";
import { ItemModal } from "@/components/ItemModal";
import { useModeStore } from "@/store/modeStore";
import { ScoreVisualizer } from "@/components/ScoreVisualizer";

interface ThemeMatch {
    id: string;
    label: string;
    score: number;
    explanation: string;
}

interface ChapterContent {
    id: number;
    is_short: boolean;
    sentence_index: number;
    sentence_text: string;
    themes: ThemeMatch[];
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
    onNavigate,
}: ChapterModalProps) {
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

    const handleReferenceClick = async (sentenceText: string, targetChapter: number) => {
        setIsRefLoading(true);
        setRefError(null);
        setSelectedReference(null);

        try {
            const targetMode = mode === 'buod' ? 'full' : 'buod';
            const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
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


    // Fetch References when toggled
    useEffect(() => {
        if (showReference && !isLoadingRefs) {
            const fetchRefs = async () => {
                setIsLoadingRefs(true);
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
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
                        } catch (e) {
                            results[s.id] = { has_reference: false };
                        }
                    }));
                    setReferenceResults({...results});
                }
                setIsLoadingRefs(false);
            };
            fetchRefs();
        }
    }, [showReference, isFullscreen, loadedChapters, content]);

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
                    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
                    const res = await fetch(`${apiUrl}/api/v1/chapters?mode=${mode}`);
                    if (res.ok) {
                        const data = await res.json();
                        const map: Record<number, string> = {};
                        data.filter((c: any) => (book.toLowerCase() === 'noli' ? c.book.toLowerCase() === 'noli' || c.book === 'Noli Me Tangere' : c.book.toLowerCase() === 'elfili' || c.book === 'El Filibusterismo'))
                            .forEach((c: any) => {
                                map[c.chapter_number] = c.chapter_title;
                            });
                        setChapterMetadata(map);
                    }
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
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
                const targetChapter = isFullscreen ? activeChapterInView : chapterNumber;
                const res = await fetch(`${apiUrl}/api/v1/chapters/${book}/${targetChapter}?mode=${mode}`, { signal: abortController.signal });

                if (!res.ok) {
                    if (res.status === 404) {
                        throw new Error(`Ang data ay hindi pa handa para sa Kabanata ${targetChapter} sa ${mode === 'full' ? 'Buong Kwento' : 'Buod'}.`);
                    }
                    throw new Error("Failed to fetch chapter content for the new mode");
                }

                const data = await res.json();

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
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        const targetMode = overrideMode || mode;
        const res = await fetch(`${apiUrl}/api/v1/chapters/${book}/${num}?mode=${targetMode}`, { signal });
        if (!res.ok) throw new Error("Failed");
        return await res.json();
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
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
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
        return parts.map((part, index) => {
            const isKeyword = characterKeywords.some(k => k.toLowerCase() === part.toLowerCase());
            if (isKeyword) {
                return (
                    <mark key={index} onClick={(e) => { e.stopPropagation(); handleCharMarkClick(part); }} className="bg-yellow-200 font-semibold px-1 rounded cursor-pointer hover:bg-yellow-300 hover:shadow-sm transition-all underline decoration-dotted decoration-brand-navy/40">
                        {part}
                    </mark>
                );
            }
            return part;
        });
    };

    const activeChapterTitle = chapterMetadata[activeChapterInView] || title;

    return (
        <>
            <AnimatePresence>
                {isOpen && (
                    <div key="chapter-modal" className={`fixed inset-0 z-50 flex items-center justify-center ${isFullscreen ? 'p-0' : 'p-4 sm:p-6'}`}>
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={handleBackdropClick} className="absolute inset-0 bg-brand-navy/70 backdrop-blur-sm" />
                        <motion.div ref={modalRef} initial={{ opacity: 0, scale: 0.95, y: 20 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.95, y: 20 }} transition={{ type: "spring", damping: 25, stiffness: 300 }} className={`relative ${isFullscreen ? 'w-full h-full bg-[#FAFAFA]' : 'w-full max-w-4xl max-h-[90vh] bg-brand-paper rounded-lg'} flex flex-col shadow-2xl overflow-hidden`}>
                            
                            {!isFullscreen && (
                                <div className="flex items-start justify-between p-4 md:p-6 border-b border-brand-gold/10 bg-brand-cream">
                                    <div className="flex-1">
                                        <span className={`text-[10px] font-bold tracking-[0.2em] uppercase ${accentColor}`}>{isNoli ? "Noli Me Tangere" : "El Filibusterismo"}</span>
                                        <h2 className="text-xl md:text-2xl font-serif text-brand-navy mt-1 font-bold">Kabanata {chapterNumber}</h2>
                                        <p className="text-brand-text/70 font-serif italic mt-1 text-sm">{title}</p>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <button onClick={() => setIsFullscreen(!isFullscreen)} className="p-2 hover:bg-black/5 rounded-full transition-colors text-brand-text/60 hover:text-brand-navy" title="Enter fullscreen"><Maximize2 size={20} /></button>
                                        <button onClick={onClose} className="p-2 hover:bg-black/5 rounded-full transition-colors text-brand-text/60 hover:text-brand-navy"><X size={24} /></button>
                                    </div>
                                </div>
                            )}

                            {!isFullscreen && (
                                <div className="flex flex-col sm:flex-row gap-4 p-4 bg-brand-cream/50 border-b border-brand-gold/10 items-start sm:items-center justify-between">
                                    <div className="flex gap-2 overflow-x-auto no-scrollbar max-w-full">
                                        <button onClick={() => setShowCharacters(!showCharacters)} className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all whitespace-nowrap ${showCharacters ? 'bg-brand-gold text-white shadow-sm' : 'bg-white/50 text-brand-text hover:bg-brand-gold/20'}`}><Users size={16} /><span className="text-sm font-bold uppercase tracking-widest">Tauhan</span></button>
                                        <button onClick={() => setShowReference(!showReference)} className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all whitespace-nowrap ${showReference ? 'bg-brand-gold text-white shadow-sm' : 'bg-white/50 text-brand-text hover:bg-brand-gold/20'}`}><BookOpen size={16} /><span className="text-sm font-bold uppercase tracking-widest">Sanggunian</span></button>
                                    </div>
                                </div>
                            )}

                            <div className={`flex w-full min-h-0 flex-1 ${isFullscreen ? 'flex-row' : 'flex-col'}`}>
                                {isFullscreen && (
                                    <div className="hidden lg:flex w-80 flex-col justify-center h-full border-r border-brand-gold/10 bg-[#F5F2EB]/50 p-12 relative">
                                        <div className="absolute top-6 left-6 flex gap-2">
                                            <button onClick={() => setIsFullscreen(false)} className="p-2 hover:bg-black/5 rounded-full text-brand-text/40 hover:text-brand-navy transition-colors" title="Exit fullscreen"><Minimize2 size={20} /></button>
                                            <button onClick={onClose} className="p-2 hover:bg-black/5 rounded-full text-brand-text/40 hover:text-brand-navy transition-colors" title="Close"><X size={20} /></button>
                                        </div>
                                        <div className="flex flex-col space-y-8">
                                            <div>
                                                <span className={`text-xs font-bold tracking-[0.3em] uppercase ${accentColor} block mb-4`}>{isNoli ? "Noli Me Tangere" : "El Filibusterismo"}</span>
                                                <h1 className="text-4xl font-serif text-brand-navy font-black leading-tight">Kabanata {activeChapterInView}</h1>
                                                <div className="w-12 h-1 bg-brand-gold/30 my-6" />
                                                <p className="text-xl font-serif italic text-brand-text/70 leading-relaxed">{activeChapterTitle}</p>
                                            </div>
                                            <div className="space-y-4 pt-4 border-t border-brand-gold/10">
                                                <button onClick={() => setShowCharacters(!showCharacters)} className={`w-full text-left text-sm font-bold uppercase tracking-widest transition-colors ${showCharacters ? 'text-brand-gold' : 'text-brand-text/50 hover:text-brand-navy'}`}>Tauhan</button>
                                                <button onClick={() => setShowReference(!showReference)} className={`w-full text-left text-sm font-bold uppercase tracking-widest transition-colors ${showReference ? 'text-brand-gold' : 'text-brand-text/50 hover:text-brand-navy'}`}>Sanggunian</button>
                                                <p className="text-[10px] text-brand-text/40 font-serif italic">I-click ang pangalan ng tauhan para makita ang Paksa</p>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                <div ref={scrollAreaRef} className={`flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-brand-gold/30 scrollbar-track-transparent bg-brand-paper relative ${isFullscreen ? '' : 'p-6 md:p-14'}`}>
                                    {isFullscreen && (
                                        <div className="lg:hidden sticky top-0 z-30 bg-brand-paper/95 backdrop-blur border-b border-brand-gold/10 p-4 flex justify-between items-center">
                                            <div><span className="text-[10px] font-bold uppercase tracking-wider text-brand-text/50">Kabanata {activeChapterInView}</span><div className="text-sm font-serif font-bold text-brand-navy truncate max-w-[200px]">{activeChapterTitle}</div></div>
                                            <div className="flex gap-2"><button onClick={() => setIsFullscreen(false)} className="p-2"><Minimize2 size={18} /></button><button onClick={onClose} className="p-2"><X size={18} /></button></div>
                                        </div>
                                    )}

                                    {isLoading && !isFullscreen ? (
                                        <div className="space-y-4 animate-pulse p-6 md:p-14">{[...Array(8)].map((_, i) => (<div key={i} className="h-4 bg-brand-gold/10 rounded w-full last:w-3/4" />))}</div>
                                    ) : (
                                        <div className={`${isFullscreen ? 'max-w-3xl mx-auto px-8 py-20' : 'max-w-3xl mx-auto mt-4'}`}>
                                            {isFullscreen && <div ref={topSentinelRef} className="h-40 flex items-center justify-center">{isLoadingMore && <div className="animate-pulse text-brand-gold font-serif italic text-sm">Kinakarga ang nakaraang kabanata...</div>}</div>}
                                            {modeError && (<div className="mb-8 p-4 bg-red-50 border border-red-200 rounded-md text-red-800 flex items-center justify-between"><span className="text-sm">{modeError}</span><button onClick={() => setModeError(null)} className="text-sm font-bold underline hover:no-underline ml-4 whitespace-nowrap">Dismiss</button></div>)}

                                            {(isFullscreen ? loadedChapters : [{ chapterNumber, title, content, isLoading }]).map((chap) => (
                                                <div key={`${chap.chapterNumber}`} id={`chapter-view-${chap.chapterNumber}`} className={isFullscreen ? "mb-40 min-h-[80vh]" : ""}>
                                                    {isFullscreen && (<div className="mb-12 text-center opacity-80"><span className="text-xs font-bold uppercase tracking-[0.2em] text-brand-text/30 block mb-2">Kabanata {chap.chapterNumber}</span><h2 className="text-3xl font-serif font-bold text-brand-navy">{chap.title}</h2><div className="w-8 h-0.5 bg-brand-gold/20 mx-auto mt-6" /></div>)}
                                                    <div className="font-serif text-brand-text leading-[2.2] text-justify space-y-8 text-lg md:text-xl">
                                                        {chap.content.reduce((paragraphs: ChapterContent[][], sentence) => {
                                                            if (paragraphs.length === 0 || paragraphs[paragraphs.length - 1].length >= 6) paragraphs.push([sentence]);
                                                            else paragraphs[paragraphs.length - 1].push(sentence);
                                                            return paragraphs;
                                                        }, []).map((para, paraIdx) => (
                                                            <p key={`${paraIdx}-${showCharacters}-${showReference}`} className="first-letter:float-left first-letter:text-[3.5rem] first-letter:font-serif first-letter:font-bold first-letter:leading-[0.8] first-letter:mr-2 indent-0">
                                                                {para.map((sentence) => {
                                                                    const isHighlighted = !isFullscreen && sentence.sentence_index === highlightSentenceIndex;
                                                                    
                                                                    let refRender = null;
                                                                    if (showReference && referenceResults[sentence.id]?.has_reference && referenceResults[sentence.id]?.score >= 70) {
                                                                        const refData = referenceResults[sentence.id];
                                                                        refRender = (
                                                                            <sup onClick={(e) => { 
                                                                                e.stopPropagation(); 
                                                                                setSelectedSentenceForRef(sentence.sentence_index); 
                                                                                setRefFullscreen(true); 
                                                                                setSelectedReference({
                                                                                    sentence_text: refData.reference_text,
                                                                                    chapter_number: chap.chapterNumber,
                                                                                    chapter_title: chap.title,
                                                                                    book: book,
                                                                                    mode: mode === 'buod' ? 'full' : 'buod',
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
                                                                            }} className="text-brand-gold cursor-pointer hover:text-brand-navy ml-0.5 font-bold text-xs bg-brand-gold/10 px-1 rounded transition-colors">[{sentence.sentence_index}]</sup>
                                                                        );
                                                                    }
                                                                    
                                                                    return (
                                                                        <span key={sentence.sentence_index} ref={isHighlighted ? highlightRef : null} className={`hover:bg-brand-gold/10 transition-all duration-200 rounded px-0.5 relative inline ${isHighlighted ? 'bg-brand-gold/20 font-bold border-b-2 border-brand-gold' : ''}`}>
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
                                <div className="flex items-center justify-between p-4 border-t border-brand-gold/10 bg-brand-cream">
                                    <button onClick={handlePrevChapter} className="flex items-center gap-2 px-4 py-2 rounded-md bg-white text-brand-text hover:bg-brand-gold/10 transition-all border border-brand-gold/10"><ChevronLeft size={18} /><span className="text-xs font-bold uppercase tracking-widest">Nakaraan</span></button>
                                    <div className="hidden sm:block text-[10px] text-brand-accent/60 font-serif uppercase tracking-[0.3em] font-bold">Leksikal at Semantikong Pag-aaral</div>
                                    <button onClick={handleNextChapter} className="flex items-center gap-2 px-4 py-2 rounded-md bg-white text-brand-text hover:bg-brand-gold/10 transition-all border border-brand-gold/10"><span className="text-xs font-bold uppercase tracking-widest">Susunod</span><ChevronRight size={18} /></button>
                                </div>
                            )}
                        </motion.div>


                        <AnimatePresence>
                            {refFullscreen && selectedSentenceForRef !== null && (
                                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-[100] bg-white flex flex-col">
                                    <div className="border-b border-brand-gold/30 bg-brand-cream/50 py-6">
                                        <div className="max-w-7xl mx-auto px-6 text-center space-y-2 relative">
                                            <h2 className="text-2xl font-serif text-brand-navy font-bold">Kaugnay na Sanggunian</h2>
                                            <p className="text-lg text-brand-text">{isNoli ? "Noli Me Tangere" : "El Filibusterismo"} · Kabanata {chapterNumber}</p>
                                            <div className="flex items-center justify-center gap-2 pt-2">
                                                <span className="px-3 py-1 bg-brand-navy text-white text-[10px] font-bold uppercase rounded-full">Mula sa {mode === 'buod' ? 'Buod' : 'Buong Kwento'}</span>
                                                <span className="text-brand-gold">→</span>
                                                <span className="px-3 py-1 bg-brand-gold text-white text-[10px] font-bold uppercase rounded-full">Patungo sa {mode === 'buod' ? 'Buong Kwento' : 'Buod'}</span>
                                            </div>
                                            <button onClick={() => { setRefFullscreen(false); setSelectedSentenceForRef(null); setSelectedReference(null); }} className="absolute top-0 right-0 p-2 hover:bg-black/5 rounded-full"><X size={24} className="text-brand-navy" /></button>
                                        </div>
                                    </div>
                                    <div className="flex-1 overflow-hidden flex">
                                        <div className="flex-1 overflow-y-auto p-8 bg-brand-paper/30"><div className="max-w-xl mx-auto space-y-6"><h3 className="text-xs font-bold uppercase tracking-widest text-brand-navy/40 border-b border-brand-gold/10 pb-2">Pinagmulang Teksto</h3><p className="font-serif text-brand-text leading-loose text-justify text-xl italic">&quot;{chapContentOnRef()?.sentence_text}&quot;</p></div></div>
                                        <div className="w-[500px] border-l border-brand-gold/30 bg-white overflow-y-auto p-10">
                                            <h3 className="text-xs font-bold uppercase tracking-widest text-brand-gold mb-8">Nahanap na Kaugnayan</h3>
                                            {isRefLoading ? (<div className="flex flex-col items-center py-20 space-y-4"><div className="w-10 h-10 border-2 border-brand-gold border-t-transparent rounded-full animate-spin" /><p className="text-sm font-serif italic text-brand-text/60 text-center">Sinisiyasat ng AI ang kabilang bersyon...</p></div>) : refError ? (<div className="py-10 text-center space-y-4"><div className="w-12 h-12 bg-red-50 text-red-400 rounded-full flex items-center justify-center mx-auto"><X size={24} /></div><p className="text-red-600 font-serif">{refError}</p></div>) : selectedReference && (
                                                <div className="space-y-8">
                                                    <div className="p-6 bg-brand-cream/20 rounded-xl border border-brand-gold/10">
                                                        <div className="mb-4 flex justify-between items-start">
                                                            <div>
                                                                <span className="text-[10px] font-bold text-brand-navy/40 uppercase tracking-tighter">Lokasyon sa {selectedReference.mode === 'full' ? 'Buong Kwento' : 'Buod'}</span>
                                                                {/* Just display the source chapter number if target chapter is not exactly known. It searches target-mode nearby chapters */}
                                                            </div>
                                                            {selectedReference.alignment_status && (
                                                                <span className={`text-[9px] uppercase font-bold px-2 py-0.5 rounded-full ${selectedReference.alignment_status === 'precise' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
                                                                    {selectedReference.alignment_status}
                                                                </span>
                                                            )}
                                                        </div>
                                                        <p className="text-xl font-serif leading-relaxed text-brand-text">&quot;{selectedReference.sentence_text}&quot;</p>
                                                        
                                                        {selectedReference.matched_characters && selectedReference.matched_characters.length > 0 && (
                                                            <div className="mt-6 pt-4 border-t border-brand-gold/10">
                                                                <span className="text-[10px] font-bold text-brand-navy/40 uppercase tracking-tighter block mb-2">Tauhan sa Sanggunian</span>
                                                                <div className="flex flex-wrap gap-2">
                                                                    {selectedReference.matched_characters.map(c => (
                                                                        <span key={c} className="bg-brand-gold/10 text-brand-navy px-2 py-1 rounded text-xs font-bold">{c}</span>
                                                                    ))}
                                                                </div>
                                                            </div>
                                                        )}

                                                        {selectedReference.buod_sentence_index !== undefined && (
                                                            <div className="mt-6 pt-4 border-t border-brand-gold/10">
                                                                <span className="text-[10px] font-bold text-brand-navy/40 uppercase tracking-tighter block mb-2">Posisyon sa CSV</span>
                                                                <div className="bg-brand-navy/5 p-4 rounded-lg flex items-center justify-between shadow-sm border border-brand-navy/10">
                                                                    <div className="flex flex-col text-sm text-slate-600 font-medium">
                                                                        <div><strong className="text-brand-navy">Pinagmulang Teksto:</strong> Pangungusap {selectedReference.buod_sentence_index}</div>
                                                                        <div><strong className="text-brand-navy">Sanggunian:</strong> Pangungusap {selectedReference.full_sentence_indices?.map((idx, i) => `${idx}${selectedReference.full_is_short?.[i] ? '*' : ''}`).join(', ')}</div>
                                                                    </div>
                                                                </div>                                               </div>
                                                        )}
                                                    </div>

                                                    <div className="p-6 bg-white rounded-xl border border-brand-gold/10 shadow-sm space-y-4">
                                                        <div className="flex justify-between items-end">
                                                            <span className="text-[10px] font-bold text-brand-gold uppercase tracking-widest">Antas ng Pagkakatulad</span>
                                                            <span className="text-2xl font-serif font-black text-brand-navy">
                                                                {Math.round(selectedReference.score * 100)}%
                                                            </span>
                                                        </div>
                                                        
                                                        <ScoreVisualizer 
                                                            semantic={Math.round(selectedReference.semantic_score * 100)} 
                                                            lexical={Math.round(selectedReference.lexical_score * 100)}
                                                            char={selectedReference.char_score === -1 ? -1 : Math.round(selectedReference.char_score * 100)}
                                                            ratio={Math.round(selectedReference.ratio_score * 100)}
                                                        />
                                                        
                                                        <p className="text-[10px] text-brand-text/40 leading-relaxed italic mt-4">
                                                            Ang porsyento ay kalkulado gamit ang Hybrid Scoring: 40% Kahulugan, 40% Salita, 10% Tauhan, at 10% Posisyon.
                                                        </p>
                                                    </div>
                                                    <div className="pt-6 border-t border-brand-gold/10"><p className="text-xs text-brand-text/50 leading-relaxed italic">Ang resultang ito ay nabuo sa pamamagitan ng Triple-Signal Segmentation at Hybrid Scoring na may Dynamic Position Window.</p></div>
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
