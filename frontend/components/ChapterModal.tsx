"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { X, Maximize2, Minimize2, ChevronLeft, ChevronRight, Users, BookOpen, Bookmark } from "lucide-react";
import { CHARACTERS, Character } from "@/lib/characterData";
import { ItemModal } from "@/components/ItemModal";
import { useModeStore } from "@/store/modeStore";

interface ThemeMatch {
    id: string;
    label: string;
    score: number;
    explanation: string;
}

interface ChapterContent {
    sentence_index: number;
    sentence_text: string;
    themes: ThemeMatch[];
}

interface BookmarkEntry {
    id: string;
    book: string;
    chapterNumber: number;
    sentenceIndex: number;
    selectedText: string;
    createdAt: number;
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

    // We don't need themeKeywords anymore as we use backend data
    // const [themeKeywords, setThemeKeywords] = useState<string[]>([]);

    const [showReference, setShowReference] = useState(false);
    const [themeFullscreen, setThemeFullscreen] = useState(false);
    const [selectedSentenceForTheme, setSelectedSentenceForTheme] = useState<number | null>(null);

    // Fullscreen multi-chapter state
    const [loadedChapters, setLoadedChapters] = useState<LoadedChapter[]>([]);
    const [chapterMetadata, setChapterMetadata] = useState<Record<number, string>>({});
    const [isLoadingMore, setIsLoadingMore] = useState(false);
    const bottomSentinelRef = useRef<HTMLDivElement>(null);
    const topSentinelRef = useRef<HTMLDivElement>(null);
    const scrollAreaRef = useRef<HTMLDivElement>(null);

    // State for tracking active chapter in fullscreen
    const [activeChapterInView, setActiveChapterInView] = useState<number>(chapterNumber);
    const initialFullscreenRef = useRef(false);
    const [bookmarks, setBookmarks] = useState<BookmarkEntry[]>([]);
    const [showBookmarksPanel, setShowBookmarksPanel] = useState(false);
    const [selectionToolbar, setSelectionToolbar] = useState<{
        top: number;
        left: number;
        sentence: ChapterContent;
        selectedText: string;
    } | null>(null);
    const [bookmarkSaveFeedback, setBookmarkSaveFeedback] = useState(false);

    // Sync active chapter with prop when NOT in fullscreen (background sync)
    useEffect(() => {
        if (!isFullscreen) {
            setActiveChapterInView(chapterNumber);
        }
    }, [chapterNumber, isFullscreen]);

    // --- Bookmark helpers (saves highlighted word only) ---
    const STORAGE_KEY = "rizalBookmarks-v2";

    const loadAllBookmarks = (): BookmarkEntry[] => {
        if (typeof window === "undefined") return [];
        try {
            const raw = window.localStorage.getItem(STORAGE_KEY);
            if (!raw) return [];
            const parsed = JSON.parse(raw);
            return Array.isArray(parsed)
                ? parsed.filter((b: any) => b.id && b.selectedText != null)
                : [];
        } catch {
            return [];
        }
    };

    const saveBookmarksForChapter = (chapterBookmarks: BookmarkEntry[]) => {
        if (typeof window === "undefined") return;
        const all = loadAllBookmarks().filter(
            (b) => !(b.book === book && b.chapterNumber === chapterNumber)
        );
        const merged = [...all, ...chapterBookmarks];
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(merged));
        setBookmarks(chapterBookmarks);
    };

    const addBookmark = (selectedText: string, sentenceIndex: number) => {
        const trimmed = selectedText.trim();
        if (!trimmed) return;
        const next: BookmarkEntry = {
            id: `bm-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
            book,
            chapterNumber,
            sentenceIndex,
            selectedText: trimmed,
            createdAt: Date.now(),
        };
        saveBookmarksForChapter([...bookmarks, next]);
    };

    const removeBookmark = (id: string) => {
        saveBookmarksForChapter(bookmarks.filter((b) => b.id !== id));
    };

    // Load bookmarks for the currently open chapter
    useEffect(() => {
        if (!isOpen) return;
        const all = loadAllBookmarks();
        setBookmarks(
            all.filter((b) => b.book === book && b.chapterNumber === chapterNumber)
        );
    }, [isOpen, book, chapterNumber]);

    // Text selection: show "Bookmark" toolbar above highlighted text
    useEffect(() => {
        if (!isOpen || isFullscreen) return;

        const handleSelection = () => {
            const sel = window.getSelection();
            if (!sel || sel.isCollapsed || !sel.rangeCount) {
                setSelectionToolbar(null);
                return;
            }
            const range = sel.getRangeAt(0);
            // Ensure selection is inside our modal content
            if (!modalRef.current?.contains(range.commonAncestorContainer)) {
                setSelectionToolbar(null);
                return;
            }
            const rect = range.getBoundingClientRect();
            if (rect.width === 0 && rect.height === 0) {
                setSelectionToolbar(null);
                return;
            }
            // Find sentence containing the selection (anchor node)
            let node: Node | null = range.startContainer;
            while (node && node !== modalRef.current) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    const el = node as Element;
                    const id = el.getAttribute?.("id");
                    if (id?.startsWith("sentence-")) {
                        const idx = parseInt(id.replace("sentence-", ""), 10);
                        const sentence = content.find((s) => s.sentence_index === idx);
                        const selectedText = range.toString().trim();
                        if (sentence && selectedText) {
                            setSelectionToolbar({
                                top: rect.top - 36,
                                left: rect.left + rect.width / 2,
                                sentence,
                                selectedText,
                            });
                            return;
                        }
                    }
                }
                node = node.parentNode;
            }
            setSelectionToolbar(null);
        };

        const handleMouseUp = () => {
            setTimeout(handleSelection, 10);
        };
        document.addEventListener("mouseup", handleMouseUp);
        document.addEventListener("selectionchange", handleSelection);

        return () => {
            document.removeEventListener("mouseup", handleMouseUp);
            document.removeEventListener("selectionchange", handleSelection);
        };
    }, [isOpen, isFullscreen, content]);

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
                            // Update only if the element is intersecting the "spy zone"
                            // Spy zone: Top 40% of the viewport (0px to -60%)
                            if (entry.isIntersecting) {
                                setActiveChapterInView(chap.chapterNumber);
                            }
                        });
                    },
                    {
                        // Threshold 0 means "as soon as one pixel is visible"
                        // RootMargin creates a "line" or "zone" effectively
                        // "-10% 0px -60% 0px" means:
                        // Top margin: -10% (triggers when element is near top)
                        // Bottom margin: -60% (ignores bottom part of screen)
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
                    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
                    const res = await fetch(`${apiUrl}/api/v1/chapters`);
                    if (res.ok) {
                        const data = await res.json();
                        const map: Record<number, string> = {};
                        data.filter((c: any) => (book === 'noli' ? c.book === 'noli' : c.book === 'elfili'))
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
    }, [isOpen, book]);

    // Initialize loadedChapters when entering fullscreen or when initial content changes
    useEffect(() => {
        if (isFullscreen) {
            setLoadedChapters([{
                chapterNumber,
                title,
                content: content || [],
                isLoading: false
            }]);

            // Allow time for render, then scroll to the chapter to hide top sentinel
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

        // Skip initial mount logic since `content` is populated by `page.tsx` on first open.
        // We only want to react to subsequent mode changes.
        if (prevModeRef.current === mode) return;

        let isMounted = true;
        const abortController = new AbortController();

        const fetchNewModeData = async () => {
            if (modeSwitchInProgress.current) return;
            modeSwitchInProgress.current = true;
            setModeError(null);

            try {
                const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

                // If we are in fullscreen, we might need to fetch all loaded chapters.
                // For now, let's just fetch the active chapter in view or the opened one to keep it simple and robust.
                const targetChapter = isFullscreen ? activeChapterInView : chapterNumber;

                const res = await fetch(`${apiUrl}/api/v1/chapters/${book}/${targetChapter}?mode=${mode}`, { signal: abortController.signal });

                if (!res.ok) {
                    if (res.status === 404) {
                        throw new Error(`Ang "Buong Kwento" ay hindi pa handa para sa Kabanata ${targetChapter}.`);
                    }
                    throw new Error("Failed to fetch chapter content for the new mode");
                }

                const data = await res.json();

                if (isMounted) {
                    // Update loaded chapters with the new data
                    setLoadedChapters(prev => prev.map(chap =>
                        chap.chapterNumber === targetChapter ? { ...chap, content: data, isLoading: false } : chap
                    ));
                }
            } catch (err: any) {
                if (err.name === 'AbortError') return;
                console.error("Mode switch fetch error:", err);
                if (isMounted) {
                    setModeError(err.message || 'Error occurred while switching modes.');
                    // If it failed, we technically shouldn't revert the global mode, 
                    // we just show the error and let them optionally click back.
                }
            } finally {
                if (isMounted) {
                    modeSwitchInProgress.current = false;
                    prevModeRef.current = mode; // Only update ref after a successful fetch cycle
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
                if (isFullscreen) {
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
    }, [isOpen, isFullscreen, onClose]);

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

    // Handle click outside (only when not fullscreen)
    const handleBackdropClick = (e: React.MouseEvent) => {
        if (!isFullscreen && modalRef.current && !modalRef.current.contains(e.target as Node)) {
            onClose();
        }
    };

    const isNoli = book === "noli";
    const accentColor = isNoli ? "text-noli-accent" : "text-fili-accent";
    const borderColor = isNoli ? "border-noli-accent" : "border-fili-accent";

    // Standard chapter counts
    const NOLI_CHAPTERS = 64;
    const FILI_CHAPTERS = 39;

    const handlePrevChapter = () => {
        if (!onNavigate) return;

        const maxChapters = isNoli ? NOLI_CHAPTERS : FILI_CHAPTERS;

        if (chapterNumber <= 1) {
            // Loop to last chapter
            onNavigate(book, maxChapters);
        } else {
            onNavigate(book, chapterNumber - 1);
        }
    };

    const handleNextChapter = () => {
        if (!onNavigate) return;

        const maxChapters = isNoli ? NOLI_CHAPTERS : FILI_CHAPTERS;

        if (chapterNumber >= maxChapters) {
            // Loop to first chapter
            onNavigate(book, 1);
        } else {
            onNavigate(book, chapterNumber + 1);
        }
    };

    const fetchSpecificChapter = async (num: number, overrideMode?: 'buod' | 'full', signal?: AbortSignal): Promise<ChapterContent[]> => {
        try {
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
            const targetMode = overrideMode || mode;
            const res = await fetch(`${apiUrl}/api/v1/chapters/${book}/${num}?mode=${targetMode}`, { signal });
            if (!res.ok) {
                if (res.status === 404 && targetMode === 'full') {
                    throw new Error("Full version not available for this chapter.");
                }
                throw new Error("Failed");
            }
            return await res.json();
        } catch (err: any) {
            console.error(err);
            throw err;
        }
    };

    const loadNextChapter = async () => {
        if (isLoadingMore || loadedChapters.length === 0) return;
        const lastChapter = loadedChapters[loadedChapters.length - 1];
        const nextNum = lastChapter.chapterNumber + 1;
        const maxChapters = isNoli ? NOLI_CHAPTERS : FILI_CHAPTERS;

        if (nextNum > maxChapters) return;

        setIsLoadingMore(true);
        const content = await fetchSpecificChapter(nextNum);
        setLoadedChapters(prev => [...prev, {
            chapterNumber: nextNum,
            title: chapterMetadata[nextNum] || `Kabanata ${nextNum}`,
            content,
            isLoading: false
        }]);
        setIsLoadingMore(false);
    };

    const loadPrevChapter = async () => {
        if (isLoadingMore || loadedChapters.length === 0) return;
        const firstChapter = loadedChapters[0];
        const prevNum = firstChapter.chapterNumber - 1;

        if (prevNum < 1) return;

        setIsLoadingMore(true);

        // 1. Capture current scroll state
        const container = scrollAreaRef.current;
        const prevScrollHeight = container?.scrollHeight || 0;
        const prevScrollTop = container?.scrollTop || 0;

        const content = await fetchSpecificChapter(prevNum);

        setLoadedChapters(prev => [{
            chapterNumber: prevNum,
            title: chapterMetadata[prevNum] || `Kabanata ${prevNum}`,
            content,
            isLoading: false
        }, ...prev]);

        // 2. Adjust scroll position after render to anchor the view
        // Using requestAnimationFrame to ensure the DOM has updated with new chapters
        requestAnimationFrame(() => {
            if (container) {
                const newScrollHeight = container.scrollHeight;
                const heightDiff = newScrollHeight - prevScrollHeight;
                // Add the height of the new content to the scroll position to keep the current view stabilized
                container.scrollTop = prevScrollTop + heightDiff;

                // Delay clearing the loading state to ensure sentinel isn't triggered immediately
                setTimeout(() => setIsLoadingMore(false), 200);
            } else {
                setIsLoadingMore(false);
            }
        });
    };

    // Intersection Observer for infinite scroll (Top/Bottom)
    useEffect(() => {
        if (!isFullscreen) return;

        const bottomObserver = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) {
                loadNextChapter();
            }
        }, { threshold: 0.1 });

        // Delay top observer to prevent immediate "prev" loading on init
        let topObserver: IntersectionObserver | null = null;

        const timer = setTimeout(() => {
            topObserver = new IntersectionObserver((entries) => {
                if (entries[0].isIntersecting) {
                    loadPrevChapter();
                }
            }, { threshold: 0.1 });

            if (topSentinelRef.current) {
                topObserver.observe(topSentinelRef.current);
            }
        }, 1000); // 1-second delay before enabling "previous" loading

        if (bottomSentinelRef.current) {
            bottomObserver.observe(bottomSentinelRef.current);
        }

        return () => {
            bottomObserver.disconnect();
            if (topObserver) topObserver.disconnect();
            clearTimeout(timer);
        };
    }, [isFullscreen, loadedChapters, isLoadingMore]);

    // ---------- Inline character modal helpers ----------
    const fetchCharAppearances = async (char: Character, sort: 'number' | 'relevance') => {
        setCharLoading(true);
        try {
            const searchTerms = [char.name, ...(char.aliases || [])];
            const searchTerm = searchTerms.join(",");
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
            const res = await fetch(
                `${apiUrl}/api/v1/characters/chapters?name=${encodeURIComponent(searchTerm)}&sort_by=${sort}`
            );
            if (!res.ok) throw new Error("Failed to fetch chapters");
            const data = await res.json();
            setCharAppearances(data);
        } catch (err) {
            console.error("Error fetching character chapters:", err);
        } finally {
            setCharLoading(false);
        }
    };

    const handleCharMarkClick = async (clickedText: string) => {
        // Find the Character whose name or alias matches the clicked text
        const match = CHARACTERS.find(c =>
            c.name.toLowerCase() === clickedText.toLowerCase() ||
            (c.aliases || []).some(a => a.toLowerCase() === clickedText.toLowerCase())
        );
        if (!match) return;
        setSelectedChar(match);
        setIsCharModalOpen(true);
        setCharAppearances([]);
        setCharSortBy('number');
        await fetchCharAppearances(match, 'number');
    };

    const handleCharModalClose = () => {
        setIsCharModalOpen(false);
        setTimeout(() => {
            setSelectedChar(null);
            setCharAppearances([]);
        }, 300);
    };

    // Helper function to highlight keywords in text
    const highlightText = (text: string) => {
        // Only highlight characters
        if (!showCharacters) {
            return text;
        }

        const keywords = characterKeywords;

        if (keywords.length === 0) {
            return text;
        }

        // Build a regex that works for:
        //   - Multi-word names  ("Maria Clara", "Padre Damaso")
        //   - Hyphenated names  ("Ben-Zayb", "Ka-Tales")
        //   - Single words      ("Ibarra", "Simoun")
        //
        // Strategy: use a negative lookbehind for word-char at the start and
        // a negative lookahead for word-char at the end.  This means the
        // matched name must not be immediately preceded or followed by a
        // letter, digit, or underscore – i.e. it must stand alone as a token.
        const pattern = keywords
            .map(k => {
                const escaped = k.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                return `(?<![\\wÀ-ÖØ-öø-ÿ])${escaped}(?![\\wÀ-ÖØ-öø-ÿ])`;
            })
            .join('|');

        const regex = new RegExp(`(${pattern})`, 'gi');

        const parts = text.split(regex);

        return parts.map((part, index) => {
            const isKeyword = keywords.some(k => k.toLowerCase() === part.toLowerCase());
            if (isKeyword) {
                return (
                    <mark
                        key={index}
                        onClick={(e) => { e.stopPropagation(); handleCharMarkClick(part); }}
                        className="bg-yellow-200 font-semibold px-1 rounded cursor-pointer hover:bg-yellow-300 hover:shadow-sm transition-all underline decoration-dotted decoration-brand-navy/40"
                        title={`View character: ${part}`}
                    >
                        {part}
                    </mark>
                );
            }
            return part;
        });
    };

    const selectedSentence = content.find(s => s.sentence_index === selectedSentenceForTheme);

    // Get metadata for the active chapter
    const activeChapterTitle = chapterMetadata[activeChapterInView] || title;

    return (
        <>
            <AnimatePresence>
                {isOpen && (
                    <div key="chapter-modal" className={`fixed inset-0 z-50 flex items-center justify-center ${isFullscreen ? 'p-0' : 'p-4 sm:p-6'}`}>
                        {/* Backdrop */}
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={handleBackdropClick}
                            className="absolute inset-0 bg-brand-navy/70 backdrop-blur-sm"
                        />

                        {/* Wrapper for modal + external bookmark icon */}
                        <div className={`relative ${isFullscreen ? 'w-full h-full' : ''}`}>
                            {/* Bookmark icon – outside the box, hanging at top-left corner (non-fullscreen) */}
                            {!isFullscreen && (
                                <button
                                    type="button"
                                    onClick={() => setShowBookmarksPanel(!showBookmarksPanel)}
                                    className={`absolute left-0 top-0 -translate-x-1/2 -translate-y-1/2 z-10 inline-flex h-9 w-9 items-center justify-center rounded-full border-2 shadow-lg transition-all ${
                                        showBookmarksPanel
                                            ? 'bg-brand-gold text-white border-brand-gold'
                                            : bookmarks.length > 0
                                                ? 'bg-white text-brand-gold border-brand-gold/50 hover:bg-brand-gold/10'
                                                : 'bg-white text-brand-text/50 border-brand-gold/30 hover:bg-brand-gold/10 hover:text-brand-navy'
                                    }`}
                                    title="Ipakita ang mga bookmark"
                                >
                                    <Bookmark size={16} className={bookmarks.length > 0 ? 'fill-current' : 'fill-none'} />
                                </button>
                            )}

                        {/* Modal Container */}
                        <motion.div
                            ref={modalRef}
                            initial={{ opacity: 0, scale: 0.95, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.95, y: 20 }}
                            transition={{ type: "spring", damping: 25, stiffness: 300 }}
                            className={`relative ${isFullscreen ? 'w-full h-full bg-[#FAFAFA]' : 'w-full max-w-4xl max-h-[90vh] bg-brand-paper rounded-lg'} flex flex-col shadow-2xl overflow-hidden`}
                        >
                            {/* Standard Header (Non-Fullscreen) */}
                            {!isFullscreen && (
                                <div className={`flex items-start justify-between p-4 md:p-6 border-b border-brand-gold/10 bg-brand-cream`}>
                                    <div className="flex-1">
                                        <span className={`text-[10px] font-bold tracking-[0.2em] uppercase ${accentColor}`}>
                                            {isNoli ? "Noli Me Tangere" : "El Filibusterismo"}
                                        </span>
                                        <h2 className="text-xl md:text-2xl font-serif text-brand-navy mt-1 font-bold">
                                            Kabanata {chapterNumber}
                                        </h2>
                                        <p className="text-brand-text/70 font-serif italic mt-1 text-sm">
                                            {title}
                                        </p>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => setIsFullscreen(!isFullscreen)}
                                            className="p-2 hover:bg-black/5 rounded-full transition-colors text-brand-text/60 hover:text-brand-navy"
                                            title="Enter fullscreen"
                                        >
                                            <Maximize2 size={20} />
                                        </button>
                                        <button
                                            onClick={onClose}
                                            className="p-2 hover:bg-black/5 rounded-full transition-colors text-brand-text/60 hover:text-brand-navy"
                                        >
                                            <X size={24} />
                                        </button>
                                    </div>
                                </div>
                            )}

                            {/* Standard Actions (Non-Fullscreen) */}
                            {!isFullscreen && (
                                <div className="flex flex-col sm:flex-row gap-4 p-4 bg-brand-cream/50 border-b border-brand-gold/10 items-start sm:items-center justify-between">
                                    <div className="flex gap-2 overflow-x-auto no-scrollbar max-w-full">
                                        <button
                                            onClick={() => setShowCharacters(!showCharacters)}
                                            className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all whitespace-nowrap ${showCharacters ? 'bg-brand-gold text-white shadow-sm' : 'bg-white/50 text-brand-text hover:bg-brand-gold/20'}`}
                                        >
                                            <Users size={16} />
                                            <span className="text-sm font-bold uppercase tracking-widest">Tauhan</span>
                                        </button>
                                        <button
                                            onClick={() => setShowThemes(!showThemes)}
                                            className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all whitespace-nowrap ${showThemes ? 'bg-brand-gold text-white shadow-sm' : 'bg-white/50 text-brand-text hover:bg-brand-gold/20'}`}
                                        >
                                            <BookOpen size={16} />
                                            <span className="text-sm font-bold uppercase tracking-widest">Paksa</span>
                                        </button>
                                        <button
                                            onClick={() => setShowReference(!showReference)}
                                            className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all whitespace-nowrap ${showReference ? 'bg-brand-gold text-white shadow-sm' : 'bg-white/50 text-brand-text hover:bg-brand-gold/20'}`}
                                        >
                                            <BookOpen size={16} />
                                            <span className="text-sm font-bold uppercase tracking-widest">Sanggunian</span>
                                        </button>
                                    </div>
                                </div>
                            )}

                            {/* Fullscreen Layout Container */}
                            <div className={`flex w-full min-h-0 flex-1 ${isFullscreen ? 'flex-row' : 'flex-col'}`}>

                                {/* Fullscreen Sidebar / Info Panel */}
                                {isFullscreen && (
                                    <div className="hidden lg:flex w-80 flex-col justify-center h-full border-r border-brand-gold/10 bg-[#F5F2EB]/50 p-12 relative">
                                        {/* Close/Minimize Controls */}
                                        <div className="absolute top-6 left-6 flex gap-2">
                                            <button
                                                onClick={() => setIsFullscreen(false)}
                                                className="p-2 hover:bg-black/5 rounded-full text-brand-text/40 hover:text-brand-navy transition-colors"
                                                title="Exit fullscreen"
                                            >
                                                <Minimize2 size={20} />
                                            </button>
                                            <button
                                                onClick={onClose}
                                                className="p-2 hover:bg-black/5 rounded-full text-brand-text/40 hover:text-brand-navy transition-colors"
                                                title="Close"
                                            >
                                                <X size={20} />
                                            </button>
                                        </div>

                                        <div className="flex flex-col space-y-8">
                                            <div>
                                                <span className={`text-xs font-bold tracking-[0.3em] uppercase ${accentColor} block mb-4`}>
                                                    {isNoli ? "Noli Me Tangere" : "El Filibusterismo"}
                                                </span>
                                                <h1 className="text-4xl font-serif text-brand-navy font-black leading-tight">
                                                    Kabanata {activeChapterInView}
                                                </h1>
                                                <div className="w-12 h-1 bg-brand-gold/30 my-6" />
                                                <p className="text-xl font-serif italic text-brand-text/70 leading-relaxed">
                                                    {activeChapterTitle}
                                                </p>
                                            </div>

                                            {/* Menu Links */}
                                            <div className="space-y-4 pt-4 border-t border-brand-gold/10">
                                                <button
                                                    onClick={() => setShowCharacters(!showCharacters)}
                                                    className={`w-full text-left text-sm font-bold uppercase tracking-widest transition-colors ${showCharacters ? 'text-brand-gold' : 'text-brand-text/50 hover:text-brand-navy'}`}
                                                >
                                                    Tauhan
                                                </button>
                                                <button
                                                    onClick={() => setShowThemes(!showThemes)}
                                                    className={`w-full text-left text-sm font-bold uppercase tracking-widest transition-colors ${showThemes ? 'text-brand-gold' : 'text-brand-text/50 hover:text-brand-navy'}`}
                                                >
                                                    Paksa
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Selection toolbar: "Bookmark" above highlighted text */}
                                {selectionToolbar && !isFullscreen && (
                                    <div
                                        className="fixed z-[54] -translate-x-1/2 -translate-y-full"
                                        style={{
                                            top: selectionToolbar.top,
                                            left: selectionToolbar.left,
                                        }}
                                    >
                                        {bookmarkSaveFeedback ? (
                                            <div className="rounded-md bg-green-600 text-white px-3 py-1.5 text-xs font-bold uppercase tracking-wider shadow-lg flex items-center gap-1.5">
                                                <span>✓</span>
                                                <span>Na-bookmark!</span>
                                            </div>
                                        ) : (
                                            <button
                                                type="button"
                                                onClick={() => {
                                                    addBookmark(
                                                        selectionToolbar.selectedText,
                                                        selectionToolbar.sentence.sentence_index
                                                    );
                                                    window.getSelection()?.removeAllRanges();
                                                    setBookmarkSaveFeedback(true);
                                                    setTimeout(() => {
                                                        setBookmarkSaveFeedback(false);
                                                        setSelectionToolbar(null);
                                                    }, 1500);
                                                }}
                                                className="rounded-md bg-brand-navy text-white px-3 py-1.5 text-xs font-bold uppercase tracking-wider shadow-lg hover:bg-brand-navy/90 transition-colors"
                                            >
                                                Bookmark
                                            </button>
                                        )}
                                    </div>
                                )}

                                {/* Main Content Scroll Area */}
                                <div ref={scrollAreaRef} className={`flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-brand-gold/30 scrollbar-track-transparent bg-brand-paper relative ${isFullscreen ? '' : 'p-6 md:p-14'}`}>
                                    {/* Mobile/Tablet Fullscreen Header Overlay (since sidebar hidden) */}
                                    {isFullscreen && (
                                        <div className="lg:hidden sticky top-0 z-30 bg-brand-paper/95 backdrop-blur border-b border-brand-gold/10 p-4 flex justify-between items-center">
                                            <div>
                                                <span className="text-[10px] font-bold uppercase tracking-wider text-brand-text/50">
                                                    Kabanata {activeChapterInView}
                                                </span>
                                                <div className="text-sm font-serif font-bold text-brand-navy truncate max-w-[200px]">
                                                    {activeChapterTitle}
                                                </div>
                                            </div>
                                            <div className="flex gap-2">
                                                <button onClick={() => setIsFullscreen(false)} className="p-2"><Minimize2 size={18} /></button>
                                                <button onClick={onClose} className="p-2"><X size={18} /></button>
                                            </div>
                                        </div>
                                    )}

                                    {isLoading && !isFullscreen ? (
                                        <div className="space-y-4 animate-pulse p-6 md:p-14">
                                            {[...Array(8)].map((_, i) => (
                                                <div key={i} className="h-4 bg-brand-gold/10 rounded w-full last:w-3/4" />
                                            ))}
                                        </div>
                                    ) : (
                                        <div className={`${isFullscreen ? 'max-w-3xl mx-auto px-8 py-20' : 'max-w-3xl mx-auto mt-4'}`}>
                                            {/* Top Sentinel */}
                                            {isFullscreen && (
                                                <div ref={topSentinelRef} className="h-40 flex items-center justify-center">
                                                    {isLoadingMore && <div className="animate-pulse text-brand-gold font-serif italic text-sm">Kinakarga ang nakaraang kabanata...</div>}
                                                </div>
                                            )}

                                            {modeError && (
                                                <div className="mb-8 p-4 bg-red-50 border border-red-200 rounded-md text-red-800 flex items-center justify-between">
                                                    <span className="text-sm">{modeError}</span>
                                                    <button onClick={() => setModeError(null)} className="text-sm font-bold underline hover:no-underline ml-4 whitespace-nowrap">Dismiss</button>
                                                </div>
                                            )}

                                            {(isFullscreen ? loadedChapters : [{ chapterNumber, title, content, isLoading }]).map((chap) => (
                                                <motion.div
                                                    key={`${chap.chapterNumber}`}
                                                    id={`chapter-view-${chap.chapterNumber}`} // ID for scroll spy
                                                    className={isFullscreen ? "mb-40 min-h-[80vh]" : ""}
                                                >
                                                    {/* Fullscreen Chapter Title (Keep as Separator) */}
                                                    {isFullscreen && (
                                                        <div className="mb-12 text-center opacity-80">
                                                            <span className="text-xs font-bold uppercase tracking-[0.2em] text-brand-text/30 block mb-2">Kabanata {chap.chapterNumber}</span>
                                                            <h2 className="text-3xl font-serif font-bold text-brand-navy">{chap.title}</h2>
                                                            <div className="w-8 h-0.5 bg-brand-gold/20 mx-auto mt-6" />
                                                        </div>
                                                    )}

                                                    <div className="font-serif text-brand-text leading-[2.2] text-justify space-y-8 text-lg md:text-xl">
                                                        {chap.content.reduce((paragraphs: ChapterContent[][], sentence) => {
                                                            if (paragraphs.length === 0 || paragraphs[paragraphs.length - 1].length >= 6) {
                                                                paragraphs.push([sentence]);
                                                            } else {
                                                                paragraphs[paragraphs.length - 1].push(sentence);
                                                            }
                                                            return paragraphs;
                                                        }, []).map((para, paraIdx) => (
                                                            <p key={`${paraIdx}-${showCharacters}-${showThemes}-${showReference}-${highlightSentenceIndex}`} className="first-letter:float-left first-letter:text-[3.5rem] first-letter:font-serif first-letter:font-bold first-letter:leading-[0.8] first-letter:mr-2 indent-0">
                                                                {para.map((sentence) => {
                                                                    const isHighlighted = !isFullscreen && sentence.sentence_index === highlightSentenceIndex;
                                                                    const themes = sentence.themes || [];
                                                                    const themeCount = themes.length;

                                                                    return (
                                                                        <span
                                                                            key={sentence.sentence_index}
                                                                            ref={isHighlighted ? highlightRef : null}
                                                                            className={`
                                                                                hover:bg-brand-gold/10 transition-all duration-200 rounded px-0.5 relative inline
                                                                                ${isHighlighted ? 'bg-brand-gold/20 font-bold border-b-2 border-brand-gold' : ''}
                                                                            `}
                                                                            id={`sentence-${sentence.sentence_index}`}
                                                                        >

                                                                            {highlightText(sentence.sentence_text)}

                                                                            {showThemes && themeCount > 0 && (
                                                                                <span
                                                                                    onClick={(e) => {
                                                                                        e.stopPropagation();
                                                                                        setSelectedSentenceForTheme(sentence.sentence_index);
                                                                                        setThemeFullscreen(true);
                                                                                    }}
                                                                                    className={`
                                                                                    inline-flex items-center justify-center 
                                                                                    w-[14px] h-[14px] rounded-full shadow-sm
                                                                                    cursor-pointer transition-transform ml-1 flex-shrink-0
                                                                                    ${themeCount === 1
                                                                                            ? 'bg-brand-gold hover:bg-brand-gold/80'
                                                                                            : 'bg-brand-navy hover:bg-brand-navy/80'}
                                                                                    text-white text-[8px] font-sans font-bold select-none leading-none z-10 p-0 indent-0
                                                                                    transform -translate-y-[6px] align-middle
                                                                                `}
                                                                                    title={`${themeCount} paksa sa pangungusap na ito`}
                                                                                >
                                                                                    {themeCount}
                                                                                </span>
                                                                            )}

                                                                            {showReference && (
                                                                                <sup className="text-brand-gold cursor-help ml-0.5 font-bold text-xs">
                                                                                    [0]
                                                                                </sup>
                                                                            )}
                                                                            {" "}
                                                                        </span>
                                                                    );
                                                                })}
                                                            </p>
                                                        ))}
                                                    </div>
                                                </motion.div>
                                            ))}

                                            {/* Bottom Sentinel */}
                                            {isFullscreen && (
                                                <div ref={bottomSentinelRef} className="h-40 flex items-center justify-center">
                                                    {isLoadingMore && <div className="animate-pulse text-brand-gold font-serif italic text-sm">Dinatlo ang susunod na kabanata...</div>}
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div >

                            {/* Navigation Footer (Non-Fullscreen) */}
                            {
                                !isFullscreen && (
                                    <div className="flex items-center justify-between p-4 border-t border-brand-gold/10 bg-brand-cream">
                                        <button
                                            onClick={handlePrevChapter}
                                            className="flex items-center gap-2 px-4 py-2 rounded-md bg-white text-brand-text hover:bg-brand-gold/10 transition-all border border-brand-gold/10"
                                        >
                                            <ChevronLeft size={18} />
                                            <span className="text-xs font-bold uppercase tracking-widest">Nakaraan</span>
                                        </button>
                                        <div className="hidden sm:block text-[10px] text-brand-accent/60 font-serif uppercase tracking-[0.3em] font-bold">
                                            Leksikal at Semantikong Pag-aaral
                                        </div>
                                        <button
                                            onClick={handleNextChapter}
                                            className="flex items-center gap-2 px-4 py-2 rounded-md bg-white text-brand-text hover:bg-brand-gold/10 transition-all border border-brand-gold/10"
                                        >
                                            <span className="text-xs font-bold uppercase tracking-widest">Susunod</span>
                                            <ChevronRight size={18} />
                                        </button>
                                    </div>
                                )
                            }
                        </motion.div>

                            {/* Bookmarks popup – separate overlay when icon is clicked */}
                            {showBookmarksPanel && !isFullscreen && (
                                <>
                                    <div
                                        className="fixed inset-0 z-[55]"
                                        aria-hidden="true"
                                        onClick={() => setShowBookmarksPanel(false)}
                                    />
                                    <motion.div
                                        initial={{ opacity: 0, y: -8 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="fixed left-8 top-24 z-[56] w-80 max-w-[calc(100vw-4rem)] rounded-2xl bg-white shadow-xl border border-brand-gold/20 overflow-hidden"
                                    >
                                        <div className="flex items-center justify-between px-4 py-3 border-b border-brand-gold/10 bg-brand-cream/80">
                                            <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.28em] text-brand-text/60">
                                                <Bookmark size={14} className="text-brand-gold" />
                                                <span>Bookmarks</span>
                                            </div>
                                            <button
                                                type="button"
                                                onClick={() => setShowBookmarksPanel(false)}
                                                className="p-1 rounded-full hover:bg-black/5 text-brand-text/50"
                                            >
                                                <X size={16} />
                                            </button>
                                        </div>
                                        <div className="max-h-64 overflow-y-auto p-3 space-y-1">
                                            {bookmarks.length === 0 ? (
                                                <p className="text-sm text-brand-text/50 italic py-4 text-center">
                                                    Walang bookmark. I-highlight ang salita at i-click ang Bookmark para mag-save.
                                                </p>
                                            ) : (
                                                bookmarks
                                                    .slice()
                                                    .sort((a, b) => a.sentenceIndex - b.sentenceIndex || a.createdAt - b.createdAt)
                                                    .map((b) => {
                                                        const words = b.selectedText.split(/\s+/);
                                                        const display = words.length > 5 ? `${words.slice(0, 5).join(" ")}......` : b.selectedText;
                                                        return (
                                                        <div
                                                            key={b.id}
                                                            className="group flex items-center gap-2 rounded-lg px-3 py-2 hover:bg-brand-gold/10 transition-colors"
                                                        >
                                                            <button
                                                                type="button"
                                                                onClick={() => {
                                                                    const el = document.getElementById(`sentence-${b.sentenceIndex}`);
                                                                    if (el) {
                                                                        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                                                    }
                                                                    setShowBookmarksPanel(false);
                                                                }}
                                                                className="flex-1 text-left text-sm leading-snug text-brand-text/80 hover:text-brand-navy truncate"
                                                                title={b.selectedText}
                                                            >
                                                                {display}
                                                            </button>
                                                            <button
                                                                type="button"
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    removeBookmark(b.id);
                                                                }}
                                                                className="shrink-0 text-xs font-medium text-brand-text/50 hover:text-red-600 transition-colors px-2 py-1"
                                                                title="Alisin ang bookmark"
                                                            >
                                                                Remove
                                                            </button>
                                                        </div>
                                                        );
                                                    })
                                            )}
                                        </div>
                                    </motion.div>
                                </>
                            )}
                        </div>

                        {/* Fullscreen Theme Explanation View */}
                        {
                            themeFullscreen && selectedSentence && (
                                <motion.div
                                    initial={{ opacity: 0 }}
                                    animate={{ opacity: 1 }}
                                    exit={{ opacity: 0 }}
                                    className="fixed inset-0 z-[60] bg-white flex flex-col"
                                >
                                    {/* Header */}
                                    <div className="border-b border-brand-gold/30 bg-brand-cream/50 py-6">
                                        <div className="max-w-7xl mx-auto px-6">
                                            <div className="text-center space-y-2">
                                                <h2 className="text-2xl font-serif text-brand-navy font-bold">
                                                    {book === "noli" ? "Noli Me Tangere" : "El Filibusterismo"}
                                                </h2>
                                                <p className="text-lg text-brand-text">
                                                    Chapter {chapterNumber}
                                                </p>
                                                <p className="text-base text-brand-text-light italic">
                                                    {title}
                                                </p>

                                                {/* Action Buttons */}
                                                <div className="flex items-center justify-center gap-3 pt-4">
                                                    <button
                                                        onClick={() => setShowCharacters(!showCharacters)}
                                                        className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all text-sm ${showCharacters ? 'bg-brand-gold text-white' : 'bg-white text-brand-text hover:bg-brand-gold/20'}`}
                                                    >
                                                        <Users size={14} />
                                                        Characters
                                                    </button>
                                                    <button
                                                        onClick={() => setShowThemes(!showThemes)}
                                                        className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all text-sm ${showThemes ? 'bg-brand-gold text-white' : 'bg-white text-brand-text hover:bg-brand-gold/20'}`}
                                                    >
                                                        <BookOpen size={14} />
                                                        Themes
                                                    </button>
                                                </div>
                                            </div>

                                            {/* Close Button */}
                                            <button
                                                onClick={() => {
                                                    setThemeFullscreen(false);
                                                    setSelectedSentenceForTheme(null);
                                                }}
                                                className="absolute top-6 right-6 p-2 hover:bg-black/5 rounded-full transition-colors"
                                            >
                                                <X size={24} className="text-brand-navy" />
                                            </button>
                                        </div>
                                    </div>

                                    {/* Content Area */}
                                    <div className="flex-1 overflow-hidden flex">
                                        {/* Left Side - Chapter Text */}
                                        <div className="flex-1 overflow-y-auto p-8">
                                            <div className="max-w-3xl mx-auto">
                                                <p className="font-serif text-brand-text leading-loose text-justify text-base">
                                                    {selectedSentence.sentence_text}
                                                </p>
                                            </div>
                                        </div>

                                        {/* Right Side - Theme Explanations */}
                                        <div className="w-96 border-l border-brand-gold/30 bg-brand-paper overflow-y-auto p-6">
                                            <h3 className="text-lg font-serif font-bold text-brand-navy mb-4">
                                                Themes in this Sentence
                                            </h3>

                                            {/* Theme Cards (Real Data) */}
                                            <div className="space-y-4">
                                                {selectedSentence.themes && selectedSentence.themes.length > 0 ? (
                                                    selectedSentence.themes.map((theme, idx) => (
                                                        <div key={idx} className="bg-white p-4 rounded-lg border border-brand-gold/20 shadow-sm hover:shadow-md transition-shadow">
                                                            <div className="flex items-start gap-3">
                                                                <div className={`
                                                            w-8 h-8 rounded-full flex items-center justify-center text-white font-bold text-sm flex-shrink-0
                                                            ${idx === 0 ? 'bg-gradient-to-br from-purple-500 to-pink-500' :
                                                                        idx === 1 ? 'bg-gradient-to-br from-blue-500 to-cyan-500' :
                                                                            'bg-gradient-to-br from-amber-500 to-orange-500'}
                                                        `}>
                                                                    {idx + 1}
                                                                </div>
                                                                <div>
                                                                    <h4 className="font-serif font-bold text-brand-navy mb-1">
                                                                        {theme.label}
                                                                    </h4>
                                                                    <p className="text-sm text-brand-text-light leading-relaxed">
                                                                        {theme.explanation}
                                                                    </p>
                                                                    <div className="mt-2 text-xs text-brand-text-light/70 font-mono">
                                                                        Relevance Score: {Math.round(theme.score * 100)}%
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    ))
                                                ) : (
                                                    <div className="p-4 text-center text-brand-text-light italic">
                                                        No themes found for this sentence.
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                    </div>
                )}
            </AnimatePresence>

            {/* Inline Character Profile Modal – opens when a highlighted name is clicked */}
            {
                selectedChar && (
                    <ItemModal
                        isOpen={isCharModalOpen}
                        onClose={handleCharModalClose}
                        title={selectedChar.name}
                        subtitle={selectedChar.role}
                        type="character"
                        chapterAppearances={charAppearances}
                        isLoading={charLoading}
                        selectedNovel={book === 'noli' ? 'noli' : 'fili'}
                        onNavigate={(navBook, navChapter) => {
                            handleCharModalClose();
                            onNavigate?.(navBook, navChapter);
                        }}
                        onSort={async (mode) => {
                            if (selectedChar) {
                                setCharSortBy(mode);
                                await fetchCharAppearances(selectedChar, mode);
                            }
                        }}
                        sortBy={charSortBy}
                    />
                )
            }
        </>
    );
}
