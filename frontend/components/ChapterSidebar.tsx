"use client";

import { useEffect, useRef, useState, useCallback, memo } from "react";
import { ChevronUp, ChevronDown } from "lucide-react";
import { useMediaQuery } from "@/hooks/useMediaQuery";

type Novel = "noli" | "fili";

interface ChapterSidebarProps {
  novel: Novel;
  children?: React.ReactNode;
}

interface Section {
  start: number;
  end: number;
}

interface MobilePillNavProps {
  sections: Section[];
  activeSectionStart: number;
  label: string;
  onGroupClick: (e: React.MouseEvent<HTMLAnchorElement>, start: number) => void;
  pillRefs: React.MutableRefObject<Map<string, HTMLElement>>;
  containerRef: React.RefObject<HTMLDivElement | null>;
}

/**
 * Stable module-level memoized component for mobile pill navigation.
 * Uses role="tablist" / role="tab" with safe end-padding for full reachability.
 */
const MobilePillNav = memo(function MobilePillNav({
  sections,
  activeSectionStart,
  label,
  onGroupClick,
  pillRefs,
  containerRef,
}: MobilePillNavProps) {
  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between mb-2 px-1">
        <span className="text-[11px] font-bold uppercase tracking-[0.22em] text-brand-text/60 font-serif">
          {label}
        </span>
        <span className="text-[10px] uppercase font-bold tracking-wider text-brand-gold font-serif">
          Piliin ang Saklaw
        </span>
      </div>

      <div className="relative">
        {/* Horizontal scroll container: relative positioned with proximity scroll snap */}
        <div
          ref={containerRef}
          role="tablist"
          aria-label="Mga saklaw ng Kabanata"
          className="relative overflow-x-auto pb-1.5 pt-0.5 no-scrollbar scroll-smooth [scroll-snap-type:x_proximity]"
          style={{ scrollPaddingInline: "1.25rem" }}
        >
          {/* Inner list: display flex, width max-content, with generous end-padding so Kabanata 61-64 is fully reachable */}
          <div className="flex w-max gap-2 px-1 pr-16">
            {sections.map(({ start, end }) => {
              const id = `kabanata-${start}`;
              const isActive = start === activeSectionStart;
              return (
                <a
                  key={id}
                  id={`section-pill-${start}`}
                  role="tab"
                  aria-selected={isActive}
                  ref={(el) => {
                    if (el) {
                      pillRefs.current.set(id, el);
                    } else {
                      pillRefs.current.delete(id);
                    }
                  }}
                  href={`#${id}`}
                  onClick={(e) => onGroupClick(e, start)}
                  className={`snap-start shrink-0 flex-none whitespace-nowrap min-h-[44px] min-w-[124px] px-4 py-2 rounded-full border text-xs font-serif font-bold uppercase tracking-wider flex items-center justify-center transition-colors select-none focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-gold ${
                    isActive
                      ? "bg-brand-navy text-white border-brand-navy shadow-md shadow-brand-navy/20 font-bold"
                      : "bg-white/90 text-brand-text/75 border-brand-gold/25 hover:border-brand-gold hover:text-brand-navy active:scale-95 shadow-xs"
                  }`}
                >
                  {end === start
                    ? `Kabanata ${start}`
                    : `Kabanata ${start}–${end}`}
                </a>
              );
            })}
          </div>
        </div>

        {/* Edge fade masks indicating horizontal scrollability */}
        <div className="pointer-events-none absolute left-0 top-0 bottom-0 w-4 bg-gradient-to-r from-brand-cream to-transparent" />
        <div className="pointer-events-none absolute right-0 top-0 bottom-0 w-4 bg-gradient-to-l from-brand-cream to-transparent" />
      </div>
    </div>
  );
});

export function ChapterSidebar({ novel, children }: ChapterSidebarProps) {
  const isDesktop = useMediaQuery("(min-width: 1024px)");

  const totalChapters = novel === "noli" ? 64 : 39;
  const sections: Section[] = Array.from(
    { length: Math.ceil(totalChapters / 4) },
    (_, idx) => {
      const start = idx * 4 + 1;
      const end = Math.min(start + 3, totalChapters);
      return { start, end };
    }
  );

  const [activeSectionStart, setActiveSectionStart] = useState<number>(
    sections[0]?.start ?? 1
  );
  const activeSectionStartRef = useRef<number>(sections[0]?.start ?? 1);

  // References for containers
  const sectionContainerRef = useRef<HTMLDivElement | null>(null);
  const desktopListRef = useRef<HTMLDivElement | null>(null);
  const mobileNavRef = useRef<HTMLDivElement | null>(null);
  const mobilePillsContainerRef = useRef<HTMLDivElement | null>(null);

  // Stable Map of pill elements keyed by groupId ('kabanata-1', 'kabanata-5', etc.)
  const pillRefs = useRef<Map<string, HTMLElement>>(new Map());

  // Scroll lock ref to prevent observer from fighting click smooth-scrolls
  const isScrollLocked = useRef(false);
  const scrollLockTimeout = useRef<NodeJS.Timeout | null>(null);

  // Desktop hover-to-scroll indicator states
  const autoScrollDirection = useRef<"up" | "down" | null>(null);
  const autoScrollFrame = useRef<number | null>(null);
  const [canScrollUp, setCanScrollUp] = useState(false);
  const [canScrollDown, setCanScrollDown] = useState(false);

  const label =
    novel === "noli" ? "Noli · Kabanata 1–64" : "El Fili · Kabanata 1–39";

  // Keep activeSectionStartRef in sync with state
  useEffect(() => {
    activeSectionStartRef.current = activeSectionStart;
  }, [activeSectionStart]);

  // Center active pill inside container (never touches page vertical scroll)
  const centerActivePill = useCallback(
    (start: number, smooth = true) => {
      const container = mobilePillsContainerRef.current;
      const pill = pillRefs.current.get(`kabanata-${start}`);
      if (!container || !pill) return;
      if (pill.offsetParent === null) return; // Guard against hidden element

      const prefersReduced = window.matchMedia(
        "(prefers-reduced-motion: reduce)"
      ).matches;
      const behavior = smooth && !prefersReduced ? "smooth" : "auto";

      // Relative math using getBoundingClientRect guarantees exact coordinates
      const containerRect = container.getBoundingClientRect();
      const pillRect = pill.getBoundingClientRect();

      const target =
        container.scrollLeft +
        (pillRect.left - containerRect.left) -
        container.clientWidth / 2 +
        pill.offsetWidth / 2;

      const maxScroll = Math.max(
        0,
        container.scrollWidth - container.clientWidth
      );
      const clamped = Math.min(Math.max(target, 0), maxScroll);

      container.scrollTo({
        left: clamped,
        behavior,
      });
    },
    []
  );

  // Dynamically compute and publish sticky offsets as CSS variables
  const updateStickyVariables = useCallback(() => {
    if (typeof window === "undefined") return { headerH: 68, pillNavH: 52 };
    const header = document.querySelector("header");
    const headerH = header ? Math.round(header.getBoundingClientRect().height) : 68;
    const pillNavH = mobileNavRef.current
      ? Math.round(mobileNavRef.current.getBoundingClientRect().height)
      : 52;

    document.documentElement.style.setProperty("--header-height", `${headerH}px`);
    document.documentElement.style.setProperty("--mobile-pill-nav-height", `${pillNavH}px`);

    const isMobile = window.innerWidth < 1024;
    const stickyMargin = isMobile ? headerH + pillNavH + 12 : headerH + 32;
    document.documentElement.style.setProperty("--chapter-scroll-margin", `${stickyMargin}px`);

    return { headerH, pillNavH };
  }, []);

  useEffect(() => {
    updateStickyVariables();
    const handleResize = () => updateStickyVariables();
    window.addEventListener("resize", handleResize, { passive: true });

    let ro: ResizeObserver | null = null;
    const header = document.querySelector("header");
    if (header && typeof ResizeObserver !== "undefined") {
      ro = new ResizeObserver(() => updateStickyVariables());
      ro.observe(header);
      if (mobileNavRef.current) ro.observe(mobileNavRef.current);
    }

    return () => {
      window.removeEventListener("resize", handleResize);
      ro?.disconnect();
    };
  }, [updateStickyVariables]);

  // Click handler: immediate active update, vertical scroll, pill centering on finish, scroll lock
  const handleGroupClick = (
    e: React.MouseEvent<HTMLAnchorElement>,
    start: number
  ) => {
    e.preventDefault();

    // 1. Immediately activate clicked group
    setActiveSectionStart(start);
    activeSectionStartRef.current = start;

    // 2. Lock scrollspy so observer cannot override clicked group during transit
    isScrollLocked.current = true;
    if (scrollLockTimeout.current) clearTimeout(scrollLockTimeout.current);

    // 3. Update URL hash
    if (window.location.hash !== `#kabanata-${start}`) {
      window.history.replaceState(null, "", `#kabanata-${start}`);
    }

    // 4. Start smooth vertical scroll to section
    const target = document.getElementById(`kabanata-${start}`);
    if (target) {
      const prefersReduced = window.matchMedia(
        "(prefers-reduced-motion: reduce)"
      ).matches;
      target.scrollIntoView({
        behavior: prefersReduced ? "auto" : "smooth",
        block: "start",
      });
    }

    // 5. Unlock scrollspy when smooth scroll completes and center the pill
    let finished = false;
    const finish = () => {
      if (finished) return;
      finished = true;
      isScrollLocked.current = false;
      window.removeEventListener("scrollend", finish);
      centerActivePill(start, true);
    };

    window.addEventListener("scrollend", finish, { once: true });
    scrollLockTimeout.current = setTimeout(finish, 800);
  };

  // Initial load with URL hash (e.g. /#kabanata-49)
  useEffect(() => {
    if (typeof window === "undefined") return;
    const hash = window.location.hash;
    if (!hash.startsWith("#kabanata-")) return;

    const chapterNum = parseInt(hash.replace("#kabanata-", ""), 10);
    if (isNaN(chapterNum)) return;

    const matchingSection = sections.find(
      (s) => chapterNum >= s.start && chapterNum <= s.end
    );
    const timer = setTimeout(() => {
      setActiveSectionStart(matchingSection.start);
      activeSectionStartRef.current = matchingSection.start;
      const target = document.getElementById(`kabanata-${matchingSection.start}`);
      if (target) {
        isScrollLocked.current = true;
        const prefersReduced = window.matchMedia(
          "(prefers-reduced-motion: reduce)"
        ).matches;
        target.scrollIntoView({
          behavior: prefersReduced ? "auto" : "smooth",
          block: "start",
        });
        centerActivePill(matchingSection.start, false);
        setTimeout(() => {
          isScrollLocked.current = false;
        }, 800);
      }
    }, 150);

    return () => clearTimeout(timer);
  }, [novel, sections, centerActivePill]);

  // IntersectionObserver Scrollspy for active state
  useEffect(() => {
    if (typeof window === "undefined") return;

    const intersectingMap = new Map<number, IntersectionObserverEntry>();

    const getStickyOffset = () => {
      const header = document.querySelector("header");
      const headerH = header ? header.getBoundingClientRect().height : 68;
      const isMobile = window.innerWidth < 1024;
      const pillNavH = mobileNavRef.current
        ? mobileNavRef.current.getBoundingClientRect().height
        : 52;
      return Math.round(headerH + (isMobile ? pillNavH : 0) + 12);
    };

    const stickyOffset = getStickyOffset();
    const rootMargin = `-${stickyOffset}px 0px -60% 0px`;

    const observer = new IntersectionObserver(
      (entries) => {
        if (isScrollLocked.current) return;

        entries.forEach((entry) => {
          const id = entry.target.id;
          const startNum = parseInt(id.replace("kabanata-", ""), 10);
          if (!isNaN(startNum)) {
            if (entry.isIntersecting) {
              intersectingMap.set(startNum, entry);
            } else {
              intersectingMap.delete(startNum);
            }
          }
        });

        if (intersectingMap.size === 0) return;

        const currentOffset = getStickyOffset();
        let bestStart = -1;

        // 1. Primary match: Section that currently spans the sticky detection line
        intersectingMap.forEach((entry, startNum) => {
          const rect = entry.boundingClientRect;
          if (rect.top <= currentOffset + 32 && rect.bottom > currentOffset) {
            bestStart = startNum;
          }
        });

        // 2. Fallback: Pick intersecting group whose top is closest to sticky offset line
        if (bestStart === -1) {
          let minDiff = Infinity;
          intersectingMap.forEach((entry, startNum) => {
            const diff = Math.abs(entry.boundingClientRect.top - currentOffset);
            if (diff < minDiff) {
              minDiff = diff;
              bestStart = startNum;
            }
          });
        }

        if (bestStart !== -1 && bestStart !== activeSectionStartRef.current) {
          activeSectionStartRef.current = bestStart;
          setActiveSectionStart(bestStart);
          if (window.location.hash !== `#kabanata-${bestStart}`) {
            window.history.replaceState(null, "", `#kabanata-${bestStart}`);
          }
        }
      },
      {
        root: null,
        rootMargin,
        threshold: [0, 0.1, 0.25, 0.5],
      }
    );

    // Observe all group section anchors
    sections.forEach(({ start }) => {
      const el = document.getElementById(`kabanata-${start}`);
      if (el) observer.observe(el);
    });

    // Handle scroll edge cases (top of chapters, bottom of page)
    const handleScrollEdgeCases = () => {
      if (isScrollLocked.current) return;

      // 1. Bottom of page edge case: Force last group active
      const isAtBottom =
        window.innerHeight + window.scrollY >=
        document.documentElement.scrollHeight - 60;
      if (isAtBottom) {
        const lastStart = sections[sections.length - 1]?.start;
        if (lastStart && activeSectionStartRef.current !== lastStart) {
          activeSectionStartRef.current = lastStart;
          setActiveSectionStart(lastStart);
          window.history.replaceState(null, "", `#kabanata-${lastStart}`);
        }
        return;
      }

      // 2. Above chapters section: Force first group active
      const sectionEl = sectionContainerRef.current;
      if (sectionEl) {
        const rect = sectionEl.getBoundingClientRect();
        const currentOffset = getStickyOffset();
        if (rect.top > currentOffset + 60) {
          const firstStart = sections[0]?.start ?? 1;
          if (activeSectionStartRef.current !== firstStart) {
            activeSectionStartRef.current = firstStart;
            setActiveSectionStart(firstStart);
            window.history.replaceState(null, "", `#kabanata-${firstStart}`);
          }
        }
      }
    };

    window.addEventListener("scroll", handleScrollEdgeCases, { passive: true });

    return () => {
      observer.disconnect();
      window.removeEventListener("scroll", handleScrollEdgeCases);
    };
  }, [sections, novel]);

  // Auto-scroll mobile pill horizontally when active group changes from scrolling (scrollspy)
  useEffect(() => {
    if (isScrollLocked.current) return;
    centerActivePill(activeSectionStart, true);
  }, [activeSectionStart, centerActivePill]);

  // Desktop hover-to-scroll helpers
  const handleDesktopMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const container = desktopListRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const y = e.clientY;
    const edgeZone = rect.height * 0.2;

    if (y < rect.top + edgeZone) {
      autoScrollDirection.current = "up";
    } else if (y > rect.bottom - edgeZone) {
      autoScrollDirection.current = "down";
    } else {
      autoScrollDirection.current = null;
    }
  };

  const stopAutoScroll = () => {
    autoScrollDirection.current = null;
  };

  const updateDesktopScrollState = () => {
    const container = desktopListRef.current;
    if (!container) {
      setCanScrollUp(false);
      setCanScrollDown(false);
      return;
    }
    const { scrollTop, scrollHeight, clientHeight } = container;
    setCanScrollUp(scrollTop > 4);
    setCanScrollDown(scrollTop + clientHeight < scrollHeight - 4);
  };

  // Auto-scroll active section link inside desktop sidebar
  useEffect(() => {
    if (!isDesktop) return;
    const container = desktopListRef.current;
    const activeLink = document.getElementById(
      `section-link-${activeSectionStart}`
    );
    if (!container || !activeLink) {
      return;
    }

    const itemTop = activeLink.offsetTop;
    const itemHeight = activeLink.clientHeight;
    const containerScrollTop = container.scrollTop;
    const containerHeight = container.clientHeight;

    if (itemTop < containerScrollTop + 30) {
      container.scrollTo({
        top: Math.max(0, itemTop - 20),
        behavior: "smooth",
      });
    } else if (itemTop + itemHeight > containerScrollTop + containerHeight - 30) {
      container.scrollTo({
        top: itemTop + itemHeight - containerHeight + 20,
        behavior: "smooth",
      });
    }
  }, [activeSectionStart, isDesktop]);

  // Animation loop for desktop hover auto-scroll
  useEffect(() => {
    if (!isDesktop) return;
    const step = () => {
      const container = desktopListRef.current;
      if (!container || !autoScrollDirection.current) {
        autoScrollFrame.current = requestAnimationFrame(step);
        updateDesktopScrollState();
        return;
      }

      const speed = 2.5;
      if (autoScrollDirection.current === "down") {
        container.scrollTop = Math.min(
          container.scrollTop + speed,
          container.scrollHeight - container.clientHeight
        );
      } else if (autoScrollDirection.current === "up") {
        container.scrollTop = Math.max(container.scrollTop - speed, 0);
      }

      updateDesktopScrollState();
      autoScrollFrame.current = requestAnimationFrame(step);
    };

    autoScrollFrame.current = requestAnimationFrame(step);
    return () => {
      if (autoScrollFrame.current != null) {
        cancelAnimationFrame(autoScrollFrame.current);
      }
    };
  }, [isDesktop]);

  return (
    <div
      ref={sectionContainerRef}
      id="chapters-section"
      className="mt-4 sm:mt-6 relative"
    >
      {/* Mobile Sticky Horizontal Pill Row: Rendered ONLY on mobile (< 1024px) */}
      {!isDesktop && (
        <div
          ref={mobileNavRef}
          className="sticky top-[var(--header-height,110px)] z-30 -mx-4 px-4 py-2.5 bg-brand-cream/95 backdrop-blur-md border-b border-brand-gold/20 shadow-xs transition-shadow duration-300"
        >
          <MobilePillNav
            sections={sections}
            activeSectionStart={activeSectionStart}
            label={label}
            onGroupClick={handleGroupClick}
            pillRefs={pillRefs}
            containerRef={mobilePillsContainerRef}
          />
        </div>
      )}

      {/* Main Flex Layout for Sidebar + Chapter Grid */}
      <div className="flex flex-col lg:flex-row gap-6 mt-4 lg:mt-0">
        {/* Desktop Sticky Sidebar: Rendered ONLY on desktop (>= 1024px) */}
        {isDesktop && (
          <aside className="w-56 shrink-0">
            <nav
              aria-label="Mga grupo ng kabanata"
              className="sticky top-[calc(var(--header-height,72px)+1.5rem)] max-h-[calc(100dvh-var(--header-height,72px)-3rem)] flex flex-col space-y-3"
            >
              <p className="text-[11px] font-bold uppercase tracking-[0.32em] text-brand-text/40 px-2 font-serif">
                {label}
              </p>

              <div className="relative flex-1 min-h-0">
                {canScrollUp && (
                  <div className="pointer-events-none absolute -top-3 left-0 right-0 flex justify-center text-brand-navy/40 z-10">
                    <ChevronUp className="h-4 w-4" />
                  </div>
                )}

                <div
                  ref={desktopListRef}
                  className="flex flex-col gap-1 overflow-y-auto max-h-[calc(100dvh-var(--header-height,72px)-6rem)] pr-2 no-scrollbar focus:outline-none"
                  tabIndex={-1}
                  onMouseMove={handleDesktopMouseMove}
                  onMouseLeave={stopAutoScroll}
                  onScroll={updateDesktopScrollState}
                >
                  {sections.map(({ start, end }) => {
                    const isActive = start === activeSectionStart;
                    return (
                      <a
                        key={start}
                        id={`section-link-${start}`}
                        href={`#kabanata-${start}`}
                        aria-current={isActive ? "true" : undefined}
                        onClick={(e) => handleGroupClick(e, start)}
                        className={`group relative flex items-center px-3 py-2 rounded-lg text-sm font-serif transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-gold select-none ${
                          isActive
                            ? "text-brand-navy font-bold bg-brand-gold/15 translate-x-1"
                            : "text-brand-text/50 [@media(hover:hover)]:hover:text-brand-navy [@media(hover:hover)]:hover:bg-brand-gold/5 [@media(hover:hover)]:hover:translate-x-1"
                        }`}
                      >
                        {/* Left Accent Bar */}
                        <span
                          className={`absolute left-0 top-1.5 bottom-1.5 w-1 rounded-full transition-all duration-200 ${
                            isActive
                              ? "bg-brand-navy scale-y-100 opacity-100"
                              : "bg-brand-gold/30 scale-y-0 opacity-0 group-hover:scale-y-75 group-hover:opacity-70"
                          }`}
                          aria-hidden="true"
                        />
                        <span className="truncate">
                          {end === start
                            ? `Kabanata ${start}`
                            : `Kabanata ${start}–${end}`}
                        </span>
                      </a>
                    );
                  })}
                </div>

                {canScrollDown && (
                  <div className="pointer-events-none absolute -bottom-3 left-0 right-0 flex justify-center text-brand-navy/40 z-10">
                    <ChevronDown className="h-4 w-4" />
                  </div>
                )}
              </div>
            </nav>
          </aside>
        )}

        {/* Chapter Grid Content Area */}
        <div className="flex-1 min-w-0">{children}</div>
      </div>
    </div>
  );
}
