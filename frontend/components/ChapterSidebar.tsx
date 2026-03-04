"use client";

import { useEffect, useRef, useState } from "react";
import { Menu, X, ChevronUp, ChevronDown } from "lucide-react";

type Novel = "noli" | "fili";

interface ChapterSidebarProps {
  novel: Novel;
}

export function ChapterSidebar({ novel }: ChapterSidebarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const totalChapters = novel === "noli" ? 64 : 39;
  const sections = Array.from({ length: Math.ceil(totalChapters / 4) }, (_, idx) => {
    const start = idx * 4 + 1;
    const end = Math.min(start + 3, totalChapters);
    return { start, end };
  });
  const [activeSectionStart, setActiveSectionStart] = useState(
    sections[0]?.start ?? 1
  );
  const desktopListRef = useRef<HTMLDivElement | null>(null);
  const autoScrollDirection = useRef<"up" | "down" | null>(null);
  const autoScrollFrame = useRef<number | null>(null);
  const [canScrollUp, setCanScrollUp] = useState(false);
  const [canScrollDown, setCanScrollDown] = useState(false);
  const label =
    novel === "noli" ? "Noli · Kabanata 1–64" : "El Fili · Kabanata 1–39";

  const handleClick = (start: number) => {
    setActiveSectionStart(start);
    // Close drawer on mobile after navigation
    setIsOpen(false);
  };

  // Hover-based auto scroll for desktop list
  const handleDesktopMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const container = desktopListRef.current;
    if (!container) return;

    const rect = container.getBoundingClientRect();
    const y = e.clientY;
    const edgeZone = rect.height * 0.2; // top 20% or bottom 20%

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
    setCanScrollUp(scrollTop > 2);
    setCanScrollDown(scrollTop + clientHeight < scrollHeight - 2);
  };

  // Auto-detect active section based on scroll position
  useEffect(() => {
    const handleScroll = () => {
      const viewportMiddle = window.innerHeight * 0.35;

      let closestStart = activeSectionStart;
      let closestDistance = Infinity;

      sections.forEach(({ start }) => {
        const el = document.getElementById(`kabanata-${start}`);
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const distance = Math.abs(rect.top - viewportMiddle);
        if (distance < closestDistance) {
          closestDistance = distance;
          closestStart = start;
        }
      });

      if (closestStart !== activeSectionStart) {
        setActiveSectionStart(closestStart);
      }
    };

    handleScroll();
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, [sections, activeSectionStart]);

  // Animation loop for hover auto-scroll
  useEffect(() => {
    const step = () => {
      const container = desktopListRef.current;
      if (!container || !autoScrollDirection.current) {
        autoScrollFrame.current = requestAnimationFrame(step);
        updateDesktopScrollState();
        return;
      }

      const speed = 2; // pixels per frame
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
  }, []);

  // Keep the active section link in view inside the desktop list
  useEffect(() => {
    const container = desktopListRef.current;
    if (!container) {
      updateDesktopScrollState();
      return;
    }
    const activeEl = document.getElementById(
      `section-link-${activeSectionStart}`
    );
    if (!activeEl) {
      updateDesktopScrollState();
      return;
    }

    const containerRect = container.getBoundingClientRect();
    const itemRect = activeEl.getBoundingClientRect();

    if (itemRect.top < containerRect.top) {
      activeEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
    } else if (itemRect.bottom > containerRect.bottom) {
      activeEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }

    updateDesktopScrollState();
  }, [activeSectionStart]);

  return (
    <>
      {/* Mobile / small screens – top bar toggle */}
      <div className="mb-4 flex items-center justify-between lg:hidden">
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="inline-flex items-center gap-2 rounded-full bg-white/80 px-4 py-2 text-xs font-bold uppercase tracking-[0.2em] text-brand-text shadow-md shadow-black/5 ring-1 ring-black/5 hover:bg-white hover:shadow-lg transition-all"
        >
          <Menu className="h-4 w-4" />
          <span className="truncate max-w-[9rem]">{label}</span>
        </button>
      </div>

      {/* Mobile drawer */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/30 backdrop-blur-[1px] lg:hidden"
          onClick={() => setIsOpen(false)}
        >
          <nav
            className="absolute left-0 top-0 h-full w-64 bg-white/95 shadow-2xl flex flex-col backdrop-blur-md"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-black/5">
              <span className="text-[11px] font-bold tracking-[0.24em] uppercase text-brand-text/60">
                {novel === "noli" ? "Noli" : "El Fili"}
              </span>
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="rounded-full p-1.5 hover:bg-black/5 transition-colors text-brand-text/70"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
              {sections.map(({ start, end }) => (
                <a
                  key={start}
                  id={`section-link-${start}-mobile`}
                  href={`#kabanata-${start}`}
                  onClick={() => handleClick(start)}
                  className={`block rounded-full px-3 py-1.5 text-sm font-serif transition-colors ${
                    start === activeSectionStart
                      ? "bg-brand-gold/20 text-brand-navy font-semibold"
                      : "text-brand-text/30 hover:bg-brand-gold/10 hover:text-brand-navy"
                  }`}
                >
                  {end === start
                    ? `Kabanata ${start}`
                    : `Kabanata ${start}–${end}`}
                </a>
              ))}
            </div>
          </nav>
        </div>
      )}

      {/* Desktop / large screens – sticky sidebar */}
      <aside className="hidden lg:block w-56 shrink-0">
        {/* Stick to a fixed offset below the header/hero so it never overlaps the top image */}
        <div className="sticky top-40 space-y-3">
          <p className="text-[11px] font-bold uppercase tracking-[0.32em] text-brand-text/40">
            {label}
          </p>
          <div className="relative">
            {canScrollUp && (
              <div className="pointer-events-none absolute -top-4 left-0 right-0 flex justify-center text-brand-text/30">
                <ChevronUp className="h-4 w-4" />
              </div>
            )}
            <div
              ref={desktopListRef}
              className="flex max-h-[40vh] flex-col gap-1.5 overflow-y-auto pr-1 no-scrollbar"
              onMouseMove={handleDesktopMouseMove}
              onMouseLeave={stopAutoScroll}
              onScroll={updateDesktopScrollState}
            >
              {sections.map(({ start, end }) => (
                <a
                  key={start}
                  id={`section-link-${start}`}
                  href={`#kabanata-${start}`}
                  onClick={() => setActiveSectionStart(start)}
                  className={`block py-1.5 text-lg leading-relaxed font-serif transition-colors transition-transform ${
                    start === activeSectionStart
                      ? "text-brand-navy font-semibold translate-x-1"
                      : "text-brand-text/30 hover:text-brand-navy hover:translate-x-1"
                  }`}
                >
                  {end === start
                    ? `Kabanata ${start}`
                    : `Kabanata ${start}–${end}`}
                </a>
              ))}
            </div>
            {canScrollDown && (
              <div className="pointer-events-none absolute -bottom-4 left-0 right-0 flex justify-center text-brand-text/30">
                <ChevronDown className="h-4 w-4" />
              </div>
            )}
          </div>
        </div>
      </aside>
    </>
  );
}


