"use client";

import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Sun, Moon, Monitor } from "lucide-react";

interface ThemeToggleProps {
  className?: string;
  variant?: "icon" | "segmented";
}

export function ThemeToggle({ className = "", variant = "icon" }: ThemeToggleProps) {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div
        className={`w-9 h-9 rounded-full bg-brand-gold/10 border border-brand-gold/20 flex items-center justify-center opacity-70 ${className}`}
        aria-hidden="true"
      >
        <span className="w-4 h-4 rounded-full bg-brand-gold/30" />
      </div>
    );
  }

  const isDark = resolvedTheme === "dark";

  if (variant === "segmented") {
    return (
      <div
        className={`flex items-center bg-white/70 dark:bg-brand-paper/80 backdrop-blur-sm rounded-full p-1 border border-brand-gold/20 shadow-xs ${className}`}
        role="group"
        aria-label="Pagpili ng Tema"
      >
        <button
          type="button"
          onClick={() => setTheme("light")}
          className={`min-h-[44px] flex-1 justify-center px-3 py-2 text-xs font-semibold rounded-full flex items-center gap-1.5 transition-all ${
            theme === "light"
              ? "bg-brand-navy dark:bg-brand-gold text-white dark:text-[#0F141C] shadow-xs"
              : "text-brand-text/80 dark:text-foreground/80 hover:text-brand-text"
          }`}
          aria-pressed={theme === "light"}
          title="Maliwanag na tema"
        >
          <Sun size={14} />
          <span>Liwanag</span>
        </button>
        <button
          type="button"
          onClick={() => setTheme("dark")}
          className={`min-h-[44px] flex-1 justify-center px-3 py-2 text-xs font-semibold rounded-full flex items-center gap-1.5 transition-all ${
            theme === "dark"
              ? "bg-brand-navy dark:bg-brand-gold text-white dark:text-[#0F141C] shadow-xs"
              : "text-brand-text/80 dark:text-foreground/80 hover:text-brand-text"
          }`}
          aria-pressed={theme === "dark"}
          title="Madilim na tema"
        >
          <Moon size={14} />
          <span>Dilim</span>
        </button>
        <button
          type="button"
          onClick={() => setTheme("system")}
          className={`min-h-[44px] flex-1 justify-center px-3 py-2 text-xs font-semibold rounded-full flex items-center gap-1.5 transition-all ${
            theme === "system"
              ? "bg-brand-navy dark:bg-brand-gold text-white dark:text-[#0F141C] shadow-xs"
              : "text-brand-text/80 dark:text-foreground/80 hover:text-brand-text"
          }`}
          aria-pressed={theme === "system"}
          title="Awtomatikong tema ayon sa sistema"
        >
          <Monitor size={14} />
          <span>Sistema</span>
        </button>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className={`min-w-[40px] min-h-[40px] w-10 h-10 rounded-full flex items-center justify-center bg-white/80 dark:bg-brand-paper/90 border border-brand-gold/20 shadow-xs hover:border-brand-gold/40 text-brand-navy dark:text-brand-gold active:scale-95 transition-all focus:outline-hidden focus:ring-2 focus:ring-brand-gold/40 cursor-pointer ${className}`}
      aria-label={isDark ? "Lumipat sa maliwanag na tema" : "Lumipat sa madilim na tema"}
      title={isDark ? "Lumipat sa maliwanag na tema" : "Lumipat sa madilim na tema"}
    >
      {isDark ? (
        <Sun size={18} className="transition-transform duration-500 rotate-0 hover:rotate-45 text-[#E8C97A]" />
      ) : (
        <Moon size={18} className="transition-transform duration-500 -rotate-12 hover:rotate-0 text-brand-navy" />
      )}
    </button>
  );
}
