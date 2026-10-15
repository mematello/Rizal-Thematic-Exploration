"use client";

import React from "react";
import { ArrowUpRight } from "lucide-react";

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="w-full bg-[#121927] text-[#F8FAFC] border-t border-[#C5A065]/30 mt-16 sm:mt-24 safe-bottom">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-10 sm:pt-14 pb-10 sm:pb-12">
        {/* Integrated Literary Epigraph with Controlled Proximity */}
        <div className="max-w-2xl mb-8 sm:mb-12">
          <blockquote className="font-serif italic text-base sm:text-xl text-[#F1F5F9] leading-relaxed tracking-wide">
            &ldquo;To foretell the destiny of a nation, it is necessary to open the book that tells of her past.&rdquo;
          </blockquote>
          <cite className="not-italic block mt-2 sm:mt-3 text-xs font-serif tracking-[0.2em] text-[#D4AF37] uppercase font-bold">
            — Dr. José Rizal
          </cite>
        </div>

        {/* 3-Column Colophon Grid with Equal Gutters and Strict Flush Alignment */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 sm:gap-10 lg:gap-14 pb-10 sm:pb-12 border-b border-[#C5A065]/20">
          {/* Column 1: System Colophon */}
          <div className="flex flex-col space-y-4">
            <h2 className="text-xl font-serif font-black tracking-tight text-[#D4AF37]">
              RIZAL<span className="text-white ml-1.5">ENGINE</span>
            </h2>

            <p className="text-sm sm:text-base text-[#CBD5E1] leading-relaxed">
              A specialized Natural Language Processing system for thematic search, sequence alignment, 
              and passage cross-referencing across José Rizal&apos;s masterworks: 
              <span className="italic text-white font-medium"> Noli Me Tangere</span> and 
              <span className="italic text-white font-medium"> El Filibusterismo</span>.
            </p>

            {/* Solid, borderless, grounded technology pills flush with left text edge */}
            <div className="flex flex-wrap gap-2 pt-1">
              <span className="px-3 py-1.5 rounded-md bg-[#1E293B] text-[#D4AF37] text-xs font-semibold tracking-wide">
                Next.js 15
              </span>
              <span className="px-3 py-1.5 rounded-md bg-[#1E293B] text-[#D4AF37] text-xs font-semibold tracking-wide">
                FastAPI
              </span>
              <span className="px-3 py-1.5 rounded-md bg-[#1E293B] text-[#D4AF37] text-xs font-semibold tracking-wide">
                XLM-RoBERTa DAPT
              </span>
              <span className="px-3 py-1.5 rounded-md bg-[#1E293B] text-[#D4AF37] text-xs font-semibold tracking-wide">
                pgvector
              </span>
            </div>
          </div>

          {/* Column 2: Academic Attribution with Natural Proximity Spacing */}
          <div className="flex flex-col space-y-5">
            <div>
              <span className="text-xs font-bold uppercase tracking-[0.16em] text-[#D4AF37] block mb-2">
                Authors & Researchers
              </span>
              <div className="space-y-1 text-base font-semibold text-white">
                <p>Marcus Kent R. Oliver</p>
                <p>Ian Kurby A. Placencia</p>
                <p>Dominic B. Vilog</p>
              </div>
            </div>

            <div>
              <span className="text-xs font-bold uppercase tracking-[0.16em] text-[#D4AF37] block mb-1">
                Thesis Adviser
              </span>
              <p className="text-base font-semibold text-white">
                Dr. Melvin A. Ballera
              </p>
            </div>

            <div className="text-sm text-[#CBD5E1] leading-relaxed pt-1">
              <p className="font-semibold text-white">Bachelor of Science in Computer Science</p>
              <p className="text-[#94A3B8]">Technological Institute of the Philippines - Manila</p>
            </div>
          </div>

          {/* Column 3: Open Resources with Authentic Logos & Typography */}
          <div className="flex flex-col space-y-4">
            <span className="text-xs font-bold uppercase tracking-[0.16em] text-[#D4AF37]">
              Open Source & Documentation
            </span>

            <nav aria-label="Footer Resources" className="flex flex-col space-y-2.5 pt-1">
              <a
                href="https://github.com/mematello/Rizal-Thematic-Exploration"
                target="_blank"
                rel="noopener noreferrer"
                className="group min-h-[44px] flex items-center justify-between text-sm text-[#F1F5F9] hover:text-[#D4AF37] bg-white/[0.04] sm:bg-transparent hover:bg-white/[0.06] sm:hover:bg-transparent px-3 sm:px-0 py-2.5 sm:py-1 rounded-lg sm:rounded-none transition-colors"
                aria-label="GitHub Repository (opens in new tab)"
              >
                <div className="flex items-center gap-2.5">
                  {/* Official GitHub Monogram SVG */}
                  <svg className="w-4 h-4 fill-current text-white group-hover:text-[#D4AF37] transition-colors" viewBox="0 0 24 24" aria-hidden="true">
                    <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
                  </svg>
                  <span className="font-semibold">GitHub Repository</span>
                </div>
                <ArrowUpRight className="w-4 h-4 text-[#D4AF37] opacity-60 group-hover:opacity-100 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
              </a>

              <a
                href="https://huggingface.co/spaces/mematello/rizal-thematic-search-api"
                target="_blank"
                rel="noopener noreferrer"
                className="group min-h-[44px] flex items-center justify-between text-sm text-[#F1F5F9] hover:text-[#D4AF37] bg-white/[0.04] sm:bg-transparent hover:bg-white/[0.06] sm:hover:bg-transparent px-3 sm:px-0 py-2.5 sm:py-1 rounded-lg sm:rounded-none transition-colors"
                aria-label="Hugging Face API Space (opens in new tab)"
              >
                <div className="flex items-center gap-2.5">
                  <span className="text-sm select-none" aria-hidden="true">🤗</span>
                  <span className="font-semibold">Hugging Face API (ZeroGPU)</span>
                </div>
                <ArrowUpRight className="w-4 h-4 text-[#D4AF37] opacity-60 group-hover:opacity-100 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
              </a>

              <a
                href="https://huggingface.co/mematello/rizal-xlm-r-dapt"
                target="_blank"
                rel="noopener noreferrer"
                className="group min-h-[44px] flex items-center justify-between text-sm text-[#F1F5F9] hover:text-[#D4AF37] bg-white/[0.04] sm:bg-transparent hover:bg-white/[0.06] sm:hover:bg-transparent px-3 sm:px-0 py-2.5 sm:py-1 rounded-lg sm:rounded-none transition-colors"
                aria-label="DAPT Model Checkpoint on Hugging Face (opens in new tab)"
              >
                <div className="flex items-center gap-2.5">
                  <span className="text-sm select-none" aria-hidden="true">🧠</span>
                  <span className="font-semibold">DAPT Model Checkpoint</span>
                </div>
                <ArrowUpRight className="w-4 h-4 text-[#D4AF37] opacity-60 group-hover:opacity-100 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
              </a>
            </nav>
          </div>
        </div>

        {/* Clean, Grounded Copyright Bar */}
        <div className="pt-8 flex flex-col sm:flex-row items-start sm:items-center justify-between text-xs sm:text-sm text-[#94A3B8] gap-3">
          <p className="font-medium">
            © {currentYear} <span className="text-white font-semibold">RizalEngine</span>. Open research & literary exploration.
          </p>
          <p className="text-left sm:text-right text-[#CBD5E1]">
            Technological Institute of the Philippines - Manila • College of Computer Studies
          </p>
        </div>
      </div>
    </footer>
  );
}
