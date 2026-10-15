"use client";

import React from "react";
import { ArrowUpRight } from "lucide-react";

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="w-full bg-[#EAE3D2] text-[#2C2C2C] border-t border-[#D5CBB8] mt-24">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-12 pb-10">
        {/* Integrated Literary Epigraph / Quote */}
        <div className="max-w-2xl mb-12">
          <blockquote className="font-serif italic text-lg sm:text-xl text-[#1B263B] leading-relaxed">
            &ldquo;To foretell the destiny of a nation, it is necessary to open the book that tells of her past.&rdquo;
          </blockquote>
          <cite className="not-italic block mt-2 text-xs font-serif tracking-[0.18em] text-[#8C6D37] uppercase font-bold">
            — Dr. José Rizal
          </cite>
        </div>

        {/* 3-Column Colophon Grid with Equal Width & Flush-Left Alignment */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10 lg:gap-14 pb-12 border-b border-[#D5CBB8]">
          {/* Column 1: System Colophon */}
          <div className="flex flex-col space-y-4">
            <h2 className="text-xl font-serif font-black tracking-tight text-[#1B263B]">
              RIZAL<span className="text-[#8C6D37] ml-1">ENGINE</span>
            </h2>

            <p className="text-sm text-[#4A4A4A] leading-relaxed">
              A specialized Natural Language Processing system for thematic search, sequence alignment, 
              and passage cross-referencing across José Rizal&apos;s masterworks: 
              <span className="italic text-[#1B263B] font-medium"> Noli Me Tangere</span> and 
              <span className="italic text-[#1B263B] font-medium"> El Filibusterismo</span>.
            </p>

            {/* Solid, borderless, grounded technology pills flush with left text edge */}
            <div className="flex flex-wrap gap-2 pt-1">
              <span className="px-2.5 py-1 rounded bg-[#1B263B]/10 text-[#1B263B] text-xs font-semibold tracking-wide">
                Next.js 15
              </span>
              <span className="px-2.5 py-1 rounded bg-[#1B263B]/10 text-[#1B263B] text-xs font-semibold tracking-wide">
                FastAPI
              </span>
              <span className="px-2.5 py-1 rounded bg-[#1B263B]/10 text-[#1B263B] text-xs font-semibold tracking-wide">
                XLM-RoBERTa DAPT
              </span>
              <span className="px-2.5 py-1 rounded bg-[#1B263B]/10 text-[#1B263B] text-xs font-semibold tracking-wide">
                pgvector
              </span>
            </div>
          </div>

          {/* Column 2: Academic Attribution with Natural Proximity Spacing */}
          <div className="flex flex-col space-y-5">
            <div>
              <span className="text-[11px] font-bold uppercase tracking-[0.16em] text-[#8C6D37] block mb-2">
                Authors & Researchers
              </span>
              <div className="space-y-1 text-sm font-semibold text-[#1B263B]">
                <p>Marcus Kent R. Oliver</p>
                <p>Ian Kurby A. Placencia</p>
                <p>Dominic B. Vilog</p>
              </div>
            </div>

            <div>
              <span className="text-[11px] font-bold uppercase tracking-[0.16em] text-[#8C6D37] block mb-1">
                Thesis Adviser
              </span>
              <p className="text-sm font-semibold text-[#1B263B]">
                Dr. Melvin A. Ballera
              </p>
            </div>

            <div className="text-xs text-[#5A5A5A] leading-relaxed">
              <p className="font-semibold text-[#1B263B]">Bachelor of Science in Computer Science</p>
              <p>Technological Institute of the Philippines - Manila</p>
            </div>
          </div>

          {/* Column 3: Open Resources with Authentic Logos / Typography */}
          <div className="flex flex-col space-y-4">
            <span className="text-[11px] font-bold uppercase tracking-[0.16em] text-[#8C6D37]">
              Open Source & Documentation
            </span>

            <nav aria-label="Footer Resources" className="flex flex-col space-y-3">
              <a
                href="https://github.com/mematello/Rizal-Thematic-Exploration"
                target="_blank"
                rel="noopener noreferrer"
                className="group flex items-center justify-between text-sm text-[#1B263B] hover:text-[#8C6D37] transition-colors py-1"
              >
                <div className="flex items-center gap-2.5">
                  {/* Official GitHub Monogram SVG */}
                  <svg className="w-4 h-4 fill-current text-[#1B263B] group-hover:text-[#8C6D37] transition-colors" viewBox="0 0 24 24" aria-hidden="true">
                    <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
                  </svg>
                  <span className="font-semibold">GitHub Repository</span>
                </div>
                <ArrowUpRight className="w-4 h-4 text-[#8C6D37] opacity-60 group-hover:opacity-100 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
              </a>

              <a
                href="https://huggingface.co/spaces/mematello/rizal-thematic-search-api"
                target="_blank"
                rel="noopener noreferrer"
                className="group flex items-center justify-between text-sm text-[#1B263B] hover:text-[#8C6D37] transition-colors py-1"
              >
                <div className="flex items-center gap-2.5">
                  <span className="text-sm select-none" aria-hidden="true">🤗</span>
                  <span className="font-semibold">Hugging Face API (ZeroGPU)</span>
                </div>
                <ArrowUpRight className="w-4 h-4 text-[#8C6D37] opacity-60 group-hover:opacity-100 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
              </a>

              <a
                href="https://huggingface.co/mematello/rizal-xlm-r-dapt"
                target="_blank"
                rel="noopener noreferrer"
                className="group flex items-center justify-between text-sm text-[#1B263B] hover:text-[#8C6D37] transition-colors py-1"
              >
                <div className="flex items-center gap-2.5">
                  <span className="text-sm select-none" aria-hidden="true">🧠</span>
                  <span className="font-semibold">DAPT Model Checkpoint</span>
                </div>
                <ArrowUpRight className="w-4 h-4 text-[#8C6D37] opacity-60 group-hover:opacity-100 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
              </a>
            </nav>
          </div>
        </div>

        {/* Clean, Grounded Copyright Bar */}
        <div className="pt-8 flex flex-col sm:flex-row items-start sm:items-center justify-between text-xs text-[#5A5A5A] gap-3">
          <p className="font-medium">
            © {currentYear} RizalEngine. Open research & literary exploration.
          </p>
          <p className="text-left sm:text-right text-[#6B6B6B]">
            Technological Institute of the Philippines - Manila • College of Computer Studies
          </p>
        </div>
      </div>
    </footer>
  );
}
