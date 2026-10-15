"use client";

import React from "react";
import { BookOpen, GraduationCap, UserCheck, Github, ExternalLink, Sparkles, Feather } from "lucide-react";

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="w-full bg-[#0F172A] text-[#F8FAFC] border-t border-[#C5A065]/35 pt-16 pb-12 mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Top Literary Quote Banner with Generous Vertical Breathing Room */}
        <div className="py-8 border-b border-[#C5A065]/25 text-center">
          <p className="font-serif italic text-lg md:text-xl text-[#F1F5F9] max-w-3xl mx-auto leading-relaxed tracking-wide">
            &quot;To foretell the destiny of a nation, it is necessary to open the book that tells of her past.&quot;
          </p>
          <span className="block mt-3 font-serif text-sm tracking-[0.2em] text-[#D4AF37] uppercase font-semibold">
            — Dr. José Rizal
          </span>
        </div>

        {/* Balanced 3-Column Grid with Equal Gutters and High Contrast */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-10 lg:gap-14 py-14">
          {/* Column 1: Platform Overview */}
          <div className="space-y-5">
            <div className="flex items-center gap-2">
              <span className="text-2xl font-serif font-black text-[#D4AF37] tracking-tight">
                RIZAL<span className="text-white font-bold ml-1.5">ENGINE</span>
              </span>
            </div>
            
            {/* Standard upright text (non-italic) with increased base size and high contrast */}
            <p className="text-sm sm:text-base text-[#CBD5E1] leading-relaxed">
              An advanced Natural Language Processing system for thematic search, sequence alignment, 
              and cross-referencing across José Rizal&apos;s literary masterworks: 
              <span className="text-white font-medium"> Noli Me Tangere</span> and 
              <span className="text-white font-medium"> El Filibusterismo</span>.
            </p>

            {/* Roomy technology pills with increased padding and spacing */}
            <div className="pt-2 flex flex-wrap gap-2.5">
              <span className="px-3.5 py-1.5 rounded-lg bg-[#1E293B] text-[#D4AF37] border border-[#C5A065]/30 text-xs font-medium shadow-sm">
                Next.js 15
              </span>
              <span className="px-3.5 py-1.5 rounded-lg bg-[#1E293B] text-[#D4AF37] border border-[#C5A065]/30 text-xs font-medium shadow-sm">
                FastAPI
              </span>
              <span className="px-3.5 py-1.5 rounded-lg bg-[#1E293B] text-[#D4AF37] border border-[#C5A065]/30 text-xs font-medium shadow-sm">
                XLM-RoBERTa (DAPT)
              </span>
              <span className="px-3.5 py-1.5 rounded-lg bg-[#1E293B] text-[#D4AF37] border border-[#C5A065]/30 text-xs font-medium shadow-sm">
                pgvector
              </span>
            </div>
          </div>

          {/* Column 2: Academic Attribution with Distinct Roles vs Names */}
          <div className="space-y-5">
            <div className="flex items-center gap-2 text-[#D4AF37] font-serif text-sm font-bold tracking-wider uppercase">
              <GraduationCap className="w-5 h-5 text-[#D4AF37]" />
              <span>Academic Attribution</span>
            </div>
            
            <div className="space-y-4">
              <div>
                <span className="text-xs font-bold uppercase tracking-widest text-[#D4AF37] block mb-1.5">
                  Authors & Researchers
                </span>
                <p className="text-white text-base font-semibold leading-snug">Marcus Kent R. Oliver</p>
                <p className="text-white text-base font-semibold leading-snug">Ian Kurby A. Placencia</p>
                <p className="text-white text-base font-semibold leading-snug">Dominic B. Vilog</p>
              </div>

              <div className="pt-1">
                <span className="text-xs font-bold uppercase tracking-widest text-[#D4AF37] block mb-1">
                  Thesis Adviser
                </span>
                <p className="text-white text-base font-semibold flex items-center gap-2">
                  <UserCheck className="w-4 h-4 text-[#D4AF37]" />
                  <span>Dr. Melvin A. Ballera</span>
                </p>
              </div>

              <div className="pt-1 text-[#CBD5E1] text-sm leading-relaxed border-t border-slate-700/60 pt-3">
                <p className="font-semibold text-white">Bachelor of Science in Computer Science</p>
                <p className="text-[#94A3B8]">Technological Institute of the Philippines - Manila</p>
              </div>
            </div>
          </div>

          {/* Column 3: Open Source & Resources */}
          <div className="space-y-5">
            <div className="flex items-center gap-2 text-[#D4AF37] font-serif text-sm font-bold tracking-wider uppercase">
              <BookOpen className="w-5 h-5 text-[#D4AF37]" />
              <span>Links & Resources</span>
            </div>

            <ul className="space-y-3.5 text-sm">
              <li>
                <a
                  href="https://github.com/mematello/Rizal-Thematic-Exploration"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2.5 text-[#E2E8F0] hover:text-[#D4AF37] transition-colors group"
                >
                  <Github className="w-4 h-4 text-[#D4AF37] group-hover:scale-110 transition-transform" />
                  <span className="font-medium">GitHub Repository</span>
                  <ExternalLink className="w-3.5 h-3.5 opacity-60 ml-auto group-hover:opacity-100" />
                </a>
              </li>
              <li>
                <a
                  href="https://huggingface.co/spaces/mematello/rizal-thematic-search-api"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2.5 text-[#E2E8F0] hover:text-[#D4AF37] transition-colors group"
                >
                  <Sparkles className="w-4 h-4 text-[#D4AF37] group-hover:scale-110 transition-transform" />
                  <span className="font-medium">Hugging Face API (ZeroGPU)</span>
                  <ExternalLink className="w-3.5 h-3.5 opacity-60 ml-auto group-hover:opacity-100" />
                </a>
              </li>
              <li>
                <a
                  href="https://huggingface.co/mematello/rizal-xlm-r-dapt"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2.5 text-[#E2E8F0] hover:text-[#D4AF37] transition-colors group"
                >
                  <Feather className="w-4 h-4 text-[#D4AF37] group-hover:scale-110 transition-transform" />
                  <span className="font-medium">Fine-tuned DAPT Model Hub</span>
                  <ExternalLink className="w-3.5 h-3.5 opacity-60 ml-auto group-hover:opacity-100" />
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Copyright & Disclaimer Bar with Increased Font Size and Clear Contrast */}
        <div className="pt-8 border-t border-[#C5A065]/25 flex flex-col sm:flex-row items-center justify-between text-sm text-[#94A3B8] gap-4">
          <p>
            © {currentYear} <span className="text-white font-semibold">RizalEngine</span>. All rights reserved.
          </p>
          <p className="text-center sm:text-right text-[#CBD5E1]">
            Technological Institute of the Philippines - Manila • College of Computer Studies
          </p>
        </div>
      </div>
    </footer>
  );
}
