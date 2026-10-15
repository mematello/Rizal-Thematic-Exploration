"use client";

import React from "react";
import { BookOpen, GraduationCap, UserCheck, Github, ExternalLink, Sparkles, Feather } from "lucide-react";

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="w-full bg-[#141C2B] text-[#E6E1D8] border-t border-[#C5A065]/30 pt-16 pb-12 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Top Literary Quote Banner */}
        <div className="pb-12 border-b border-[#C5A065]/20 text-center">
          <p className="font-serif italic text-base md:text-lg text-[#D8CFBC] max-w-3xl mx-auto leading-relaxed">
            &quot;To foretell the destiny of a nation, it is necessary to open the book that tells of her past.&quot;
          </p>
          <span className="block mt-2 font-serif text-xs md:text-sm tracking-widest text-[#C5A065] uppercase">
            — Dr. José Rizal
          </span>
        </div>

        {/* Main 3-Column Grid */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-10 lg:gap-12 py-12">
          {/* Column 1: Platform Overview (5 cols) */}
          <div className="md:col-span-5 space-y-4">
            <div className="flex items-center gap-2">
              <span className="text-xl font-serif font-black text-[#C5A065] tracking-tight">
                RIZAL<span className="text-white font-bold ml-1">ENGINE</span>
              </span>
            </div>
            <p className="text-sm text-[#A9B1BD] leading-relaxed pr-4">
              An advanced Natural Language Processing system for thematic search, sequence alignment, 
              and cross-referencing across José Rizal&apos;s masterworks: 
              <span className="text-[#E6E1D8] italic"> Noli Me Tangere</span> and 
              <span className="text-[#E6E1D8] italic"> El Filibusterismo</span>.
            </p>
            <div className="pt-2 flex flex-wrap gap-2 text-xs">
              <span className="px-2.5 py-1 rounded-md bg-[#1C273A] text-[#C5A065] border border-[#C5A065]/20">
                Next.js 15
              </span>
              <span className="px-2.5 py-1 rounded-md bg-[#1C273A] text-[#C5A065] border border-[#C5A065]/20">
                FastAPI
              </span>
              <span className="px-2.5 py-1 rounded-md bg-[#1C273A] text-[#C5A065] border border-[#C5A065]/20">
                XLM-RoBERTa (DAPT)
              </span>
              <span className="px-2.5 py-1 rounded-md bg-[#1C273A] text-[#C5A065] border border-[#C5A065]/20">
                pgvector
              </span>
            </div>
          </div>

          {/* Column 2: Academic & Institutional Attribution (4 cols) */}
          <div className="md:col-span-4 space-y-4">
            <div className="flex items-center gap-2 text-[#C5A065] font-serif text-sm font-semibold tracking-wider uppercase">
              <GraduationCap className="w-4 h-4" />
              <span>Academic Attribution</span>
            </div>
            
            <div className="space-y-3 text-xs">
              <div>
                <span className="text-[#8E9AAB] block uppercase tracking-wider text-[10px]">Authors & Researchers</span>
                <p className="text-[#F0EBE1] font-medium text-sm">Marcus Kent R. Oliver</p>
                <p className="text-[#F0EBE1] font-medium text-sm">Ian Kurby A. Placencia</p>
                <p className="text-[#F0EBE1] font-medium text-sm">Dominic B. Vilog</p>
              </div>

              <div className="pt-1">
                <span className="text-[#8E9AAB] block uppercase tracking-wider text-[10px]">Thesis Adviser</span>
                <p className="text-[#F0EBE1] font-medium text-sm flex items-center gap-1.5">
                  <UserCheck className="w-3.5 h-3.5 text-[#C5A065]" />
                  <span>Dr. Melvin A. Ballera</span>
                </p>
              </div>

              <div className="pt-1 text-[#A9B1BD] leading-normal">
                <p className="font-semibold text-[#D8CFBC]">Bachelor of Science in Computer Science</p>
                <p>Technological Institute of the Philippines - Manila</p>
              </div>
            </div>
          </div>

          {/* Column 3: Open Source & Resources (3 cols) */}
          <div className="md:col-span-3 space-y-4">
            <div className="flex items-center gap-2 text-[#C5A065] font-serif text-sm font-semibold tracking-wider uppercase">
              <BookOpen className="w-4 h-4" />
              <span>Links & Resources</span>
            </div>

            <ul className="space-y-2.5 text-xs">
              <li>
                <a
                  href="https://github.com/mematello/Rizal-Thematic-Exploration"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-[#C8D1DE] hover:text-[#C5A065] transition-colors"
                >
                  <Github className="w-4 h-4 text-[#C5A065]" />
                  <span>GitHub Repository</span>
                  <ExternalLink className="w-3 h-3 opacity-60 ml-auto" />
                </a>
              </li>
              <li>
                <a
                  href="https://huggingface.co/spaces/mematello/rizal-thematic-search-api"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-[#C8D1DE] hover:text-[#C5A065] transition-colors"
                >
                  <Sparkles className="w-4 h-4 text-[#C5A065]" />
                  <span>Hugging Face API</span>
                  <ExternalLink className="w-3 h-3 opacity-60 ml-auto" />
                </a>
              </li>
              <li>
                <a
                  href="https://huggingface.co/mematello/rizal-xlm-r-dapt"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-[#C8D1DE] hover:text-[#C5A065] transition-colors"
                >
                  <Feather className="w-4 h-4 text-[#C5A065]" />
                  <span>Fine-tuned DAPT Model</span>
                  <ExternalLink className="w-3 h-3 opacity-60 ml-auto" />
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Copyright & Disclaimer Bar */}
        <div className="pt-8 mt-4 border-t border-[#C5A065]/20 flex flex-col sm:flex-row items-center justify-between text-xs text-[#8E9AAB] gap-4">
          <p>
            © {currentYear} <span className="text-[#D8CFBC] font-medium">RizalEngine</span>. All rights reserved.
          </p>
          <p className="text-center sm:text-right text-[#A9B1BD]">
            Technological Institute of the Philippines - Manila • College of Computer Studies
          </p>
        </div>
      </div>
    </footer>
  );
}
