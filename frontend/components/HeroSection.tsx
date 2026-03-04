"use client";

import { motion } from "framer-motion";
import { NovelBackground } from "@/components/NovelBackground";

interface HeroSectionProps {
    novel: "noli" | "fili" | "both";
}

interface HeroSectionProps {
    novel: "noli" | "fili" | "both";
}

const stats = [
    { label: "Lexical model usage", value: 40, color: "bg-[#C5A065]" },
    { label: "Semantic model usage", value: 30, color: "bg-brand-navy" },
    { label: "Lexical + Semantic combined", value: 70, color: "bg-fili-accent" },
];

/** Elegant Philippine Sun icon used as the floating centerpiece */
function PhilippineSunIcon() {
    return (
        <svg
            viewBox="0 0 120 120"
            xmlns="http://www.w3.org/2000/svg"
            className="w-full h-full"
        >
            <defs>
                <radialGradient id="sunIconGrad" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#E8C97A" />
                    <stop offset="60%" stopColor="#C5A065" />
                    <stop offset="100%" stopColor="#9B7A42" />
                </radialGradient>
                <radialGradient id="sunGlowIcon" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#C5A065" stopOpacity="0.6" />
                    <stop offset="100%" stopColor="#C5A065" stopOpacity="0" />
                </radialGradient>
                <filter id="iconGlow">
                    <feGaussianBlur stdDeviation="4" result="blur" />
                    <feMerge>
                        <feMergeNode in="blur" />
                        <feMergeNode in="SourceGraphic" />
                    </feMerge>
                </filter>
            </defs>

            {/* Outer glow halo */}
            <circle cx="60" cy="60" r="58" fill="url(#sunGlowIcon)" />

            {/* 8 sun rays — alternating thick/thin */}
            {Array.from({ length: 8 }).map((_, i) => {
                const angle = (i * 45 * Math.PI) / 180;
                const inner = 26;
                const outer = i % 2 === 0 ? 52 : 46;
                const tipWidth = i % 2 === 0 ? 4.5 : 2.5;
                const perpAngle = angle + Math.PI / 2;
                const ix = 60 + Math.cos(angle) * inner;
                const iy = 60 + Math.sin(angle) * inner;
                const ox = 60 + Math.cos(angle) * outer;
                const oy = 60 + Math.sin(angle) * outer;
                const side1x = ix + Math.cos(perpAngle) * tipWidth;
                const side1y = iy + Math.sin(perpAngle) * tipWidth;
                const side2x = ix - Math.cos(perpAngle) * tipWidth;
                const side2y = iy - Math.sin(perpAngle) * tipWidth;
                return (
                    <polygon
                        key={i}
                        points={`${side1x},${side1y} ${ox},${oy} ${side2x},${side2y}`}
                        fill="url(#sunIconGrad)"
                        filter="url(#iconGlow)"
                    />
                );
            })}

            {/* Main sun circle */}
            <circle cx="60" cy="60" r="22" fill="url(#sunIconGrad)" filter="url(#iconGlow)" />
            {/* Inner bright highlight */}
            <circle cx="60" cy="60" r="14" fill="#F0D490" opacity="0.6" />
            {/* Tiny center dot */}
            <circle cx="60" cy="60" r="5" fill="#E8C97A" />
        </svg>
    );
}

/** Ornamental book icon */
function BookIcon() {
    return (
        <svg viewBox="0 0 36 36" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
            <defs>
                <linearGradient id="bookGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#1B263B" />
                    <stop offset="100%" stopColor="#2E4057" />
                </linearGradient>
            </defs>
            <rect x="4" y="4" width="28" height="28" rx="3" fill="url(#bookGrad)" />
            <rect x="6" y="4" width="3" height="28" fill="#C5A065" opacity="0.8" />
            <rect x="11" y="10" width="16" height="1.5" rx="0.75" fill="#C5A065" opacity="0.6" />
            <rect x="11" y="14" width="12" height="1.5" rx="0.75" fill="#C5A065" opacity="0.4" />
            <rect x="11" y="18" width="14" height="1.5" rx="0.75" fill="#C5A065" opacity="0.4" />
            <rect x="11" y="22" width="10" height="1.5" rx="0.75" fill="#C5A065" opacity="0.3" />
        </svg>
    );
}

/** Small quill icon */
function QuillIcon() {
    return (
        <svg viewBox="0 0 36 36" xmlns="http://www.w3.org/2000/svg" className="w-full h-full">
            <path d="M28 4 Q36 14 16 30 Q12 26 18 18 Q22 12 28 4Z" fill="#8D2D2D" opacity="0.85" />
            <path d="M28 4 Q22 12 18 18 Q16 15 20 12 Q24 9 28 4Z" fill="#C5707070" opacity="0.5" />
            <line x1="16" y1="22" x2="12" y2="34" stroke="#1B263B" strokeWidth="1.5" strokeLinecap="round" />
            <path d="M11 33 L9 38 L14 36 Z" fill="#1B263B" />
        </svg>
    );
}

export function HeroSection({ novel }: HeroSectionProps) {
    const isFili = novel === "fili";

    return (
        <section className="relative w-full py-14 md:py-28 overflow-hidden">
            {/* Novel-themed illustrated background with dynamic mode */}
            <NovelBackground novel={novel === "both" ? "noli" : novel} />

            {/* Subtle vignette overlay to blend edges */}
            <div
                className="absolute inset-0 pointer-events-none"
                style={{
                    background:
                        "radial-gradient(ellipse 90% 80% at 50% 50%, transparent 40%, rgba(252,250,247,0.35) 100%)",
                }}
            />

            <div className="relative z-10 max-w-7xl mx-auto px-4 grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                {/* ===== LEFT: Text Content ===== */}
                <motion.div
                    initial={{ opacity: 0, x: -30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.7, ease: "easeOut" }}
                    className="flex flex-col space-y-6"
                >
                    {/* Badge */}
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.1 }}
                        className="inline-flex items-center space-x-2 bg-brand-gold/12 border border-brand-gold/25 px-4 py-1.5 rounded-full w-fit backdrop-blur-sm"
                    >
                        <span className="w-1.5 h-1.5 rounded-full bg-brand-gold animate-pulse" />
                        <span className="text-[10px] font-bold tracking-widest text-brand-gold uppercase">
                            Proyektong Pananaliksik
                        </span>
                    </motion.div>

                    {/* Main title with novel names highlighted */}
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.7, delay: 0.15 }}
                    >
                        {novel === "noli" && (
                            <h2 className="text-4xl md:text-5xl lg:text-6xl font-serif text-brand-navy leading-[1.08] font-black tracking-tight">
                                Tuklasin ang{" "}
                                <span
                                    className="relative inline-block"
                                    style={{
                                        background: "linear-gradient(135deg, #C5A065 0%, #9B7A42 60%, #C5A065 100%)",
                                        WebkitBackgroundClip: "text",
                                        WebkitTextFillColor: "transparent",
                                        backgroundClip: "text",
                                    }}
                                >
                                    Noli Me Tangere
                                </span>
                            </h2>
                        )}
                        {novel === "fili" && (
                            <h2 className="text-4xl md:text-5xl lg:text-6xl font-serif text-brand-navy leading-[1.08] font-black tracking-tight">
                                Tuklasin ang{" "}
                                <span
                                    className="relative inline-block italic"
                                    style={{
                                        color: "#8D2D2D",
                                        textShadow: "0 0 20px rgba(141,45,45,0.2)",
                                    }}
                                >
                                    El Filibusterismo
                                </span>
                            </h2>
                        )}
                        {novel === "both" && (
                            <h2 className="text-4xl md:text-5xl lg:text-6xl font-serif text-brand-navy leading-[1.08] font-black tracking-tight">
                                Tuklasin ang{" "}
                                <span
                                    className="relative inline-block"
                                    style={{
                                        background: "linear-gradient(135deg, #C5A065 0%, #9B7A42 60%, #C5A065 100%)",
                                        WebkitBackgroundClip: "text",
                                        WebkitTextFillColor: "transparent",
                                        backgroundClip: "text",
                                    }}
                                >
                                    Noli Me Tangere
                                </span>
                                <br />
                                <span
                                    className="text-3xl md:text-4xl lg:text-5xl font-light italic"
                                    style={{
                                        color: "rgba(141,45,45,0.7)",
                                    }}
                                >
                                    at El Filibusterismo
                                </span>
                            </h2>
                        )}
                    </motion.div>

                    {/* Thin gold rule */}
                    <motion.div
                        initial={{ scaleX: 0 }}
                        animate={{ scaleX: 1 }}
                        transition={{ duration: 0.7, delay: 0.3 }}
                        className="origin-left h-px w-40"
                        style={{
                            background:
                                "linear-gradient(to right, #C5A065, transparent)",
                        }}
                    />

                    {/* Description */}
                    <motion.p
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.6, delay: 0.35 }}
                        className="text-base md:text-lg text-brand-text/70 font-serif leading-relaxed max-w-xl"
                    >
                        Isang matalinong sistema ng paghahanap na nagsasama ng{" "}
                        <em>leksikal</em> at <em>semantikong</em> pag-unawa — upang mas
                        malalim na mapag-aralan ang mga tema, karakter, at kabanata ng
                        mga nobelang ito ni <strong className="text-brand-gold">Dr. José Rizal</strong>.
                    </motion.p>

                    {/* Novel title pills */}
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: 0.45 }}
                        className="flex items-center gap-3 flex-wrap"
                    >
                        <span
                            className={`px-4 py-1.5 rounded-full text-xs font-bold tracking-widest border transition-all duration-700 ${!isFili ? "bg-brand-gold/10 border-brand-gold/40 text-brand-gold shadow-[0_0_12px_rgba(197,160,101,0.2)]" : "bg-brand-gold/5 border-brand-gold/20 text-brand-gold/60"}`}
                        >
                            NOLI ME TANGERE · 1887
                        </span>
                        <span
                            className={`px-4 py-1.5 rounded-full text-xs font-bold tracking-widest border transition-all duration-700 ${isFili ? "bg-fili-accent/10 border-fili-accent/40 text-fili-accent shadow-[0_0_12px_rgba(141,45,45,0.2)]" : "bg-fili-accent/5 border-fili-accent/20 text-fili-accent/60"}`}
                        >
                            EL FILIBUSTERISMO · 1891
                        </span>
                    </motion.div>

                    {/* Icons row */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.5, delay: 0.55 }}
                        className="flex items-center space-x-4 pt-2"
                    >
                        <div className="flex -space-x-2">
                            {["bg-brand-gold/15", "bg-brand-navy/10", "bg-fili-accent/10", "bg-brand-gold/10"].map((bg, i) => (
                                <div
                                    key={i}
                                    className={`w-9 h-9 rounded-full border border-brand-gold/20 ${bg} flex items-center justify-center shadow-sm bg-white backdrop-blur-sm`}
                                >
                                    {i === 0 && (
                                        <svg viewBox="0 0 24 24" className="w-4 h-4 text-brand-gold" fill="none" stroke="currentColor" strokeWidth="1.5">
                                            <path d="M12 6.5c0 0-1.5-2-4-2a4 4 0 0 0 0 8c2.5 0 4-2 4-2s1.5 2 4 2a4 4 0 0 0 0-8c-2.5 0-4 2-4 2Z" />
                                        </svg>
                                    )}
                                    {i === 1 && (
                                        <svg viewBox="0 0 24 24" className="w-4 h-4 text-brand-navy" fill="none" stroke="currentColor" strokeWidth="1.5">
                                            <circle cx="12" cy="8" r="4" />
                                            <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" />
                                        </svg>
                                    )}
                                    {i === 2 && (
                                        <svg viewBox="0 0 24 24" className="w-4 h-4 text-fili-accent" fill="none" stroke="currentColor" strokeWidth="1.5">
                                            <path d="M12 2l2 7h7l-5.7 4.2 2.2 6.8L12 16l-5.5 4 2.2-6.8L3 9h7z" />
                                        </svg>
                                    )}
                                    {i === 3 && (
                                        <svg viewBox="0 0 24 24" className="w-4 h-4 text-brand-gold" fill="none" stroke="currentColor" strokeWidth="1.5">
                                            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                                            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                                        </svg>
                                    )}
                                </div>
                            ))}
                        </div>
                        <p className="text-xs font-semibold text-brand-text-light tracking-wider uppercase">
                            Mga Karakter · Tema · Kabanata
                        </p>
                    </motion.div>
                </motion.div>

                {/* ===== RIGHT: Floating Icon + Stats Card ===== */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.92 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.9, ease: "easeOut", delay: 0.2 }}
                    className="relative"
                >
                    {/* ── POLISHED FLOATING ICON ── */}
                    <div className="flex justify-center mb-8">
                        <div className="relative">
                            {/* Outer ambient glow ring */}
                            <motion.div
                                className="absolute inset-0 rounded-full"
                                animate={{
                                    background: isFili
                                        ? "radial-gradient(circle, rgba(141,45,45,0.15) 0%, transparent 60%)"
                                        : "radial-gradient(circle, rgba(197,160,101,0.35) 0%, transparent 70%)"
                                }}
                                transition={{ duration: 1.5 }}
                                style={{
                                    transform: "scale(1.8)",
                                    filter: "blur(16px)",
                                }}
                            />

                            {/* Rotating ring ornament */}
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
                                className="absolute inset-0 rounded-full"
                                style={{ transform: "scale(1.22)" }}
                            >
                                <svg viewBox="0 0 120 120" className="w-full h-full" style={{ width: "100%", height: "100%" }}>
                                    <circle
                                        cx="60" cy="60" r="56"
                                        fill="none"
                                        stroke={isFili ? "#A64D4D" : "#C5A065"}
                                        strokeWidth="0.8"
                                        strokeDasharray="4 8"
                                        opacity={isFili ? 0.6 : 0.45}
                                        className="transition-colors duration-1000"
                                    />
                                    {/* Decorative diamonds on ring */}
                                    {[0, 90, 180, 270].map((deg) => {
                                        const rad = (deg * Math.PI) / 180;
                                        const x = 60 + Math.cos(rad) * 56;
                                        const y = 60 + Math.sin(rad) * 56;
                                        return (
                                            <polygon
                                                key={deg}
                                                points={`${x},${y - 4} ${x + 3},${y} ${x},${y + 4} ${x - 3},${y}`}
                                                fill={isFili ? "#A64D4D" : "#C5A065"}
                                                opacity={isFili ? 0.7 : 0.55}
                                                className="transition-colors duration-1000"
                                            />
                                        );
                                    })}
                                </svg>
                            </motion.div>

                            {/* Main floating orb */}
                            <motion.div
                                animate={{ y: [0, -14, 0] }}
                                transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
                                className="relative w-36 h-36"
                                style={{
                                    filter: isFili
                                        ? "drop-shadow(0 20px 40px rgba(141,45,45,0.25)) drop-shadow(0 4px 12px rgba(141,45,45,0.1))"
                                        : "drop-shadow(0 20px 40px rgba(197,160,101,0.45)) drop-shadow(0 4px 12px rgba(0,0,0,0.18))",
                                    transition: "filter 1s ease"
                                }}
                            >
                                {/* Glassmorphism orb base */}
                                <motion.div
                                    className="absolute inset-0 rounded-full transition-colors duration-1000"
                                    style={{
                                        background: isFili
                                            ? "radial-gradient(circle at 38% 35%, rgba(255,245,245,0.95) 0%, rgba(240,230,230,0.9) 40%, rgba(220,200,200,0.85) 100%)"
                                            : "radial-gradient(circle at 38% 35%, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.72) 40%, rgba(240,228,205,0.85) 100%)",
                                        boxShadow: isFili
                                            ? "inset 0 2px 8px rgba(255,255,255,0.9), inset 0 -4px 12px rgba(141,45,45,0.15), 0 8px 32px rgba(141,45,45,0.1)"
                                            : "inset 0 2px 8px rgba(255,255,255,0.9), inset 0 -4px 12px rgba(197,160,101,0.25), 0 8px 32px rgba(197,160,101,0.3)",
                                        border: isFili
                                            ? "1.5px solid rgba(141,45,45,0.20)"
                                            : "1.5px solid rgba(197,160,101,0.30)",
                                    }}
                                />
                                {/* Specular highlight */}
                                <div
                                    className="absolute rounded-full"
                                    style={{
                                        top: "14%",
                                        left: "20%",
                                        width: "40%",
                                        height: "22%",
                                        background:
                                            "radial-gradient(ellipse, rgba(255,255,255,0.85) 0%, transparent 100%)",
                                        transform: "rotate(-20deg)",
                                    }}
                                />
                                {/* Philippine Sun icon */}
                                <div className="absolute inset-0 flex items-center justify-center p-7">
                                    <PhilippineSunIcon />
                                </div>
                            </motion.div>

                            {/* ── ORBITING MICRO-ICONS with Noli/Fili Transition ── */}
                            {/* Book icon — Orbts between top-right (Noli) and bottom-left (Fili) */}
                            <motion.div
                                animate={{
                                    rotate: isFili ? 180 : 0, // Orbit around center
                                }}
                                transition={{ duration: 1.2, ease: "easeInOut" }}
                                className="absolute inset-0 w-full h-full pointer-events-none"
                            >
                                <motion.div
                                    // Counter-rotate the icon itself so it stays upright
                                    animate={{ rotate: isFili ? -180 : 0 }}
                                    transition={{ duration: 1.2, ease: "easeInOut" }}
                                    className="absolute -top-3 -right-3 w-11 h-11 rounded-2xl flex items-center justify-center p-2.5 transition-all duration-1000"
                                    style={{
                                        background: isFili
                                            ? "linear-gradient(135deg, rgba(255,250,250,0.98) 0%, rgba(245,240,240,0.95) 100%)"
                                            : "linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(252,250,247,0.9) 100%)",
                                        boxShadow: isFili
                                            ? "0 4px 16px rgba(141,45,45,0.15), 0 1px 4px rgba(0,0,0,0.05)"
                                            : "0 4px 16px rgba(197,160,101,0.25), 0 1px 4px rgba(0,0,0,0.10)",
                                        border: isFili
                                            ? "1px solid rgba(141,45,45,0.15)"
                                            : "1px solid rgba(197,160,101,0.20)",
                                    }}
                                >
                                    <BookIcon />
                                </motion.div>
                            </motion.div>

                            {/* Quill icon — Orbits between bottom-left (Noli) and top-right (Fili) */}
                            <motion.div
                                animate={{
                                    rotate: isFili ? 180 : 0
                                }}
                                transition={{ duration: 1.2, ease: "easeInOut" }}
                                className="absolute inset-0 w-full h-full pointer-events-none"
                            >
                                <motion.div
                                    // Counter-rotate the icon itself so it stays upright
                                    animate={{ rotate: isFili ? -180 : 0 }}
                                    transition={{ duration: 1.2, ease: "easeInOut" }}
                                    className="absolute -bottom-2 -left-5 w-10 h-10 rounded-2xl flex items-center justify-center p-2 transition-all duration-1000"
                                    style={{
                                        background: isFili
                                            ? "linear-gradient(135deg, rgba(255,250,250,0.98) 0%, rgba(245,240,240,0.95) 100%)"
                                            : "linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(252,250,247,0.9) 100%)",
                                        boxShadow: isFili
                                            ? "0 4px 16px rgba(141,45,45,0.15), 0 1px 4px rgba(0,0,0,0.05)"
                                            : "0 4px 16px rgba(141,45,45,0.20), 0 1px 4px rgba(0,0,0,0.10)",
                                        border: isFili
                                            ? "1px solid rgba(141,45,45,0.15)"
                                            : "1px solid rgba(141,45,45,0.18)",
                                    }}
                                >
                                    <QuillIcon />
                                </motion.div>
                            </motion.div>

                            {/* Small sparkle — far right */}
                            <motion.div
                                animate={{ scale: [1, 1.3, 1], opacity: [0.6, 1, 0.6] }}
                                transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut", delay: 0.8 }}
                                className="absolute top-1/2 -right-8 -translate-y-1/2 w-8 h-8 flex items-center justify-center"
                            >
                                <svg viewBox="0 0 24 24" className="w-full h-full">
                                    <path
                                        d="M12 2 L13.5 10.5 L22 12 L13.5 13.5 L12 22 L10.5 13.5 L2 12 L10.5 10.5 Z"
                                        fill={isFili ? "#A64D4D" : "#C5A065"}
                                        opacity="0.7"
                                        className="transition-colors duration-1000"
                                    />
                                </svg>
                            </motion.div>
                        </div>
                    </div>

                    {/* ── STATS CARD ── */}
                    <div
                        className="rounded-3xl p-7 border overflow-hidden relative transition-all duration-1000"
                        style={{
                            background: isFili
                                ? "linear-gradient(145deg, rgba(242,240,237,0.95) 0%, rgba(235,230,225,0.90) 100%)"
                                : "linear-gradient(145deg, rgba(252,250,247,0.95) 0%, rgba(255,255,255,0.90) 100%)",
                            backdropFilter: "blur(12px)",
                            borderColor: isFili
                                ? "rgba(141,45,45,0.15)"
                                : "rgba(197,160,101,0.18)",
                            boxShadow: isFili
                                ? "0 8px 40px rgba(141,45,45,0.08), 0 2px 8px rgba(141,45,45,0.05)"
                                : "0 8px 40px rgba(27,38,59,0.08), 0 2px 8px rgba(197,160,101,0.10)",
                        }}
                    >
                        {/* Corner ornament */}
                        <div className="absolute top-0 right-0 w-24 h-24 opacity-5 pointer-events-none">
                            <svg viewBox="0 0 100 100" className="w-full h-full">
                                <path d="M100,0 Q60,0 60,40 Q60,100 100,100 L100,0Z" fill={isFili ? "#8D2D2D" : "#C5A065"} />
                            </svg>
                        </div>
                        {/* Decorative bg icon */}
                        <div className="absolute bottom-4 left-4 opacity-[0.04] pointer-events-none">
                            <svg viewBox="0 0 120 120" className="w-28 h-28">
                                {Array.from({ length: 8 }).map((_, i) => {
                                    const angle = (i * 45 * Math.PI) / 180;
                                    return (
                                        <line
                                            key={i}
                                            x1={60 + Math.cos(angle) * 28}
                                            y1={60 + Math.sin(angle) * 28}
                                            x2={60 + Math.cos(angle) * 52}
                                            y2={60 + Math.sin(angle) * 52}
                                            stroke={isFili ? "#8D2D2D" : "#C5A065"}
                                            strokeWidth={i % 2 === 0 ? "4" : "2"}
                                        />
                                    );
                                })}
                                <circle cx="60" cy="60" r="24" fill={isFili ? "#8D2D2D" : "#C5A065"} />
                            </svg>
                        </div>

                        <div className="relative z-10 space-y-6">
                            <div className="flex justify-between items-end">
                                <div>
                                    <h4 className={`text-base font-serif font-bold leading-tight transition-colors duration-1000 text-brand-navy`}>
                                        Kahusayan ng Modelo
                                    </h4>
                                    <p className={`text-xs mt-0.5 transition-colors duration-1000 text-brand-text-light`}>
                                        Model performance metrics
                                    </p>
                                </div>
                                <span
                                    className="text-[9px] font-black tracking-widest px-2 py-1 rounded-full transition-all duration-1000"
                                    style={{
                                        color: isFili ? "#A64D4D" : "#C5A065",
                                        background: isFili ? "rgba(141,45,45,0.08)" : "rgba(197,160,101,0.10)",
                                        border: isFili ? "1px solid rgba(141,45,45,0.15)" : "1px solid rgba(197,160,101,0.20)",
                                    }}
                                >
                                    LIVE DATA
                                </span>
                            </div>

                            <div className="space-y-5">
                                {stats.map((stat, idx) => (
                                    <div key={idx} className="space-y-2">
                                        <div className="flex justify-between text-sm">
                                            <span className={`font-medium transition-colors duration-1000 text-brand-text/80`}>{stat.label}</span>
                                            <span className={`font-bold transition-colors duration-1000 text-brand-navy`}>{stat.value}%</span>
                                        </div>
                                        <div className={`w-full h-1.5 rounded-full overflow-hidden transition-colors duration-1000 bg-brand-cream`}>
                                            <motion.div
                                                initial={{ width: 0 }}
                                                animate={{ width: `${stat.value}%` }}
                                                transition={{
                                                    duration: 1.6,
                                                    ease: "circOut",
                                                    delay: 0.6 + idx * 0.2,
                                                }}
                                                className={`h-full ${stat.color} rounded-full`}
                                                style={{
                                                    boxShadow: "0 0 8px rgba(0,0,0,0.15)",
                                                }}
                                            />
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {/* Novel labels at bottom */}
                            <div className={`flex gap-3 pt-1 border-t transition-colors duration-1000 ${isFili ? "border-brand-navy/10" : "border-brand-gold/10"}`}>
                                <div className="flex items-center gap-1.5">
                                    <div className="w-2 h-2 rounded-full bg-[#C5A065]" />
                                    <span className={`text-[10px] font-semibold uppercase tracking-widest transition-colors duration-1000 text-brand-text-light`}>
                                        Noli
                                    </span>
                                </div>
                                <div className="flex items-center gap-1.5">
                                    <div className="w-2 h-2 rounded-full bg-fili-accent" />
                                    <span className={`text-[10px] font-semibold uppercase tracking-widest transition-colors duration-1000 text-brand-text-light`}>
                                        Fili
                                    </span>
                                </div>
                                <div className="flex items-center gap-1.5">
                                    <div className="w-2 h-2 rounded-full bg-brand-navy" />
                                    <span className={`text-[10px] font-semibold uppercase tracking-widest transition-colors duration-1000 text-brand-text-light`}>
                                        Parehong Modelo
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </motion.div>
            </div>
        </section>
    );
}
